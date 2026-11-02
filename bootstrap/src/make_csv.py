# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import csv
import time
from urllib.parse import quote

from pytz import timezone
import argparse
import requests
from azure.cosmos import CosmosClient
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_utils import (
    format_timestamp_pst,
    parse_timestamp_pst,
)

import importlib.util

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'orcasite_feeds.py'))
spec = importlib.util.spec_from_file_location("orcasite_feeds", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load orcasite_feeds from {module_path}")
orcasite_feeds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orcasite_feeds)
OrcasiteFeed = orcasite_feeds.OrcasiteFeed
get_orcasite_feeds = orcasite_feeds.get_orcasite_feeds

COSMOS_URL = os.environ.get("COSMOS_URL", "").strip() or "https://aifororcasmetadatastore.documents.azure.com:443/"
COSMOS_KEY = os.environ.get("COSMOS_KEY", "<your-primary-key>")
COSMOS_DB = os.environ.get("COSMOS_DB", "predictions")
COSMOS_CONTAINER = os.environ.get("COSMOS_CONTAINER", "metadata")

@dataclass
class OrcasiteDetection:
    id: str
    feed: OrcasiteFeed
    timestamp: datetime          # center or start time (we'll define)
    source: str                  # "machine" | "human"
    category: str                # e.g., "whale", "vessel", "none"
    description: str             # free text; may be ""
    idempotency_key: str

@dataclass
class OrcaHelloDetection:
    id: str
    feed: OrcasiteFeed
    timestamp: Optional[datetime]
    status: str                  # e.g., "confirmed", "rejected", etc.
    confidence: Optional[float] = None  # Confidence score from model inference
    comments: str = ""
    tags: str = ""
    modelId: Optional[str] = None

# Terms in a detection description that indicate the label cannot be determined with confidence.
SKIP_TERMS = {'?', 'not sure', 'unsure', 'possibl', 'sounds like', 'sounded like', 'ould be'}

def get_label(
    orcasite_det: OrcasiteDetection,
    orcahello_det: Optional[OrcaHelloDetection],
) -> Optional[str]:
    """
    Derives a normalized label for an OrcasiteDetection based on its category, description, and an optional OrcaHelloDetection match.

    Parameters:
        orcasite_det (OrcasiteDetection): The detection to label.
        orcahello_det (Optional[OrcaHelloDetection]): An optional id-matched OrcaHello detection used to upgrade some whale labels.

    Returns:
        str | None: `'resident'`, `'transient'`, `'humpback'`, `'water'`, `'vessel'`, `'jingle'`, `'human'`, or `'bird'` when a label can be determined; `None` when the label is unknown.
    """
    desc = (orcasite_det.description or "").lower()
    cat = (orcasite_det.category or "").lower()

    # Skip detections whose description indicates the label cannot be determined.
    if any(term in desc for term in SKIP_TERMS):
        return None

    # 1. Resident.
    if cat == "whale":
        if ("resident" in desc) or ("pod" in desc):
            return "resident"
        if orcahello_det and orcahello_det.status.lower() == "confirmed":
            return "resident"

    # 2. Transient.
    if cat == "whale":
        if ("bigg" in desc) or ("transient" in desc):
            return "transient"

    # 3. Humpback.
    if cat == "whale":
        if "humpback" in desc:
            return "humpback"

    if orcahello_det and orcahello_det.status.lower() == "unreviewed":
        return None

    # 4. Other.
    if orcasite_det.source == "machine" and not ("click" in desc or "call" in desc or "whistle" in desc):
        if "human" in desc or "radio" in desc:
            return "human"
        if "vessel" in desc or "ship" in desc or "boat" in desc or "train" in desc:
            return "vessel"
        if "jingl" in desc:
            return "jingle"
        if "bird" in desc or "pigu" in desc or "keir" in desc:
            return "bird"
        if "water" in desc:
            return "water"

    # Unknown.
    return None

NEAR_MIN = timedelta(minutes=10)
PACIFIC_TZ = timezone('US/Pacific')
UTC_TZ = timezone('UTC')
MAX_DETECTION_PAGES = 1000  # Safety limit to prevent infinite loops (500k detections max)
DETECTION_FETCH_MAX_ATTEMPTS = 3
DETECTION_FETCH_RETRY_DELAY_SECONDS = 2

def is_isolated_human_whale(
    det: OrcasiteDetection,
    all_detections: List[OrcasiteDetection],
) -> bool:
    """
    True if there is NO machine whale detection at same feed within NEAR_MIN minutes.
    """
    for d in all_detections:
        if d.feed.id != det.feed.id:
            continue
        if d.source != "machine":
            continue
        if (d.category or "").lower() != "whale":
            continue
        if abs(d.timestamp - det.timestamp) <= NEAR_MIN:
            return False
    return True

def find_nearby_machine_detection(
    det: OrcasiteDetection,
    all_detections: List[OrcasiteDetection],
) -> Optional[OrcasiteDetection]:
    """
    Returns the first machine whale detection at the same feed within NEAR_MIN minutes, or None.

    Parameters:
        det (OrcasiteDetection): The human detection to find a nearby machine match for.
        all_detections (List[OrcasiteDetection]): All detections for the same feed.

    Returns:
        OrcasiteDetection | None: The nearest machine whale detection within NEAR_MIN, or None if not found.
    """
    for d in all_detections:
        if d.feed.id != det.feed.id:
            continue
        if d.source != "machine":
            continue
        if (d.category or "").lower() != "whale":
            continue
        if abs(d.timestamp - det.timestamp) <= NEAR_MIN:
            return d
    return None

@dataclass
class Classification:
    kind: str       # 'tp_human_only', 'tp_machine_only', 'fp_machine_only', 'tp_both', 'skip'
    include: bool

def classify_detection(
    det: OrcasiteDetection,
    label: str,
    all_detections: List[OrcasiteDetection],
) -> Classification:
    """
    Assigns a Classification to an OrcasiteDetection based on its label, source, category, and nearby detections.

    Parameters:
        det (OrcasiteDetection): The detection to classify.
        label (str): The computed label for the detection (e.g., 'resident', 'transient', 'humpback', 'water', 'vessel', 'jingle', 'human', or None).
        all_detections (List[OrcasiteDetection]): All detections for the same feed used to determine temporal context.

    Returns:
        Classification: An object with `kind` set to one of:
            - 'tp_human_only' for human-sourced whale detections with no nearby machine whale detections,
            - 'tp_machine_only' for machine-sourced whale detections,
            - 'fp_machine_only' for machine-sourced non-whale detections,
            - 'tp_both' for human-sourced whale detections with a nearby machine whale detection,
            - 'skip' for detections that should be excluded.
        The `include` field indicates whether the detection should be included in downstream processing (`true` if included, `false` otherwise).
    """
    cat = (det.category or "").lower()
    src = (det.source or "").lower()

    # Skip non-whale human / non-machine things already filtered by label logic.
    if label is None:
        return Classification(kind="skip", include=False)

    # False positive, machine-only (label 'human', 'vessel', 'jingle', 'bird', or 'water').
    if (label == "human" or label == "vessel" or label == "jingle" or label == "water" or label == "bird") and src == "machine":
        return Classification(kind="fp_machine_only", include=True)

    # True positive, human-only (include).
    if cat == "whale" and src == "human":
        if is_isolated_human_whale(det, all_detections):
            return Classification(kind="tp_human_only", include=True)
        else:
            # There is a nearby machine whale detection → not human-only
            return Classification(kind="tp_both", include=True)

    # True positive, machine-only (maybe include).
    if cat == "whale" and src == "machine":
        # You can gate this on some flag if 'maybe include' is optional.
        return Classification(kind="tp_machine_only", include=True)

    # False positive, human-only and true negative are effectively already skipped.
    return Classification(kind="skip", include=False)

def get_orcasite_detections(
    feed: OrcasiteFeed,
    max_attempts: int = DETECTION_FETCH_MAX_ATTEMPTS,
    retry_delay_seconds: int = DETECTION_FETCH_RETRY_DELAY_SECONDS,
) -> List[OrcasiteDetection]:
    """
    Fetch detections from the Orcasite API for the given feed.

    Parses API response into a list of OrcasiteDetection objects filtered to the provided feed. Timestamps are parsed from ISO strings and will be None if parsing fails. On network or parsing errors the function prints an error and returns an empty list.

    Parameters:
        feed (OrcasiteFeed): Feed whose detections should be retrieved and matched by feed.id.

    Returns:
        List[OrcasiteDetection]: Detections associated with the specified feed.

    Raises:
        ValueError: If ``max_attempts`` is less than 1.
        requests.exceptions.ReadTimeout: If the Orcasite API read times out on every retry attempt.
        Exception: Re-raises non-timeout request/parsing errors immediately.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be >= 0")

    # Base endpoint.
    base_url = "https://live.orcasound.net/api/json/detections"

    # Fields we want (already URL-encoded in your example).
    fields = (
        "id,playlist_timestamp,player_offset,timestamp,description,"
        "source,category,feed_id,idempotency_key"
    )

    # Build query parameters.
    limit = 250
    params = {
        "page[limit]": limit,
        "fields[detection]": fields,
        "filter[feed_id]": feed.id,
    }

    for attempt in range(1, max_attempts + 1):
        dets = []
        page_count = 0
        offset = 0

        try:
            # Loop through all pages until no more data is returned.
            while page_count < MAX_DETECTION_PAGES:
                params["page[offset]"] = offset

                print(f"Fetching Orcasite page {page_count}...")
                r = requests.get(base_url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()

                items = data.get("data", [])

                # If no data is returned, we've fetched all pages.
                if not items:
                    print(f"Finished after page {page_count}")
                    break

                for item in items:
                    attrs = item.get("attributes", {})

                    # Parse timestamp safely.
                    ts_raw = attrs.get("timestamp")
                    try:
                        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    except Exception:
                        ts = None

                    det = OrcasiteDetection(
                        id=item.get("id"),
                        feed=feed,
                        timestamp=ts,
                        source=attrs.get("source"),
                        category=attrs.get("category"),
                        description=attrs.get("description") or "",
                        idempotency_key=attrs.get("idempotency_key") or "",
                    )

                    dets.append(det)

                # Increment offset for next page.
                offset += limit
                page_count += 1

            return dets

        except requests.exceptions.ReadTimeout as exc:
            if attempt == max_attempts:
                print(f"Error fetching detections for feed {feed.id} after {max_attempts} attempts: {exc}")
                raise
            print(
                f"Read timeout fetching detections for feed {feed.id} "
                f"(attempt {attempt}/{max_attempts}); retrying in {retry_delay_seconds} seconds."
            )
            time.sleep(retry_delay_seconds)
        except Exception as e:
            print(f"Error fetching detections for feed {feed.id}: {e}")
            raise

def get_node_name_for_feed(feed: OrcasiteFeed) -> str:
    """
    Retrieve the node name associated with an OrcasiteFeed.

    Returns:
        node_name (str): The feed's node_name.
    """
    return feed.node_name

def get_orcahello_detections(feed: OrcasiteFeed) -> List[OrcaHelloDetection]:
    """
    Retrieve OrcaHello detections and return those whose audio URI contains the given feed's node_name.

    Parameters:
        feed (OrcasiteFeed): Feed whose node_name is used to filter OrcaHello detection audio URIs.

    Returns:
        List[OrcaHelloDetection]: A list of detections associated with the feed. Each detection's `timestamp` is a `datetime` or `None` if parsing failed, and `status` is one of `"confirmed"`, `"rejected"`, or `"unreviewed"`.
    """
    node_name = get_node_name_for_feed(feed)      # e.g., "rpi_sunset_bay"

    # Build Cosmos DB container client on demand so that importing this module
    # does not require a valid COSMOS_KEY (which matters for unit tests and
    # local development without Cosmos credentials).
    cosmos_client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)
    cosmos_database = cosmos_client.get_database_client(COSMOS_DB)
    container = cosmos_database.get_container_client(COSMOS_CONTAINER)

    # Cosmos DB SQL query.
    query = """
        SELECT * FROM c
        WHERE CONTAINS(c.audioUri, @node_name)
        ORDER BY c.timestamp DESC
    """

    params = [
        {"name": "@node_name", "value": node_name}
    ]

    # Cosmos DB returns an iterator that handles pagination internally.
    items = container.query_items(
        query=query,
        parameters=params,
        enable_cross_partition_query=True
    )

    results = []

    for item in items:
        found = item.get("SRKWFound", "").lower()
        reviewed = item.get("reviewed", False)

        if reviewed and found == "yes":
            status = "confirmed"
        elif reviewed and found == "no":
            status = "rejected"
        else:
            status = "unreviewed"

        # Parse timestamp.
        ts_raw = item.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            ts = None

        # Extract confidence if available.
        confidence = item.get("whaleFoundConfidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = None

        det = OrcaHelloDetection(
            id=item.get("id"),
            feed=feed,
            timestamp=ts,
            status=status,
            confidence=confidence,
            comments=(item.get("comments") or "").strip(),
            tags=(item.get("tags") or "").strip()
        )

        results.append(det)

    return results


def generate_uri(node: str, dt: datetime) -> str:
    """
    Generate a URI for the Orcasound bouts interface.

    Parameters:
        node (str): The node name (e.g., "andrews-bay").
        dt (datetime): The datetime in UTC.

    Returns:
        str: URI in the format "https://live.orcasound.net/bouts/new/{node}?time={utc_time}".
    """
    # Ensure the datetime is in UTC.
    utc_dt = dt.astimezone(UTC_TZ)
    # Format as ISO 8601 with milliseconds and Z suffix.
    time_str = utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    # URL encode the time parameter.
    time_encoded = quote(time_str, safe='')
    return f"https://live.orcasound.net/bouts/new/{node}?time={time_encoded}"

def process_all_feeds(
    output_root: Path,
    feed_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
):
    """
    Generate a consolidated CSV file of selected Orcasite detections, optionally filtered by feed
    and matched with corresponding OrcaHello detections.

    This function retrieves all available Orcasite feeds, optionally filters them by
    a specific node name, fetches detections from both Orcasite and OrcaHello for each
    feed, matches detections in time, classifies them, and writes the resulting
    detections to a CSV file named ``detections.csv`` under the given ``output_root``.

    Parameters:
        output_root (Path): Directory in which the output CSV file ``detections.csv`` will be
            created. The directory (and any missing parents) will be created if it does not exist.
        feed_filter (str | None, optional): If provided, only feeds whose ``node_name`` matches
            this value are processed. If no feed matches the given name, the function logs a
            message and returns without creating a CSV file. Defaults to ``None`` (process all feeds).
        start_time (datetime | None, optional): If provided, only detections with a timestamp
            >= start_time are included. Defaults to ``None`` (no lower bound).
        end_time (datetime | None, optional): If provided, only detections with a timestamp
            <= end_time are included. Defaults to ``None`` (no upper bound).
    """
    feeds = get_orcasite_feeds()

    if feed_filter:
        feeds = [f for f in feeds if f.node_name == feed_filter]
        if not feeds:
            print(f"No feed found with node_name '{feed_filter}'")
            return

    # Collect all detections first before writing to CSV.
    all_rows = []

    for feed in feeds:
        print(f"Processing feed {feed.id} ({feed.node_name})")
        orcasite_dets = get_orcasite_detections(feed)
        orcahello_dets = get_orcahello_detections(feed)

        for det in orcasite_dets:
            # Filter by time range if specified.
            if det.timestamp is not None:
                if start_time is not None and det.timestamp < start_time:
                    continue
                if end_time is not None and det.timestamp > end_time:
                    continue

            orcahello_match = None
            if det.source == "machine":
                orcahello_match = next((d for d in orcahello_dets if d.id == det.idempotency_key), None)
                if orcahello_match is None:
                    print(f"Warning: couldn't find matching OrcaHello detection for {det.feed.slug} {det.timestamp}")
                    continue

            label = get_label(det, orcahello_match)
            if label is None:
                # label unknown -> skip.
                continue

            classification = classify_detection(det, label, orcasite_dets)
            if not classification.include:
                continue

            # Collect row data with timestamp for sorting.
            category = label
            node_name = det.feed.node_name  # e.g., "rpi_sunset_bay"
            timestamp_pst = format_timestamp_pst(det.timestamp)
            uri = generate_uri(det.feed.slug, det.timestamp)

            # Get confidence from orcahello_match if available.
            confidence = ""
            if orcahello_match and orcahello_match.confidence is not None:
                confidence = f"{orcahello_match.confidence:.4f}"
            elif classification.kind == "tp_both":
                # For tp_both detections (human + nearby machine), look up confidence
                # from the nearby machine detection's OrcaHello match.
                nearby_machine = find_nearby_machine_detection(det, orcasite_dets)
                if nearby_machine is not None:
                    nearby_orcahello = next(
                        (d for d in orcahello_dets if d.id == nearby_machine.idempotency_key),
                        None
                    )
                    if nearby_orcahello is not None and nearby_orcahello.confidence is not None:
                        confidence = f"{nearby_orcahello.confidence:.4f}"

            all_rows.append((det.timestamp, category, node_name, timestamp_pst, uri, det.description, classification.kind, confidence))

    # Sort all rows by timestamp (first element of tuple) - oldest first, newest last.
    all_rows.sort(key=lambda row: row[0])

    # Deduplicate rows based on (Category, NodeName, Timestamp, URI).
    # Keep the first occurrence of each unique detection.
    seen = set()
    unique_rows = []
    for row in all_rows:
        # Create a key from (category, node_name, timestamp, uri).
        key = (row[1], row[2], row[3], row[4])  # Category, NodeName, Timestamp, URI
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
        else:
            print(f"Skipping duplicate: {row[1]}, {row[2]}, {row[3]}")

    print(f"\nTotal rows: {len(all_rows)}, Unique rows: {len(unique_rows)}, Duplicates removed: {len(all_rows) - len(unique_rows)}")

    # Create CSV file and write sorted unique rows.
    csv_path = output_root / "detections.csv"
    output_root.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, lineterminator='\n')
        # Write header.
        csv_writer.writerow(['Category', 'NodeName', 'StartTimestamp', 'URI', 'Description', 'Notes', 'Confidence'])

        # Write sorted unique rows (exclude the timestamp used for sorting).
        for row in unique_rows:
            csv_writer.writerow([row[1], row[2], row[3], row[4], row[5], row[6], row[7]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CSV of detections from Orcasite and OrcaHello data"
    )
    parser.add_argument(
        "--feed",
        type=str,
        help="Process only this feed (by node_name, e.g., rpi_sunset_bay)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="YYYY_MM_DD_HH_MM_SS_PST",
        help="Include only detections with timestamp >= this value (e.g., 2025_01_01_00_00_00_PST)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026_03_17_00_00_00_PST",
        metavar="YYYY_MM_DD_HH_MM_SS_PST|now",
        help=(
            "Include only detections with timestamp <= this value "
            "(e.g., 2026_03_17_00_00_00_PST). "
            "Use 'now' to include all detections with no upper bound. "
            "Defaults to 2026_03_17_00_00_00_PST."
        )
    )
    args = parser.parse_args()

    start_time = parse_timestamp_pst(args.start) if args.start else None
    end_time = None if (args.end or "").lower() == "now" else parse_timestamp_pst(args.end)

    output_root = Path("bootstrap/csv")
    process_all_feeds(output_root, feed_filter=args.feed, start_time=start_time, end_time=end_time)
