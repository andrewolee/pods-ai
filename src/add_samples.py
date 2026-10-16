#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Split a 60-second audio sample into 3-second segments and run inference on each segment.

Usage:
    # Node name and timestamp inferred from filename of 60-second audio sample:
    python add_samples.py rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav

    # Override node name and/or timestamp explicitly:
    python add_samples.py recording.wav --node-name rpi_orcasound_lab \\
        --timestamp 2025_01_15_12_30_00_PST
    python add_samples.py recording.wav --node-name rpi_sunset_bay \\
        --timestamp 2025_01_15_12_30_00_PST \\
        --model-path /path/to/custom-model

    # Download audio from URI:
    python add_samples.py --uri "https://live.orcasound.net/bouts/new/sunset-bay?time=2024-07-04T18%3A05%3A59.000Z"

Saves 3-second segments with a 2-second hop to the "new/" output directory
(configurable with --output-dir) using the same filename convention as
output/wav/humpback/ etc.: {node_name_with_hyphens}_{timestamp_pst}.wav.
The timestamp in each filename reflects the actual start time of that sample.

Runs inference using the PODS-AI model (podsai) and prints the predicted class
label for each segment.  The default model is davethaler/whale-call-detector on
HuggingFace Hub; override with --model-path.

Output is printed in manual_samples.csv format (can be copy-pasted directly):
Category,NodeName,StartTimestamp,URI,Description,Notes,Confidence

If a corrected class is provided, rows whose predicted class already matches the
corrected class are omitted from the printed output.

URI/Description/Notes Lookup:
- The script looks up the detection in detections.csv (default: bootstrap/csv/detections.csv)
  by matching NodeName and StartTimestamp, and uses the URI, Description, and Notes from that row
- If not found in detections.csv, generates a URI from the timestamp and uses
  fallback_description when provided (otherwise empty Description), with Notes="manual"

If --node-name and --timestamp are omitted the script parses them from the
input filename.  The filename must follow the convention used by the
download_wavs.py outputs:
    {node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
For example, rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav yields
node_name=rpi_orcasound_lab and timestamp=2025_12_17_22_34_03_PST.
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from urllib.parse import quote, unquote, urlparse, parse_qs

import ffmpeg
from pytz import timezone

from model_inference import get_model_inference
from orcasite_feeds import get_orcasite_feeds, OrcasiteFeed
from audio_utils import (
    download_60s_audio,
    format_timestamp_pst,
)

SEGMENT_DURATION = 3  # Duration of each segment in seconds.
HOP_DURATION = 2  # Hop size between segments in seconds.
DEFAULT_OUTPUT_DIR = "new"  # Default output directory for segments.
DEFAULT_MODEL_PATH = "davethaler/whale-call-detector"  # Default HuggingFace model ID.
from model_revisions import LATEST_PODSAI_MODEL_REVISION
DEFAULT_DETECTIONS_CSV = "bootstrap/csv/detections.csv"  # Default path to detections.csv
PACIFIC_TZ = timezone("US/Pacific")  # Pacific timezone for timestamp formatting.
UTC_TZ = timezone("UTC")  # UTC timezone for URI generation.

# Regex that matches filenames produced by download_wavs.py:
# {node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
_FILENAME_PATTERN = re.compile(
    r"^(?P<node>.+?)_(?P<ts>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_PST)\.wav$",
    re.IGNORECASE,
)

# Cache for Orcasite feeds to avoid repeated API calls.
_FEEDS_CACHE: Optional[list[OrcasiteFeed]] = None


@dataclass
class DetectionInfo:
    """Information from a detection row in detections.csv."""
    uri: str
    description: str
    notes: str
    tags: str


def parse_node_and_timestamp_from_filename(wav_file: str) -> tuple[str, str]:
    """
    Parse the hydrophone node name and PST timestamp from a WAV filename.

    Expects filenames following the convention used by download_wavs.py:
        {node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
    For example, "rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav" yields
    node_name="rpi_orcasound_lab" and timestamp="2025_12_17_22_34_03_PST".
    Hyphens in the node-name portion are converted back to underscores.

    Args:
        wav_file: Path (or bare filename) of the WAV file to parse.

    Returns:
        Tuple of (node_name, timestamp_str) where node_name uses underscores.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    stem = Path(wav_file).name
    match = _FILENAME_PATTERN.match(stem)
    if not match:
        raise ValueError(
            f"Cannot infer node name and timestamp from filename: {stem!r}. "
            "The filename must follow the convention "
            "{node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav "
            "(e.g., rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav). "
            "Use --node-name and --timestamp to provide them explicitly."
        )
    node_name = match.group("node").replace("-", "_")
    timestamp_str = match.group("ts")
    return node_name, timestamp_str


def parse_timestamp_pst(timestamp_str: str) -> datetime:
    """
    Parse a PST timestamp string in the format YYYY_MM_DD_HH_MM_SS_PST.

    Args:
        timestamp_str: Timestamp string (e.g., "2025_12_24_17_51_23_PST").

    Returns:
        Parsed datetime object localized to the Pacific timezone.
    """
    timestamp_str = timestamp_str.replace("_PST", "")
    dt_naive = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
    return PACIFIC_TZ.localize(dt_naive)


def get_node_slug(node_name: str) -> str:
    """
    Look up the slug for a node_name from Orcasite feeds.

    Args:
        node_name: Internal node name (e.g., "rpi_orcasound_lab").

    Returns:
        URL slug for the node (e.g., "orcasound-lab").

    Raises:
        ValueError: If the node_name is not found in the Orcasite feeds.
    """
    global _FEEDS_CACHE

    # Load feeds from API if not already cached.
    if _FEEDS_CACHE is None:
        try:
            _FEEDS_CACHE = get_orcasite_feeds()
        except Exception as e:
            raise ValueError(f"Failed to fetch Orcasite feeds: {e}") from e

    # Look up the node_name in the feeds.
    for feed in _FEEDS_CACHE:
        if feed.node_name == node_name:
            return feed.slug

    # If not found, raise an error with available node names.
    available = [f.node_name for f in _FEEDS_CACHE]
    raise ValueError(
        f"Node name '{node_name}' not found in Orcasite feeds. "
        f"Available nodes: {', '.join(available)}"
    )


def get_node_name_from_slug(slug: str) -> str:
    """
    Look up the node_name for a slug from Orcasite feeds.

    Args:
        slug: URL slug (e.g., "orcasound-lab").

    Returns:
        Internal node name (e.g., "rpi_orcasound_lab").

    Raises:
        ValueError: If the slug is not found in the Orcasite feeds.
    """
    global _FEEDS_CACHE

    # Load feeds from API if not already cached.
    if _FEEDS_CACHE is None:
        try:
            _FEEDS_CACHE = get_orcasite_feeds()
        except Exception as e:
            raise ValueError(f"Failed to fetch Orcasite feeds: {e}") from e

    # Look up the slug in the feeds.
    for feed in _FEEDS_CACHE:
        if feed.slug == slug:
            return feed.node_name

    # If not found, raise an error with available slugs.
    available = [f.slug for f in _FEEDS_CACHE]
    raise ValueError(
        f"Slug '{slug}' not found in Orcasite feeds. "
        f"Available slugs: {', '.join(available)}"
    )


def parse_uri(uri: str) -> tuple[str, str]:
    """
    Parse a detection URI to extract node name and timestamp.

    Expects URIs in the format:
    https://live.orcasound.net/bouts/new/{slug}?time={utc_time}

    Args:
        uri: Detection URI (e.g., "https://live.orcasound.net/bouts/new/sunset-bay?time=2024-07-04T18%3A05%3A59.000Z")

    Returns:
        Tuple of (node_name, timestamp_str) where timestamp is in PST format.

    Raises:
        ValueError: If the URI cannot be parsed.
    """
    parsed = urlparse(uri)

    # Extract slug from path: /bouts/new/{slug}
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) < 3 or path_parts[0] != 'bouts' or path_parts[1] != 'new':
        raise ValueError(f"Invalid URI format: {uri}. Expected /bouts/new/{{slug}}?time=...")

    slug = path_parts[2]

    # Get node name from slug.
    node_name = get_node_name_from_slug(slug)

    # Extract time parameter.
    query_params = parse_qs(parsed.query)
    if 'time' not in query_params:
        raise ValueError(f"URI missing 'time' query parameter: {uri}")

    time_str = query_params['time'][0]

    # Parse UTC time and convert to PST.
    # Format: 2024-07-04T18:05:59.000Z
    try:
        utc_dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        # Try without milliseconds.
        utc_dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")

    utc_dt = UTC_TZ.localize(utc_dt)
    pst_dt = utc_dt.astimezone(PACIFIC_TZ)
    timestamp_str = format_timestamp_pst(pst_dt)

    return node_name, timestamp_str


def generate_uri(node_name: str, timestamp_str: str) -> str:
    """
    Generate a URI for the Orcasound bouts interface from a node name and PST timestamp.

    Args:
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
        timestamp_str: PST timestamp string in format 'YYYY_MM_DD_HH_MM_SS_PST'.

    Returns:
        URI in format "https://live.orcasound.net/bouts/new/{slug}?time={utc_time}".

    Raises:
        ValueError: If the node_name is not found in Orcasite feeds.
    """
    # Look up the slug for this node_name.
    slug = get_node_slug(node_name)

    # Parse timestamp and convert to UTC.
    dt = parse_timestamp_pst(timestamp_str)
    utc_dt = dt.astimezone(UTC_TZ)

    # Format as ISO 8601 with milliseconds and Z suffix.
    time_str = utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # URL encode the time parameter.
    time_encoded = quote(time_str, safe='')

    return f"https://live.orcasound.net/bouts/new/{slug}?time={time_encoded}"


def lookup_detection_in_csv(node_name: str, timestamp_str: str, detections_csv: str) -> Optional[DetectionInfo]:
    """
    Look up detection info in detections.csv by matching NodeName and StartTimestamp.

    Args:
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
        timestamp_str: PST timestamp string (e.g., "2023_10_28_07_33_52_PST").
        detections_csv: Path to detections.csv file.

    Returns:
        DetectionInfo with uri, description, notes, and tags if found, None otherwise.
    """
    detections_path = Path(detections_csv)
    if not detections_path.exists():
        print(f"Note: detections.csv not found at {detections_path}, will generate URI from timestamp")
        return None

    try:
        with open(detections_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Match both NodeName and StartTimestamp.
                if row.get('NodeName', '') == node_name and row.get('StartTimestamp', '') == timestamp_str:
                    uri = row.get('URI', '').strip()
                    description = row.get('Description', '').strip()
                    notes = row.get('Notes', '').strip()
                    tags = row.get('Tags', '').strip()

                    if uri:
                        print(f"Found detection in detections.csv:")
                        print(f"  URI: {uri}")
                        print(f"  Description: {description if description else '(empty)'}")
                        print(f"  Notes: {notes if notes else '(empty)'}")
                        print(f"  Tags: {tags if tags else '(empty)'}")
                        return DetectionInfo(uri=uri, description=description, notes=notes, tags=tags)
    except Exception as e:
        print(f"Warning: Failed to read detections.csv: {e}")
        return None

    return None


def split_wav_into_segments(
    wav_file: str,
    node_name: str,
    start_timestamp: str,
    output_dir: Path,
    segment_duration: int = SEGMENT_DURATION,
    hop_duration: int = HOP_DURATION,
) -> list[tuple[Path, str]]:
    """
    Split a 60-second audio sample into fixed-duration segments with a hop and save to output_dir.

    Uses the same filename convention as output/wav/humpback/ etc.:
    {node_name_with_hyphens}_{timestamp_pst}.wav, where the timestamp is the
    actual start time of each sample.

    Args:
        wav_file: Path to the input 60-second WAV file.
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
        start_timestamp: PST timestamp of the start of the recording
            (e.g., "2025_01_15_12_30_00_PST").
        output_dir: Directory to save the segment WAV files.
        segment_duration: Duration of each segment in seconds (default: 3).
        hop_duration: Hop size between segment start times in seconds (default: 2).

    Returns:
        List of (segment_path, timestamp_str) tuples for each saved segment,
        in order of increasing start time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe the audio file to get duration.
    try:
        probe = ffmpeg.probe(wav_file)
        duration = float(probe["format"]["duration"])
    except Exception as e:
        print(f"Error: Could not probe {wav_file}: {e}", file=sys.stderr)
        return []

    # Compute number of segment positions (sliding window).
    # Each position starts at pos_idx * hop_duration seconds.
    # The last position must start early enough that the full segment fits.
    num_positions = int((duration - segment_duration) // hop_duration) + 1
    if num_positions < 1:
        num_positions = 1

    # Parse base timestamp and build filename prefix.
    start_time = parse_timestamp_pst(start_timestamp)
    # Replace underscores with hyphens in the node name (matches download_wavs.py convention).
    node_name_in_filename = node_name.replace("_", "-")

    segments: list[tuple[Path, str]] = []
    for pos_idx in range(num_positions):
        start_offset = pos_idx * hop_duration
        seg_time = start_time + timedelta(seconds=start_offset)
        timestamp_str = format_timestamp_pst(seg_time)
        filename = f"{node_name_in_filename}_{timestamp_str}.wav"
        out_path = output_dir / filename

        if out_path.exists():
            print(f"Skipping (already exists): {out_path}")
            segments.append((out_path, timestamp_str))
            continue

        try:
            stream = ffmpeg.input(wav_file, ss=start_offset)
            stream = ffmpeg.output(
                stream,
                str(out_path),
                t=segment_duration,
                acodec="pcm_s16le",
                ar=44100,
                ac=1,
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            print(f"Saved: {out_path}")
            segments.append((out_path, timestamp_str))
        except Exception as e:
            print(
                f"Warning: Failed to extract segment at offset {start_offset}s: {e}",
                file=sys.stderr,
            )

    return segments


def get_segment_prediction(model: object, segment_path: Path) -> tuple[str, float]:
    """
    Run inference on a single segment WAV file and return the predicted class and confidence.

    Uses the PODS-AI (podsai) model output format: the predicted label is taken
    from the 'global_prediction_label' key and confidence from 'global_confidence'.

    Args:
        model: Loaded model inference object (from get_model_inference).
        segment_path: Path to the segment WAV file to score.

    Returns:
        Tuple of (predicted_class_label, confidence_score) where confidence is 0.0-1.0.
        Returns ("unknown", 0.0) if inference fails.
    """
    try:
        result = model.predict(str(segment_path))
    except Exception as e:
        print(f"Warning: Inference failed for {segment_path}: {e}", file=sys.stderr)
        return "unknown", 0.0

    label = result.get("global_prediction_label", "unknown")
    confidence = float(result.get("global_confidence", 0.0))
    return label, confidence


def add_training_3s_samples(
    wav_file: Optional[str] = None,
    uri: Optional[str] = None,
    node_name: Optional[str] = None,
    start_timestamp: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    model_revision: Optional[str] = LATEST_PODSAI_MODEL_REVISION,
    detections_csv: str = DEFAULT_DETECTIONS_CSV,
    model: Optional[object] = None,
    corrected_class: Optional[str] = None,
    fallback_description: Optional[str] = None,
    fallback_notes: Optional[str] = None,
    fallback_tags: Optional[str] = None
) -> list[dict]:
    """
    Split a 60-second audio sample into 3-second segments, save them, and run inference on each.

    Saves segments to output_dir using the filename convention
    {node_name_with_hyphens}_{timestamp_pst}.wav and returns a list of
    dictionaries with manual_samples.csv fields.  Inference always uses the
    PODS-AI (podsai) model type.

    Either wav_file or uri must be provided:
    - If wav_file is provided, uses that local file
    - If uri is provided, downloads the 60-second audio from Orcasound

    If node_name or start_timestamp are not provided they are inferred from the
    wav_file filename (or from the uri), which must follow the convention used by download_wavs.py:
    {node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
    (e.g., rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav).

    Args:
        wav_file: Path to the input WAV file. Either wav_file or uri must be provided.
        uri: Detection URI to download audio from. Either wav_file or uri must be provided.
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
            Inferred from wav_file filename or uri if not provided.
        start_timestamp: PST timestamp of the start of the recording
            (e.g., "2025_01_15_12_30_00_PST").
            Inferred from wav_file filename or uri if not provided.
        output_dir: Directory to save segments (default: "new").
        model_path: HuggingFace Hub model ID or path to a local model directory
            (default: DEFAULT_MODEL_PATH).
        model_revision: Git commit hash to pin the HuggingFace Hub model revision.
            Ignored when model_path is a local directory (default: LATEST_PODSAI_MODEL_REVISION).
        detections_csv: Path to detections.csv for detection lookup (default: "bootstrap/csv/detections.csv").
        model: Optional preloaded PODS-AI inference model to reuse instead of
            loading from model_path.
        corrected_class: Optional corrected class. When provided, rows whose
            predicted class already matches this class are not printed.
        fallback_description: Optional description to use when the detection
            is not found in detections.csv.
        fallback_notes: Optional notes to use when the detection is not found
            in detections.csv. Defaults to "manual" when not provided.
        fallback_tags: Optional tags to use when the detection is not found
            in detections.csv. Defaults to None when not provided.

    Returns:
        List of dictionaries with keys matching manual_samples.csv format:
        Category, NodeName, StartTimestamp, URI, Description, Notes, Confidence, Tags.

    Raises:
        ValueError: If neither wav_file nor uri is provided, or if node_name or
            start_timestamp cannot be inferred and are not provided.
    """
    if wav_file is None and uri is None:
        raise ValueError("Either wav_file or uri must be provided")

    if wav_file is not None and uri is not None:
        raise ValueError("Cannot specify both wav_file and uri")

    # Handle URI-based download.
    temp_dir = None
    if uri is not None:
        # Parse node name and timestamp from URI.
        if node_name is None or start_timestamp is None:
            inferred_node, inferred_ts = parse_uri(uri)
            if node_name is None:
                node_name = inferred_node
            if start_timestamp is None:
                start_timestamp = inferred_ts

        print(f"Downloading 60-second audio from URI...")
        print(f"  Node: {node_name}")
        print(f"  StartTimestamp: {start_timestamp}")

        # Download the 60-second WAV file.
        temp_dir = TemporaryDirectory()
        min_end_timestamp_pst_str=format_timestamp_pst(parse_timestamp_pst(start_timestamp) + timedelta(seconds=60))
        wav_path = download_60s_audio(
            node_name=node_name,
            min_end_timestamp_pst_str=min_end_timestamp_pst_str,
            tmp_dir=temp_dir.name)

        if wav_path is None:
            temp_dir.cleanup()
            raise ValueError(f"Failed to download audio for {node_name} at {start_timestamp}")

        wav_file = wav_path
        print(f"Downloaded to: {wav_file}")

    # At this point wav_file is set (either user-provided or downloaded).
    if node_name is None or start_timestamp is None:
        inferred_node, inferred_ts = parse_node_and_timestamp_from_filename(wav_file)
        if node_name is None:
            node_name = inferred_node
        if start_timestamp is None:
            start_timestamp = inferred_ts

    out_dir = Path(output_dir)

    # Try to look up detection info in detections.csv.
    detection_info = lookup_detection_in_csv(node_name, start_timestamp, detections_csv)
    if detection_info:
        # Use Description and Notes from detections.csv.
        shared_description = detection_info.description
        shared_notes = detection_info.notes
        shared_tags = ""
    else:
        shared_description = (fallback_description or "").strip()
        candidate_notes = (fallback_notes or "").strip()
        shared_notes = candidate_notes if candidate_notes else "manual"
        shared_tags = (fallback_tags or "").strip()

    # Split the WAV and save segments.
    segments = split_wav_into_segments(wav_file, node_name, start_timestamp, out_dir)

    # Clean up temporary directory if we downloaded the file.
    if temp_dir is not None:
        temp_dir.cleanup()

    if not segments:
        return []

    # Load the model once and run inference on each segment.
    if model is None:
        print(f"\nLoading podsai model from {model_path}...")
        model = get_model_inference(model_type="podsai", model_path=model_path,
                                    model_revision=model_revision)

    results: list[dict] = []
    print("\nSegments in manual_samples.csv format:")
    csv_writer = csv.writer(sys.stdout, lineterminator="\n")
    csv_writer.writerow(
        ["Category", "NodeName", "StartTimestamp", "URI", "Description", "Notes", "Confidence", "Tags"]
    )

    for seg_path, timestamp_str in segments:
        label, confidence = get_segment_prediction(model, seg_path)

        # Convert confidence from 0.0-1.0 to 0.0-100.0.
        confidence_pct = confidence * 100

        # Generate URI matching this segment's timestamp.
        segment_uri = generate_uri(node_name, timestamp_str)

        # Create row dict matching manual_samples.csv format.
        row = {
            "Category": label,
            "NodeName": node_name,
            "StartTimestamp": timestamp_str,
            "URI": segment_uri,
            "Description": shared_description,
            "Notes": shared_notes,
            "Confidence": f"{confidence_pct:.1f}",
            "Tags": shared_tags
        }
        results.append(row)

        if corrected_class is None or label != corrected_class:
            # Print in CSV format (ready to copy-paste) unless already corrected.
            csv_writer.writerow(
                [
                    corrected_class if corrected_class is not None else label,
                    node_name,
                    timestamp_str,
                    segment_uri,
                    shared_description,
                    shared_notes,
                    row["Confidence"],
                    shared_tags,
                ]
            )

    return results


def add_testing_60s_sample(
    node_name: Optional[str] = None,
    start_timestamp: Optional[str] = None,
    detections_csv: str = DEFAULT_DETECTIONS_CSV,
    corrected_class: Optional[str] = None,
    fallback_description: Optional[str] = None,
    fallback_notes: Optional[str] = None,
    fallback_tags: Optional[str] = None
) -> dict:
    """
    Returns a dictionary with manual_testing_60s_samples.csv fields.

    ``node_name`` and ``start_timestamp`` identify the 60-second sample and are required.
    The URI is generated from those values.

    Args:
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
            Inferred from wav_file filename or uri if not provided.
        start_timestamp: PST timestamp of the start of the recording
            (e.g., "2025_01_15_12_30_00_PST").
            Inferred from wav_file filename or uri if not provided.
        detections_csv: Path to detections.csv for detection lookup (default: "bootstrap/csv/detections.csv").
        corrected_class: Optional corrected class. When provided, rows whose
            predicted class already matches this class are not printed.
        fallback_description: Optional description to use when the detection
            is not found in detections.csv.
        fallback_notes: Optional notes to use when the detection is not found
            in detections.csv. Defaults to "manual" when not provided.
        fallback_tags: Optional tags to use when the detection is not found
            in detections.csv. Defaults to None when not provided.

    Returns:
        Dictionary with keys matching testing_60s_samples.csv format:
        Category, NodeName, StartTimestamp, URI, Description, Notes, Confidence, Tags.

    Raises:
        ValueError: If neither wav_file nor uri is provided, or if node_name or
            start_timestamp cannot be inferred and are not provided.
    """
    # Try to look up detection info in detections.csv.
    detection_info = lookup_detection_in_csv(node_name, start_timestamp, detections_csv)
    if detection_info:
        # Use Description and Notes from detections.csv.
        shared_description = detection_info.description
        shared_notes = detection_info.notes
        shared_tags = detection_info.tags
    else:
        shared_description = (fallback_description or "").strip()
        candidate_notes = (fallback_notes or "").strip()
        shared_notes = candidate_notes if candidate_notes else "manual"
        shared_tags = (fallback_tags or "").strip()

    # Generate URI matching this segment's timestamp.
    segment_uri = generate_uri(node_name, start_timestamp)

    # Create row dict matching manual_samples.csv format.
    row = {
            "Category": corrected_class,
            "NodeName": node_name,
            "StartTimestamp": start_timestamp,
            "URI": segment_uri,
            "Description": shared_description,
            "Notes": shared_notes,
            "Confidence": 100,
            "Tags": shared_tags
        }

    # Print in CSV format (ready to copy-paste) unless already corrected.
    csv_writer = csv.writer(sys.stdout, lineterminator="\n")
    csv_writer.writerow(
        [
            corrected_class,
            node_name,
            start_timestamp,
            segment_uri,
            shared_description,
            shared_notes,
            row["Confidence"],
            shared_tags,
        ]
        )

    return row


def main() -> int:
    """Entry point for the add_samples CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Split a 60-second audio sample into 3-second segments with 2-second hop, "
            "save to the output directory using the standard filename convention "
            "({node_name_with_hyphens}_{timestamp_pst}.wav), "
            "and run inference on each segment. "
            "Output is printed in manual_samples.csv format for easy copy-paste. "
            "Either provide a local WAV file or a detection URI to download audio."
        )
    )
    parser.add_argument(
        "wav_file",
        nargs='?',
        help="Path to the input WAV file to segment (optional if --uri is provided).",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help=(
            "Detection URI to download audio from "
            "(e.g., 'https://live.orcasound.net/bouts/new/sunset-bay?time=2024-07-04T18%%3A05%%3A59.000Z'). "
            "If provided, downloads a 60-second WAV file instead of using a local file."
        ),
    )
    parser.add_argument(
        "--node-name",
        default=None,
        help=(
            "Hydrophone node name (e.g., 'rpi_orcasound_lab'). "
            "Used in output filenames (underscores are replaced with hyphens). "
            "Inferred from the input filename or URI if not provided."
        ),
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "PST timestamp of the start of the recording "
            "(e.g., '2025_01_15_12_30_00_PST'). "
            "Each segment filename encodes the actual start time of that sample. "
            "Inferred from the input filename or URI if not provided."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save segments (default: {DEFAULT_OUTPUT_DIR!r}).",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=(
            "HuggingFace Hub model ID or path to a local podsai model directory "
            f"(default: {DEFAULT_MODEL_PATH!r})."
        ),
    )
    parser.add_argument(
        "--model-revision",
        default=LATEST_PODSAI_MODEL_REVISION,
        help=(
            "Git commit hash to pin the HuggingFace Hub model revision. "
            "Only used when --model-path is a Hub model ID. "
            f"Defaults to the pinned revision ({LATEST_PODSAI_MODEL_REVISION})."
        ),
    )
    parser.add_argument(
        "--detections-csv",
        default=DEFAULT_DETECTIONS_CSV,
        help=(
            f"Path to detections.csv for detection lookup (default: {DEFAULT_DETECTIONS_CSV!r})."
        ),
    )
    parser.add_argument(
        "--class",
        dest="corrected_class",
        default=None,
        help=(
            "Optional corrected class. When provided, printed rows whose "
            "predicted class already matches it are omitted."
        ),
    )

    args = parser.parse_args()

    # Validate arguments.
    if args.wav_file is None and args.uri is None:
        print("Error: Either wav_file or --uri must be provided", file=sys.stderr)
        parser.print_help()
        return 1

    if args.wav_file is not None and args.uri is not None:
        print("Error: Cannot specify both wav_file and --uri", file=sys.stderr)
        return 1

    if args.wav_file is not None and not Path(args.wav_file).exists():
        print(f"Error: WAV file not found: {args.wav_file}", file=sys.stderr)
        return 1

    try:
        results = add_training_3s_samples(
            wav_file=args.wav_file,
            uri=args.uri,
            node_name=args.node_name,
            start_timestamp=args.timestamp,
            output_dir=args.output_dir,
            model_path=args.model_path,
            model_revision=args.model_revision,
            detections_csv=args.detections_csv,
            corrected_class=args.corrected_class,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        return 1

    if not results:
        print("No segments were processed.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
