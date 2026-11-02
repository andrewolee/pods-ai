# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Read-only MCP Server for Orcasound data interrogation (stdio transport).
Tools — all usable without AKS access:
1. list_hydrophones — active stations from the Orcasite feeds API.
2. get_recent_detections — latest sound detections from the Orcasite API.
3. list_s3_recordings — available HLS timestamp folders in S3 for a node.
4. get_sample_stats — category distribution in local training / testing CSVs.
5. find_unlabeled_detections — Orcasite detections not yet present in local CSVs.
6. compare_models_on_clip — run OrcaHello (HuggingFace) + PODS-AI on a local WAV.
7. generate_spectrogram — generate a PNG spectrogram for a local WAV clip.
8. export_unlabeled_to_csv — extract unlabeled remote data and save directly to disk.
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from mcp.server.fastmcp import FastMCP
import requests
import structlog

# Platform-specific imports for non-blocking stdin check.
# Both are imported unconditionally so they are always patchable in tests.
try:
    import msvcrt  # Windows only
except ImportError:
    # Provide a stub on non-Windows so the name is always patchable in tests.
    class _MsvcrtStub:  # noqa: N801
        @staticmethod
        def kbhit() -> bool:
            return False  # pragma: no cover
    msvcrt = _MsvcrtStub()  # type: ignore[assignment]

import select  # Unix/Linux/macOS (and Windows with sockets)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)
logger = structlog.get_logger("orcasound_mcp")

# Resolve project layout regardless of working directory.
_SRC_DIR = Path(__file__).parent.absolute()
_PROJECT_ROOT = _SRC_DIR.parent
_CSV_DIR = _PROJECT_ROOT / "output" / "csv"

sys.path.insert(0, str(_SRC_DIR))
from orcasite_feeds import get_orcasite_feeds_with_retry  # noqa: E402

mcp = FastMCP("Orcasound MCP")

S3_BUCKET = "audio-orcasound-net"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _validate_node_name(node_name: str) -> None:
    if not node_name or not all(c.isalnum() or c in ("_", "-") for c in node_name):
        raise ValueError(
            f"Invalid node_name '{node_name}'. "
            "Only letters, digits, hyphens, and underscores are allowed."
        )


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Return all rows of a CSV as a list of dicts. Raises FileNotFoundError if absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected CSV not found: {path}. "
            "Run the pipeline scripts (make_csv → extract_training_samples → merge_training_samples) first."
        )
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tool 1 — Hydrophone stations.
# ---------------------------------------------------------------------------

@mcp.tool()
def list_hydrophones() -> List[Dict[str, Any]]:
    """Return metadata for every active Orcasound hydrophone station.

    Data comes from the public Orcasite feeds API (no credentials required).
    Each entry includes name, geographic coordinates, S3 bucket, AWS region,
    CloudFront URL, and visibility flag.
    """
    logger.info("list_hydrophones")
    feeds = get_orcasite_feeds_with_retry()
    return [
        {
            "id": f.id,
            "name": f.name,
            "node_name": f.node_name,
            "slug": f.slug,
            "latitude": f.location[0],
            "longitude": f.location[1],
            "bucket": f.bucket,
            "region": f.bucket_region,
            "visible": f.visible,
            "cloudfront_url": f.cloudfront_url,
        }
        for f in feeds
    ]


# ---------------------------------------------------------------------------
# Tool 2 — Recent detections from Orcasite.
# ---------------------------------------------------------------------------

@mcp.tool()
def get_recent_detections(node_name: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent sound detections for a hydrophone station.

    Queries the public Orcasite JSON API. Each record includes timestamp,
    category (whale / vessel / other), source (human / machine), and description.

    Args:
        node_name: Hydrophone node identifier, e.g., 'rpi_sunset_bay'. Also accepts the station slug, e.g., 'sunset-bay'.
        limit: Number of detections to retrieve (1–250, default 50).
    """
    _validate_node_name(node_name)
    if not 1 <= limit <= 250:
        raise ValueError("limit must be between 1 and 250.")

    logger.info("get_recent_detections", node_name=node_name, limit=limit)

    feeds = get_orcasite_feeds_with_retry()
    feed = next(
        (f for f in feeds if f.node_name == node_name or f.slug == node_name),
        None,
    )
    if feed is None:
        raise ValueError(f"No hydrophone station found matching '{node_name}'.")

    resp = requests.get(
        "https://live.orcasound.net/api/json/detections",
        params={
            "page[limit]": limit,
            "fields[detection]": (
                "id,playlist_timestamp,player_offset,timestamp,"
                "description,source,category,feed_id,idempotency_key"
            ),
            "filter[feed_id]": feed.id,
        },
        timeout=15,
    )
    resp.raise_for_status()

    return [
        {
            "id": item.get("id"),
            "timestamp": item.get("attributes", {}).get("timestamp"),
            "source": item.get("attributes", {}).get("source"),
            "category": item.get("attributes", {}).get("category"),
            "description": item.get("attributes", {}).get("description"),
            "player_offset": item.get("attributes", {}).get("player_offset"),
            "playlist_timestamp": item.get("attributes", {}).get("playlist_timestamp"),
        }
        for item in resp.json().get("data", [])
    ]


# ---------------------------------------------------------------------------
# Tool 3 — S3 recording index.
# ---------------------------------------------------------------------------

@mcp.tool()
def list_s3_recordings(
    node_name: str,
    date_prefix: Optional[str] = None,
    max_results: int = 100,
) -> Dict[str, Any]:
    """List available HLS recording folders in S3 for a specific hydrophone node.

    Each folder is a Unix-epoch timestamp marking the start of a ~60 s segment.
    No AWS credentials are required (public read access).

    Stream URL pattern:
      https://s3-us-west-2.amazonaws.com/audio-orcasound-net/<node_name>/hls/<epoch>/live.m3u8

    Args:
        node_name: Hydrophone node identifier, e.g., 'rpi_sunset_bay'.
        date_prefix: Optional epoch prefix to narrow listing, e.g., '174' for recent 2025 data.
        max_results: Maximum number of timestamps to return (1–1000, default 100).
    """
    _validate_node_name(node_name)
    if not 1 <= max_results <= 1000:
        raise ValueError("max_results must be between 1 and 1000.")

    logger.info("list_s3_recordings", node_name=node_name, date_prefix=date_prefix)

    prefix = f"{node_name}/hls/"
    if date_prefix:
        prefix += date_prefix

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    timestamps: List[str] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            timestamps.append(cp["Prefix"].rstrip("/").split("/")[-1])
        if len(timestamps) >= max_results:
            break

    timestamps = sorted(timestamps)[-max_results:]

    return {
        "node_name": node_name,
        "bucket": S3_BUCKET,
        "total_found": len(timestamps),
        "timestamps": timestamps,
    }


# ---------------------------------------------------------------------------
# Tool 4 — Sample stats from local CSVs.
# ---------------------------------------------------------------------------

@mcp.tool()
def get_sample_stats(split: str = "training") -> Dict[str, Any]:
    """Return category and per-station distribution for the local training or testing samples.

    Reads the CSV files generated by the pipeline scripts in output/csv/.
    Use this to spot class imbalances or gaps in hydrophone coverage before training.

    Args:
        split: Which CSV to analyze — 'training' (default) or 'testing'.
    """
    if split not in ("training", "testing"):
        raise ValueError("split must be 'training' or 'testing'.")

    csv_path = _CSV_DIR / f"{split}_samples.csv"
    logger.info("get_sample_stats", split=split, path=str(csv_path))

    rows = _read_csv(csv_path)

    # Category counts.
    categories: Dict[str, int] = {}
    # Per-station counts.
    per_station: Dict[str, Dict[str, int]] = {}

    for row in rows:
        cat = row.get("Category", "unknown").strip()
        node = row.get("NodeName", "unknown").strip()

        categories[cat] = categories.get(cat, 0) + 1

        if node not in per_station:
            per_station[node] = {}
        per_station[node][cat] = per_station[node].get(cat, 0) + 1

    return {
        "split": split,
        "total_samples": len(rows),
        "by_category": dict(sorted(categories.items(), key=lambda x: -x[1])),
        "by_station": {
            node: dict(sorted(cats.items(), key=lambda x: -x[1]))
            for node, cats in sorted(per_station.items())
        },
    }


# ---------------------------------------------------------------------------
# Tool 5 — Find unlabeled detections.
# ---------------------------------------------------------------------------

@mcp.tool()
def find_unlabeled_detections(
    node_name: str,
    limit: int = 100,
) -> Dict[str, Any]:
    """Find recent Orcasite detections for a station that are NOT in any local CSV.

    Useful for identifying new human-annotated data that hasn't been pulled
    into the training pipeline yet.

    Args:
        node_name: Hydrophone node identifier, e.g., 'rpi_sunset_bay'.
        limit: How many recent Orcasite detections to fetch for comparison (1–250).
    """
    _validate_node_name(node_name)
    if not 1 <= limit <= 250:
        raise ValueError("limit must be between 1 and 250.")

    logger.info("find_unlabeled_detections", node_name=node_name)

    # Collect all URIs already in local CSVs.
    known_uris: set[str] = set()
    for csv_name in ("detections.csv", "training_samples.csv", "testing_samples.csv"):
        path = _CSV_DIR / csv_name
        if path.exists():
            for row in _read_csv(path):
                uri = row.get("URI", "").strip()
                if uri:
                    known_uris.add(uri)

    # Fetch recent detections from Orcasite.
    api_detections = get_recent_detections(node_name=node_name, limit=limit)

    # A detection is "unlabeled" if its idempotency_key / playlist URI is not in any local CSV.
    unlabeled = []
    for det in api_detections:
        # Orcasite playlist_timestamp is the closest proxy to the CSV URI timestamp.
        playlist_ts = str(det.get("playlist_timestamp") or "")
        if not playlist_ts or not any(playlist_ts in uri for uri in known_uris):
            unlabeled.append(det)

    return {
        "node_name": node_name,
        "fetched_from_api": len(api_detections),
        "already_in_local_csvs": len(api_detections) - len(unlabeled),
        "unlabeled_count": len(unlabeled),
        "unlabeled": unlabeled,
    }


# ---------------------------------------------------------------------------
# Tool 6 — Compare OrcaHello vs PODS-AI on a local WAV.
# ---------------------------------------------------------------------------

@mcp.tool()
def compare_models_on_clip(
    wav_path: str,
    podsai_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run OrcaHello (HuggingFace) and the local PODS-AI model on the same WAV clip
    and return side-by-side scores.

    OrcaHello is downloaded automatically from HuggingFace Hub on first use
    (model: orcasound/orcahello-srkw-detector-v1). No AKS or Azure access required.

    Args:
        wav_path: Absolute path to a local .wav file.
        podsai_model_path: Path or HuggingFace Hub ID of the PODS-AI model to compare.
        Defaults to 'davethaler/whale-call-detector' if not provided.

    Returns:
        Comparison payload. Each model's details includes the optional
        ``global_prediction_labels`` list for modern multi-label consumers; legacy
        consumers may continue using ``global_prediction_label``.
    """
    path = Path(wav_path)
    if not path.is_absolute() or not path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    if path.suffix.lower() != ".wav":
        raise ValueError("Only .wav files are supported.")

    podsai_model_path = podsai_model_path or "davethaler/whale-call-detector"
    logger.info("compare_models_on_clip", wav=wav_path, podsai_model=podsai_model_path)

    results: Dict[str, Any] = {}

    for model_key, model_type, model_path in [
        ("orcahello", "orcahello", None),
        ("podsai", "podsai", podsai_model_path),
    ]:
        try:
            # Lazy import — heavy ML deps only loaded when this tool is actually called.
            # Import inside the try/except so missing optional deps don't crash the entire tool.
            from model_inference import get_model_inference
            from LiveInferenceOrchestrator import is_positive_label
            model = get_model_inference(
                model_type=model_type,
                model_path=model_path,
                auto_download=True,
            )
            preds = model.predict(str(path))
            if preds.get("error"):
                raise RuntimeError(f"{model_key} inference failed: {preds['error']}")
            global_prediction_label = preds.get("global_prediction_label")
            global_prediction_labels = preds.get("global_prediction_labels")
            if global_prediction_labels is None:
                global_prediction_labels = (
                    [global_prediction_label]
                    if global_prediction_label
                    and is_positive_label(global_prediction_label)
                    else []
                )
            results[model_key] = {
                "global_prediction_label": global_prediction_label,
                "global_prediction_labels": global_prediction_labels,
                "global_confidence": round(float(preds.get("global_confidence", 0.0)), 4),
                "local_confidences": [round(float(c), 4) for c in preds.get("local_confidences", [])],
                "hop_duration_s": float(preds.get("hop_duration", 1.0)),
                "segment_duration_s": float(preds.get("segment_duration", 3.0)),
                "error": None,
            }
        except Exception as exc:
            logger.warning("model_inference_failed", model=model_key, error=str(exc))
            results[model_key] = {"error": str(exc)}

    # Agreement flag — both models must succeed and agree on the top-level label.
    orca_label = results.get("orcahello", {}).get("global_prediction_label")
    pods_label = results.get("podsai", {}).get("global_prediction_label")
    agree = (orca_label is not None and orca_label == pods_label)

    return {
        "wav_path": wav_path,
        "models_agree": agree,
        "orcahello_label": orca_label,
        "podsai_label": pods_label,
        "details": results,
    }


# ---------------------------------------------------------------------------
# Tool 7 — Generate a spectrogram PNG from a local WAV clip.
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_spectrogram(
    wav_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a spectrogram PNG for a local WAV clip.

    This is useful for visual debugging when reviewing detections, comparing model
    outputs, or inspecting unknown hydrophone recordings without leaving the agent
    session.

    Args:
        wav_path: Absolute path to a local .wav file.
        output_path: Optional output PNG path. If omitted, the PNG is saved next to
            the input WAV using the same basename and a .png extension.

    Returns:
        Dictionary containing the input wav path and generated PNG path.
    """
    path = Path(wav_path)
    if not path.is_absolute() or not path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    if path.suffix.lower() != ".wav":
        raise ValueError("Only .wav files are supported.")

    resolved_output: Optional[Path] = None
    if output_path:
        output_file = Path(output_path)
        if not output_file.is_absolute():
            output_file = (path.parent / output_file).resolve()
        if output_file.is_dir():
            output_file = output_file / f"{path.stem}.png"
        elif output_file.suffix.lower() == "":
            output_file = output_file.with_suffix(".png")
        elif output_file.suffix.lower() != ".png":
            raise ValueError("output_path must be a .png file or a directory path.")
        resolved_output = output_file

    logger.info("generate_spectrogram", wav=wav_path, output_path=str(resolved_output) if resolved_output else None)

    if output_path is None:
        default_png_path = path.with_suffix(".png")
        if default_png_path.exists():
            raise FileExistsError(f"Spectrogram output already exists: {default_png_path}")
    elif resolved_output and resolved_output.exists():
        raise FileExistsError(f"Spectrogram output already exists: {resolved_output}")

    try:
        from spectrogram_visualizer import write_spectrogram
    except Exception as exc:  # pragma: no cover - dependency guard for minimal MCP installs
        raise RuntimeError(
            "Spectrogram generation requires the full audio stack from requirements.txt "
            "(librosa, matplotlib, soundfile, numpy)."
        ) from exc

    png_path = Path(write_spectrogram(path, output_path=resolved_output))
    if png_path.exists():
        return {
            "wav_path": str(path),
            "png_path": str(png_path),
            "output_path": str(png_path),
            "size_bytes": png_path.stat().st_size,
        }

    raise RuntimeError(f"Spectrogram generation failed for {wav_path}")

# ---------------------------------------------------------------------------
# Tool 8 — Export an unlabeled detections dataset to a CSV file.
# ---------------------------------------------------------------------------
@mcp.tool()
def export_unlabeled_to_csv(
    node_name: str,
    output_filename: str,
    limit: int = 100,
) -> str:
    """Find unlabeled detections for a station and save them to a new CSV file.

    Args:
        node_name: Hydrophone node identifier, e.g., 'rpi_sunset_bay'.
        output_filename: A simple filename (e.g., 'sunset_bay.csv') in output/csv/.
        limit: How many recent Orcasite detections to fetch for comparison (1–250).
    """
    _validate_node_name(node_name)
    if not 1 <= limit <= 250:
        raise ValueError("limit must be between 1 and 250.")

    # Validate the output filename to prevent directory traversal.
    filename_path = Path(output_filename)
    if (
        filename_path.name != output_filename
        or ".." in output_filename
        or not output_filename.lower().endswith(".csv")
    ):
        raise ValueError(
            "output_filename must be a simple filename ending in '.csv' "
            "without any directory or path components."
        )

    # Re-use existing tool logic to get the data.
    data = find_unlabeled_detections(node_name, limit)
    unlabeled_items = data.get("unlabeled", [])

    if not unlabeled_items:
        return "No unlabeled detections found to export."

    # Ensure the destination directory exists.
    _CSV_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _CSV_DIR / output_filename

    # Write the data directly to a CSV.
    keys = unlabeled_items[0].keys()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(unlabeled_items)

    return f"Successfully created dataset! Saved {len(unlabeled_items)} rows to {output_path}"


def run_dual_handshake(mcp):
    """
    Allow both Claude Desktop (client-initiated) and Visual Studio (server-initiated)
    MCP handshakes.

    Checks for client input without blocking, avoiding stdin race conditions.
    """
    client_message = None

    # Platform-specific non-blocking check for available stdin.
    if sys.platform == "win32":
        # Windows: Use msvcrt.kbhit() to check without blocking.
        time.sleep(0.2)  # Give client a moment to send data.

        if msvcrt.kbhit():
            # Client sent something — read one line and save it.
            client_message = sys.stdin.readline()
    else:
        # Unix/Linux/macOS: Use select to check stdin readiness.
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if ready:
            # Client sent something — read one line and save it.
            client_message = sys.stdin.readline()

    if not client_message or not client_message.strip():
        # No client message within timeout — server-initiated handshake (Visual Studio).
        init_msg = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"}
        }
        sys.stdout.write(json.dumps(init_msg) + "\n")
        sys.stdout.flush()
    else:
        # Client-initiated handshake (Claude Desktop) — prepend message back to stdin.
        # Create a wrapper that yields the saved message first, then normal stdin.
        original_stdin = sys.stdin

        class PrependedStdin:
            """Wrapper that returns one prepended line, then delegates to original stdin."""
            def __init__(self, prepended_line, original):
                self._prepended = prepended_line
                self._original = original
                self._consumed = False

            def readline(self):
                if not self._consumed:
                    self._consumed = True
                    return self._prepended
                return self._original.readline()

            def read(self, size=-1):
                if not self._consumed:
                    self._consumed = True
                    result = self._prepended
                    if size > 0:
                        remaining = size - len(result)
                        if remaining > 0:
                            result += self._original.read(remaining)
                        else:
                            result = result[:size]
                    else:
                        result += self._original.read()
                    return result
                return self._original.read(size)

            def __getattr__(self, name):
                return getattr(self._original, name)

        sys.stdin = PrependedStdin(client_message, original_stdin)

    # Hand control to FastMCP — it is now the sole stdin consumer.
    mcp.run()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("mcp_server_starting", transport="stdio")
    run_dual_handshake(mcp)
