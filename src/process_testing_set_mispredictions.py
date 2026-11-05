#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Propose training-sample additions from testing-set mispredictions."""

import argparse
import csv
import sys
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse

from audio_utils import format_timestamp_pacific, parse_timestamp_pacific

TARGET_LABELS = ("resident", "transient", "humpback")
TRAINING_COLUMNS = [
    "Category",
    "NodeName",
    "StartTimestamp",
    "URI",
    "Description",
    "Notes",
    "Confidence",
]
TESTING_COLUMNS = [
    "Category",
    "NodeName",
    "StartTimestamp",
    "URI",
    "Description",
    "Notes",
    "Confidence",
]


def normalize_label(label: Any) -> str:
    """Normalize labels for consistent comparison."""
    normalized = str(label or "").strip().lower()
    return "resident" if normalized == "srkw" else normalized


def _to_utc_time_param(timestamp_pst: str) -> str:
    """Convert YYYY_MM_DD_HH_MM_SS_PST to UTC ISO string used by bouts URLs."""
    utc_dt = parse_timestamp_pacific(timestamp_pst).astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def build_segment_uri(source_uri: str, segment_timestamp: str) -> str:
    """Build a segment URI by replacing the query time in an existing bouts URI."""
    if not source_uri:
        return source_uri

    parsed = urlparse(source_uri)
    if not parsed.scheme or not parsed.netloc:
        return source_uri

    query = parse_qs(parsed.query, keep_blank_values=True)
    query["time"] = [_to_utc_time_param(segment_timestamp)]
    updated_query = urlencode(query, doseq=True, quote_via=quote, safe="")
    return urlunparse(parsed._replace(query=updated_query))


def find_wav_file(testing_row: dict[str, str], wav_dir: Path) -> Optional[Path]:
    """Return the testing WAV path for one testing_60s_samples.csv row."""
    category = (testing_row.get("Category") or "").strip()
    node_name = (testing_row.get("NodeName") or "").strip()
    start_timestamp = (testing_row.get("StartTimestamp") or "").strip()
    if not category or not node_name or not start_timestamp:
        return None

    node_name_in_filename = node_name.replace("_", "-")
    wav_filename = f"{node_name_in_filename}_{start_timestamp}.wav"
    wav_path = wav_dir / category / wav_filename
    if wav_path.exists():
        return wav_path
    return None


def to_local_prediction_labels(local_predictions: list[Any], id2label: dict[int, str]) -> list[str]:
    """Convert local predictions to normalized labels."""
    labels: list[str] = []
    for local_prediction in local_predictions:
        if isinstance(local_prediction, str):
            labels.append(normalize_label(local_prediction))
            continue
        if isinstance(local_prediction, int):
            labels.append(normalize_label(id2label.get(local_prediction, local_prediction)))
            continue
        labels.append(normalize_label(local_prediction))
    return labels


def get_global_whale_labels(inference: dict[str, Any]) -> list[str]:
    """Return normalized global whale labels from inference output."""
    global_labels = inference.get("global_prediction_labels")
    if global_labels is None:
        global_label = normalize_label(inference.get("global_prediction_label", ""))
        return [global_label] if global_label in TARGET_LABELS else []

    normalized_labels: list[str] = []
    for label in global_labels:
        normalized = normalize_label(label)
        if normalized in TARGET_LABELS and normalized not in normalized_labels:
            normalized_labels.append(normalized)
    return normalized_labels


def find_top_non_adjacent_segments(
    local_prediction_labels: list[str],
    local_confidences: list[Any],
    target_label: str,
    min_confidence: float = 0.80,
) -> list[tuple[int, float]]:
    """Find two non-adjacent target segments with the highest combined confidence."""
    candidates: list[tuple[int, float]] = []
    for index, (label, confidence_raw) in enumerate(zip(local_prediction_labels, local_confidences)):
        label_normalized = normalize_label(label)
        confidence = float(confidence_raw)
        if label_normalized != target_label or confidence <= min_confidence:
            continue
        candidates.append((index, confidence))

    if len(candidates) < 2:
        return []

    best_pair: Optional[tuple[tuple[int, float], tuple[int, float]]] = None
    best_score = -1.0
    best_peak = -1.0
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            if abs(first[0] - second[0]) <= 1:
                continue
            score = first[1] + second[1]
            peak = max(first[1], second[1])
            if score > best_score or (score == best_score and peak > best_peak):
                best_pair = (first, second)
                best_score = score
                best_peak = peak

    if best_pair is None:
        return []
    return sorted(best_pair, key=lambda item: item[1], reverse=True)


def build_training_row(
    testing_row: dict[str, str],
    segment_index: int,
    segment_confidence: float,
    hop_duration: float,
) -> dict[str, str]:
    """Compose a training_3s_samples-style row for one segment proposal."""
    clip_start = parse_timestamp_pacific((testing_row.get("StartTimestamp") or "").strip())
    segment_timestamp = format_timestamp_pacific(
        clip_start + timedelta(seconds=segment_index * float(hop_duration))
    )
    return {
        "Category": (testing_row.get("Category") or "").strip(),
        "NodeName": (testing_row.get("NodeName") or "").strip(),
        "StartTimestamp": segment_timestamp,
        "URI": build_segment_uri((testing_row.get("URI") or "").strip(), segment_timestamp),
        "Description": (testing_row.get("Description") or "").strip(),
        "Notes": (testing_row.get("Notes") or "").strip(),
        "Confidence": f"{float(segment_confidence) * 100:.1f}",
    }


def triage_testing_set_mispredictions(
    testing_csv: Path,
    wav_dir: Path,
    model_path: str,
    model_revision: Optional[str] = None,
    min_confidence: float = 0.80,
    actual_category_filter: Optional[str] = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    """Return proposed training rows and testing rows to remove."""
    proposals_for_training: list[dict[str, str]] = []
    proposals_to_remove: list[dict[str, str]] = []
    summary = {
        "rows_seen": 0,
        "missing_wav": 0,
        "inference_errors": 0,
        "proposed_training_rows": 0,
        "proposed_testing_removals": 0,
    }
    normalized_category_filter = (
        actual_category_filter.strip().lower() if actual_category_filter else None
    )

    with open(testing_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    from model_inference import get_model_inference

    model = get_model_inference(
        model_type="podsai",
        model_path=model_path,
        model_revision=model_revision,
    )
    id2label = getattr(model, "id2label", {}) or {}

    # Count the number of rows to process for progress reporting, after filtering by category if applicable.
    if normalized_category_filter:
        total = sum(
            1
            for row in rows
            if normalize_label(row.get("Category", "")) == normalized_category_filter
        )
    else:
        total = len(rows)

    removal_keys: set[tuple[str, str, str]] = set()
    for testing_row in rows:
        category = normalize_label(testing_row.get("Category", ""))
        if normalized_category_filter and category != normalized_category_filter:
            continue
        summary["rows_seen"] += 1
        print(f"Processing row {summary['rows_seen']} of {total}...")
        wav_path = find_wav_file(testing_row, wav_dir)
        if wav_path is None:
            summary["missing_wav"] += 1
            continue

        try:
            inference = model.predict(str(wav_path))
        except Exception as exc:
            print(
                f"Warning: inference failed for {wav_path}: {exc}",
                file=sys.stderr,
            )
            summary["inference_errors"] += 1
            continue

        global_labels = get_global_whale_labels(inference)
        if not global_labels:
            continue
        print(f"   Global whale predictions: {', '.join(global_labels)}")

        local_prediction_labels = to_local_prediction_labels(
            inference.get("local_predictions", []),
            id2label,
        )
        local_confidences = inference.get("local_confidences", [])
        if len(local_prediction_labels) != len(local_confidences):
            print(
                "Warning: skipping row due to mismatched inference lengths "
                f"(predictions={len(local_prediction_labels)}, "
                f"confidences={len(local_confidences)}) for {wav_path}",
                file=sys.stderr,
            )
            summary["inference_errors"] += 1
            continue
        hop_duration = float(inference.get("hop_duration", 2.0))

        proposed_for_row = False
        for target_label in TARGET_LABELS:
            if category == target_label or target_label not in global_labels:
                continue
            best_segments = find_top_non_adjacent_segments(
                local_prediction_labels,
                local_confidences,
                target_label=target_label,
                min_confidence=min_confidence,
            )
            if len(best_segments) != 2:
                continue

            for segment_index, segment_confidence in best_segments:
                proposals_for_training.append(
                    build_training_row(
                        testing_row=testing_row,
                        segment_index=segment_index,
                        segment_confidence=segment_confidence,
                        hop_duration=hop_duration,
                    )
                )
            proposed_for_row = True

        if proposed_for_row:
            key = (
                (testing_row.get("Category") or "").strip(),
                (testing_row.get("NodeName") or "").strip(),
                (testing_row.get("StartTimestamp") or "").strip(),
            )
            if key not in removal_keys:
                removal_keys.add(key)
                proposals_to_remove.append(
                    {column: (testing_row.get(column) or "").strip() for column in TESTING_COLUMNS}
                )

    summary["proposed_training_rows"] = len(proposals_for_training)
    summary["proposed_testing_removals"] = len(proposals_to_remove)
    return proposals_for_training, proposals_to_remove, summary


def print_rows(title: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Print rows in CSV format with a section title."""
    print(title)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    print()


def main() -> int:
    """Run testing-set misprediction triage."""
    parser = argparse.ArgumentParser(
        description=(
            "Propose training_3s_samples rows from testing_60s_samples mispredictions "
            "and list testing rows to remove."
        )
    )
    parser.add_argument(
        "--testing-csv",
        default="output/csv/testing_60s_samples.csv",
        help="Path to testing_60s_samples.csv.",
    )
    parser.add_argument(
        "--wav-dir",
        default="output/testing-wav",
        help="Root directory containing downloaded testing WAV files.",
    )
    parser.add_argument(
        "--model-path",
        default="davethaler/whale-call-detector",
        help="PODS-AI model path or HuggingFace model ID.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional model revision for HuggingFace model IDs.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.80,
        help="Minimum local segment confidence required for proposal generation.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        metavar="CATEGORY",
        help="Process only detections whose actual category matches this value.",
    )
    args = parser.parse_args()

    testing_csv = Path(args.testing_csv)
    wav_dir = Path(args.wav_dir)
    if not testing_csv.exists():
        print(f"Error: testing CSV not found: {testing_csv}", file=sys.stderr)
        return 1
    if not wav_dir.exists():
        print(f"Error: WAV directory not found: {wav_dir}", file=sys.stderr)
        return 1

    training_rows, testing_rows_to_remove, summary = triage_testing_set_mispredictions(
        testing_csv=testing_csv,
        wav_dir=wav_dir,
        model_path=args.model_path,
        model_revision=args.model_revision,
        min_confidence=args.min_confidence,
        actual_category_filter=args.category,
    )

    print()
    print_rows(
        "Proposed rows for output/csv/training_3s_samples.csv:",
        training_rows,
        TRAINING_COLUMNS,
    )
    print_rows(
        "Proposed rows to remove from output/csv/testing_60s_samples.csv:",
        testing_rows_to_remove,
        TESTING_COLUMNS,
    )
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
