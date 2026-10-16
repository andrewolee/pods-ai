#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Process confirmed OrcaHello SRKW detections into corrected manual samples.

For each confirmed OrcaHello detection in the selected timeframe, this script:
1. Downloads the corresponding 60-second WAV file.
2. Runs PODS-AI inference on the full WAV.
3. If PODS-AI predicts resident globally, treats it as not a false negative and skips it.
4. Splits the WAV into segments and runs PODS-AI segment inference.
5. Runs OrcaHello inference on each segment.
6. Appends segments where OrcaHello predicts resident but PODS-AI does not to
   new_manual_training_samples.csv or new_manual_testing_samples.csv with corrected class
   "resident", avoiding duplicates.
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from add_samples import (
    DEFAULT_DETECTIONS_CSV,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    add_testing_60s_sample,
    add_training_3s_samples,
)
from audio_utils import (
    download_60s_audio,
    format_timestamp_pst,
    get_orcahello_detections,
    parse_timestamp_pst,
)
from manual_samples_utils import (
    CSV_FIELDNAMES,
    append_manual_samples,
    filter_overlapping_samples,
    load_existing_uris,
)
from model_inference import get_model_inference
from orcasite_feeds import get_orcasite_feeds_with_retry

DEFAULT_MANUAL_TRAINING_SAMPLES_CSV = "output/csv/new_manual_training_samples.csv"
DEFAULT_MANUAL_TESTING_SAMPLES_CSV = "output/csv/new_manual_testing_samples.csv"
DEFAULT_TRAINING_SAMPLES_CSV = "output/csv/training_3s_samples.csv"
DEFAULT_TESTING_SAMPLES_CSV = "output/csv/testing_60s_samples.csv"
DEFAULT_ORCAHELLO_MODEL_PATH = "orcasound/orcahello-srkw-detector-v1"
TRAINING_SAMPLE_DURATION_SECONDS = 3
TESTING_SAMPLE_DURATION_SECONDS = 60
AUDIO_OFFSET_SECONDS = 2
WHALE_CLASSES = {"resident", "transient", "humpback"}


def is_orcahello_resident_prediction(label: str) -> bool:
    """Return True when an OrcaHello prediction label indicates resident presence."""
    normalized = (label or "").strip().lower()
    return normalized in {"resident", "whale"}


def process_false_negatives(
    manual_samples_path: Path,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    detections_csv: str = DEFAULT_DETECTIONS_CSV,
    orcahello_model_path: str = DEFAULT_ORCAHELLO_MODEL_PATH,
    feed_filter: Optional[str] = None,
    predicted_category_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    destination_samples_path: Optional[Path] = None,
    for_training: bool = True,
) -> dict[str, int]:
    """Process confirmed OrcaHello SRKW detections in the selected timeframe."""
    summary = {
        "confirmed": 0,
        "download_failed": 0,
        "not_false_negative": 0,
        "processing_failed": 0,
        "mismatched_segments": 0,
        "wrong_whale_class_segments": 0,
        "appended": 0,
        "duplicates": 0,
        "overlaps": 0,
    }
    normalized_category_filter = (
        predicted_category_filter.strip().lower() if predicted_category_filter else None
    )
    feeds = get_orcasite_feeds_with_retry()
    if not feeds:
        print("No Orcasite feeds available; nothing to process.")
        return summary

    if feed_filter:
        feeds = [feed for feed in feeds if feed.node_name == feed_filter]
        if not feeds:
            print(f"No feed found with node_name '{feed_filter}'")
            return summary

    existing_uris = load_existing_uris(manual_samples_path)
    candidate_rows = []

    print(f"Loading podsai model from {model_path}...")
    podsai_model = get_model_inference(model_type="podsai", model_path=model_path)
    if for_training:
        print(f"Loading orcahello model from {orcahello_model_path}...")
        orcahello_model = get_model_inference(
            model_type="orcahello", model_path=orcahello_model_path
        )

    for feed in feeds:
        print(f"Processing feed {feed.node_name}")
        for detection in get_orcahello_detections(feed, start_time, end_time):
            if detection.status.lower() != "confirmed" or detection.timestamp is None:
                continue
            # OrcaHello detections are returned in descending timestamp order.
            # Once older than start_time, all remaining detections in this feed are older.
            if start_time is not None and detection.timestamp < start_time:
                break
            if end_time is not None and detection.timestamp > end_time:
                continue

            summary["confirmed"] += 1
            timestamp_str = format_timestamp_pst(
                detection.timestamp - timedelta(seconds=AUDIO_OFFSET_SECONDS)
            )
            print(f"Checking confirmed OrcaHello detection at {timestamp_str}")

            with TemporaryDirectory() as temp_dir:
                min_end_timestamp_pst_str = format_timestamp_pst(detection.timestamp + timedelta(seconds=60))
                wav_path = download_60s_audio(node_name=feed.node_name, min_end_timestamp_pst_str=min_end_timestamp_pst_str, tmp_dir=temp_dir)
                if wav_path is None:
                    print(f"Skipping {feed.node_name} {timestamp_str}: failed to download audio.")
                    summary["download_failed"] += 1
                    continue

                try:
                    full_inference = podsai_model.predict(wav_path)
                    full_prediction_label = str(
                        full_inference.get("global_prediction_label", "")
                    ).strip().lower()
                    if normalized_category_filter and full_prediction_label != normalized_category_filter:
                        continue
                    if full_prediction_label == "resident":
                        print(
                            f"Skipping {feed.node_name} {timestamp_str}: "
                            "PODS-AI global prediction is resident."
                        )
                        summary["not_false_negative"] += 1
                        continue

                    if not for_training:
                        testing_row = add_testing_60s_sample(
                            node_name=feed.node_name,
                            start_timestamp=timestamp_str,
                            detections_csv=detections_csv,
                            corrected_class="resident",
                            fallback_description=detection.comments,
                            fallback_notes="tp_machine",
                            fallback_tags=detection.tags,
                        )
                        candidate_rows.append(testing_row)
                        continue

                    print(
                        f"Running add_samples.py for {feed.node_name} {timestamp_str} "
                        "with corrected class 'resident'."
                    )
                    podsai_segment_rows = add_training_3s_samples(
                        wav_file=wav_path,
                        node_name=feed.node_name,
                        start_timestamp=timestamp_str,
                        output_dir=str(output_dir),
                        model_path=model_path,
                        detections_csv=detections_csv,
                        model=podsai_model,
                        corrected_class="resident",
                        fallback_description=detection.comments,
                        fallback_notes="tp_machine",
                        fallback_tags=detection.tags,
                    )
                except Exception as exc:
                    print(f"Skipping {feed.node_name} {timestamp_str}: processing failed ({exc}).")
                    summary["processing_failed"] += 1
                    continue

            mismatched_rows: list[dict] = []
            node_name_in_filename = feed.node_name.replace("_", "-")
            for row in podsai_segment_rows:
                segment_timestamp = (row.get("StartTimestamp") or "").strip()
                if not segment_timestamp:
                    continue

                segment_filename = f"{node_name_in_filename}_{segment_timestamp}.wav"
                segment_path = output_dir / segment_filename
                if not segment_path.exists():
                    print(f"Skipping segment (missing): {segment_path}")
                    continue

                try:
                    orcahello_result = orcahello_model.predict(str(segment_path))
                except Exception as exc:
                    print(f"Warning: OrcaHello inference failed for {segment_path}: {exc}")
                    continue

                if not is_orcahello_resident_prediction(
                    str(orcahello_result.get("global_prediction_label", ""))
                ):
                    continue

                podsai_label = (row.get("Category") or "").strip().lower()
                if podsai_label == "resident":
                    continue

                if podsai_label in WHALE_CLASSES:
                    summary["wrong_whale_class_segments"] += 1

                updated_row = dict(row)
                updated_row["Category"] = "resident"
                mismatched_rows.append(updated_row)

            summary["mismatched_segments"] += len(mismatched_rows)
            candidate_rows.extend(mismatched_rows)

    if destination_samples_path is None:
        destination_samples_path = Path(
            DEFAULT_TRAINING_SAMPLES_CSV if for_training else DEFAULT_TESTING_SAMPLES_CSV
        )
    sample_duration = (
        TRAINING_SAMPLE_DURATION_SECONDS
        if for_training
        else TESTING_SAMPLE_DURATION_SECONDS
    )
    comparison_path = (
        Path(DEFAULT_TESTING_SAMPLES_CSV)
        if for_training
        else Path(DEFAULT_TRAINING_SAMPLES_CSV)
    )
    comparison_duration = (
        TESTING_SAMPLE_DURATION_SECONDS
        if for_training
        else TRAINING_SAMPLE_DURATION_SECONDS
    )
    new_rows = []
    candidate_uris = set(existing_uris)
    for row in candidate_rows:
        uri = (row.get("URI") or "").strip()
        if uri in candidate_uris:
            summary["duplicates"] += 1
            continue
        candidate_uris.add(uri)
        new_rows.append(row)

    proposed_rows, overlaps = filter_overlapping_samples(
        new_rows,
        destination_samples_path,
        sample_duration,
        comparison_paths=((comparison_path, comparison_duration),),
    )
    summary["overlaps"] = overlaps

    appended, duplicates = append_manual_samples(
        manual_samples_path,
        proposed_rows,
        existing_uris,
        fieldnames=CSV_FIELDNAMES,
    )
    summary["appended"] = appended
    summary["duplicates"] += duplicates

    print(f"Proposed rows for {destination_samples_path.as_posix()}:")
    csv_writer = csv.DictWriter(
        sys.stdout,
        fieldnames=CSV_FIELDNAMES,
        lineterminator="\n",
        extrasaction="ignore",
    )
    csv_writer.writeheader()
    csv_writer.writerows(proposed_rows)

    return summary


def main() -> int:
    """Run the false-negative processing CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Process confirmed OrcaHello detections, find 60-second false negatives "
            "where PODS-AI misses resident calls, and append corrected resident "
            "sub-segments to new_manual_training_samples.csv for the training set,"
            "or new_manual_testing_samples.csv for the testing set."
        )
    )
    parser.add_argument(
        "--set",
        type=str,
        default="training",
        help="training or testing (default: training).",
    )
    parser.add_argument(
        "--feed",
        type=str,
        help="Process only this feed (by node_name, e.g., rpi_sunset_bay).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="YYYY_MM_DD_HH_MM_SS_PST",
        help="Include only detections with timestamp >= this value.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="now",
        metavar="YYYY_MM_DD_HH_MM_SS_PST|now",
        help=(
            "Include only detections with timestamp <= this value. "
            "Use 'now' to remove the upper bound."
        ),
    )
    parser.add_argument(
        "--manual-samples-csv",
        default=None,
        help="Path to the manual samples CSV; defaults to the file for the selected set.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where add_samples.py should write segment WAV files.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="PODS-AI model path or HuggingFace model ID.",
    )
    parser.add_argument(
        "--orcahello-model-path",
        default=DEFAULT_ORCAHELLO_MODEL_PATH,
        help="OrcaHello model path or HuggingFace model ID.",
    )
    parser.add_argument(
        "--detections-csv",
        default=DEFAULT_DETECTIONS_CSV,
        help="Path to detections.csv for add_samples.py metadata lookups.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        metavar="CATEGORY",
        help="Process only detections whose PODS-AI predicted category matches this value.",
    )
    args = parser.parse_args()

    start_time = parse_timestamp_pst(args.start) if args.start else None
    end_time = None if (args.end or "").lower() == "now" else parse_timestamp_pst(args.end)
    result_set = args.set.lower()
    if result_set not in ("training", "testing"):
        print(f"Invalid set value: {args.set}. Must be 'training' or 'testing'.")
        return 1

    manual_samples_csv = args.manual_samples_csv
    if not manual_samples_csv:
        manual_samples_csv = (
            DEFAULT_MANUAL_TRAINING_SAMPLES_CSV
            if result_set == "training"
            else DEFAULT_MANUAL_TESTING_SAMPLES_CSV
        )

    summary = process_false_negatives(
        manual_samples_path=Path(manual_samples_csv),
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        detections_csv=args.detections_csv,
        orcahello_model_path=args.orcahello_model_path,
        feed_filter=args.feed,
        predicted_category_filter=args.category,
        start_time=start_time,
        end_time=end_time,
        for_training=(result_set == "training"),
    )

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
