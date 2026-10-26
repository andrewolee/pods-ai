#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Process rejected OrcaHello resident detections into corrected manual samples.

For each rejected OrcaHello detection in the selected timeframe, this script:
1. Downloads the corresponding 60-second WAV file.
2. Runs PODS-AI inference on the full WAV.
3. Infers the corrected class from the OrcaHello comments.
4. Runs add_samples.py on the WAV file.
5. Appends whale-predicted segments that do not already match the corrected class
   to new_manual_training_samples.csv or new_manual_testing_samples.csv with the
   corrected class, avoiding duplicates.
"""

import argparse
import csv
import io
import re
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from add_samples import DEFAULT_DETECTIONS_CSV, DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, add_training_3s_samples, add_testing_60s_sample
from audio_utils import (
    SKIP_TERMS,
    download_60s_audio,
    format_timestamp_pst,
    get_orcahello_detections,
    parse_timestamp_pst,
)
from manual_samples_utils import (
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
TRAINING_SAMPLE_DURATION_SECONDS = 3
TESTING_SAMPLE_DURATION_SECONDS = 60
AUDIO_OFFSET_SECONDS = 2
BIRD_TERMS = ("bird", "pigu", "keir")
RESIDENT_TERMS = ("resident", "pod")
TRANSIENT_TERMS = ("bigg", "transient")
HUMAN_TERMS = ("human", "radio")
VESSEL_TERMS = ("vessel", "ship", "boat", "train")
OTHER_TERMS = ("seal","sea lion")
# Checked only after RESIDENT_TERMS/TRANSIENT_TERMS/"humpback"/HUMAN_TERMS, so a
# comment mentioning "humpback" or "human" is matched by those terms first when
# unnegated.  "hum" itself is matched with a word-boundary regex (see
# HUM_WORD_PATTERN below) since "humpback" and "human" both literally start
# with "hum" as a substring, which would otherwise match even when negated
# (e.g. "No humpback nor vessel" still contains the substring "hum").
HUM_TERMS = ("electrical", "60hz")
HUM_WORD_PATTERN = re.compile(r"\bhum\b")
WHALE_CLASSES = {"resident", "transient", "humpback"}
OTHER_CLASSES = {"bird", "human", "vessel", "jingle", "water", "other", "hum"}
# Older tag used before moderators standardized on "hum".
HUM_TAG_ALIASES = {"60hz"}
# Phrases that negate "humpback" or "vessel" labels (e.g. human-written "No humpback nor vessel").
NO_HUMPBACK_TERMS = ("no humpback", "not humpback")
NO_VESSEL_TERMS = ("no vessel", "nor vessel", "no boat", "nor boat", "no ship", "nor ship", "no train", "nor train")


def get_corrected_class(comments: str, tags: Optional[str]) -> Optional[str]:
    """Infer the corrected class from OrcaHello moderation tags and comments.

    Use tags if available, otherwise infer from comments.  Return None if no class can be inferred.

    Auto-generated "AI: …" prefix comment lines are stripped before parsing so that
    phrases like "AI: humpback" do not influence the result.  Explicit negations
    ("No humpback", "No humpback nor vessel") are recognised:

    * "No humpback" alone (no other positive signal) – returns ``"water"``.
    * "No humpback nor vessel" (and no other positive signal) – returns
      ``"water"``.
    """
    if tags:
        normalized_tags = [tag.strip().lower() for tag in tags.split(";")]

        # A real whale-species tag always takes priority over a noise tag,
        # regardless of which appears first in the comma/semicolon-separated
        # list (e.g. "vessel;resident" or "hum;transient;orca;whale" should
        # resolve to the whale species, not the noise tag, since a genuine
        # call was present).
        for normalized_tag in normalized_tags:
            if normalized_tag in WHALE_CLASSES:
                return normalized_tag

        for normalized_tag in normalized_tags:
            if normalized_tag in HUM_TAG_ALIASES:
                return "hum"
            if normalized_tag in OTHER_CLASSES:
                return normalized_tag
        return "water"

    # Drop auto-generated "AI: …" lines so they do not influence class inference.
    human_lines = [
        line for line in (comments or "").splitlines()
        if not line.strip().lower().startswith("ai:")
    ]
    text = " ".join(human_lines).lower()

    if not text.strip() or any(term in text for term in SKIP_TERMS):
        return None

    # Detect explicit negations before checking for positive keywords.
    no_humpback = any(term in text for term in NO_HUMPBACK_TERMS)
    no_vessel = any(term in text for term in NO_VESSEL_TERMS)

    if any(term in text for term in RESIDENT_TERMS):
        return "resident"
    if any(term in text for term in TRANSIENT_TERMS):
        return "transient"
    if "humpback" in text and not no_humpback:
        return "humpback"
    if any(term in text for term in BIRD_TERMS):
        return "bird"
    if any(term in text for term in HUMAN_TERMS):
        return "human"
    if any(term in text for term in HUM_TERMS) or HUM_WORD_PATTERN.search(text):
        return "hum"
    if any(term in text for term in VESSEL_TERMS) and not no_vessel:
        return "vessel"
    if "jingl" in text:
        return "jingle"
    if any(term in text for term in OTHER_TERMS):
        return "other"
    if "water" in text:
        return "water"
    # No positive signal found: the sound is ambient noise.
    return "water"


def _in_time_range(
    ts: datetime,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
) -> bool:
    """Return True if *ts* falls within [start_time, end_time] (inclusive, open bounds if None)."""
    return (start_time is None or ts >= start_time) and (end_time is None or ts <= end_time)


def process_false_positives(
    manual_samples_path: Path,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    detections_csv: str = DEFAULT_DETECTIONS_CSV,
    feed_filter: Optional[str] = None,
    for_training: bool = True,
    actual_category_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    destination_samples_path: Optional[Path] = None,
) -> dict[str, int]:
    """Process rejected OrcaHello detections in the selected timeframe.

    The returned summary includes ``confirmed`` and ``unreviewed`` counters for
    in-range detections that were not rejected, giving diagnostic context when
    ``rejected`` is zero (e.g., because human review is still pending).
    """
    summary = {
        "confirmed": 0,
        "unreviewed": 0,
        "rejected": 0,
        "download_failed": 0,
        "not_false_positive": 0,
        "processing_failed": 0,
        "unknown_class": 0,
        "whale_mismatch_segments": 0,
        "appended": 0,
        "duplicates": 0,
        "overlaps": 0,
    }
    normalized_category_filter = (
        actual_category_filter.strip().lower() if actual_category_filter else None
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
    if for_training:
        print(f"Loading podsai model from {model_path}...")
        model = get_model_inference(model_type="podsai", model_path=model_path)

    candidate_rows = []
    for feed in feeds:
        for detection in get_orcahello_detections(feed, start_time, end_time):
            if detection.timestamp is None:
                continue
            status = detection.status.lower()
            if status != "rejected":
                # Count in-range non-rejected detections for diagnostic context.
                if _in_time_range(detection.timestamp, start_time, end_time):
                    if status == "confirmed":
                        summary["confirmed"] += 1
                    else:
                        summary["unreviewed"] += 1
                continue
            # OrcaHello detections are returned in descending timestamp order.
            # Once we are older than the requested start time, the remaining
            # detections for this feed will also be too old.
            if start_time is not None and detection.timestamp < start_time:
                break
            if end_time is not None and detection.timestamp > end_time:
                continue

            timestamp_str_pst = format_timestamp_pst(
                detection.timestamp - timedelta(seconds=AUDIO_OFFSET_SECONDS)
            )
            summary["rejected"] += 1
            corrected_class = get_corrected_class(detection.comments, detection.tags)
            if corrected_class is None:
                print(f"Skipping {feed.node_name} {timestamp_str_pst}: could not determine corrected class from comment '{detection.comments}' and tags '{detection.tags}'.")
                summary["unknown_class"] += 1
                continue
            if normalized_category_filter and corrected_class != normalized_category_filter:
                continue

            with TemporaryDirectory() as temp_dir:
                if not for_training:
                    testing_start_timestamp = timestamp_str_pst
                    with redirect_stdout(io.StringIO()):
                        segment_row = add_testing_60s_sample(
                            node_name=feed.node_name,
                            start_timestamp=testing_start_timestamp,
                            detections_csv=detections_csv,
                            corrected_class=corrected_class,
                            fallback_description=detection.comments,
                            fallback_notes="fp_machine",
                            fallback_tags=detection.tags,
                        )
                    segment_rows = [segment_row]
                else:
                    # For the training set, we need to run PODS-AI inference on the 60-second WAV to find mismatched whale-class segments.
                    print(f"Checking rejected OrcaHello detection at {timestamp_str_pst}")

                    min_end_timestamp_pst_str = format_timestamp_pst(detection.timestamp + timedelta(seconds=60))
                    wav_path = download_60s_audio(node_name=feed.node_name, min_end_timestamp_pst_str=min_end_timestamp_pst_str, tmp_dir=temp_dir)
                    if wav_path is None:
                        print(f"Skipping {feed.node_name} {timestamp_str_pst}: failed to download audio.")
                        summary["download_failed"] += 1
                        continue

                    try:
                        inference = model.predict(wav_path)
                        if inference.get("global_prediction_label") != "resident":
                            print(
                                f"Continuing with {feed.node_name} {timestamp_str_pst}: "
                                "PODS-AI global prediction is not resident."
                            )
                            summary["not_false_positive"] += 1

                        print(
                            f"Running add_samples.py for {feed.node_name} {timestamp_str_pst} "
                            f"with corrected class '{corrected_class}'."
                        )

                        segment_rows = add_training_3s_samples(
                            wav_file=wav_path,
                            node_name=feed.node_name,
                            start_timestamp=timestamp_str_pst,
                            output_dir=str(output_dir),
                            model_path=model_path,
                            detections_csv=detections_csv,
                            model=model,
                            corrected_class=corrected_class,
                            fallback_description=detection.comments,
                            fallback_notes="fp_machine",
                            fallback_tags=detection.tags,
                        )
                    except Exception as exc:
                        print(f"Skipping {feed.node_name} {timestamp_str_pst}: processing failed ({exc}).")
                        summary["processing_failed"] += 1
                        continue

            # For training, only append whale-class subsegments whose predicted
            # category differs from the corrected class (mismatches). For testing,
            # add_testing_60s_sample already returns a single row with Category set
            # to the corrected_class; include that row directly in the append set
            # so testing samples are written.
            if not for_training:
                append_rows = list(segment_rows)
                # Count testing rows as processed segments for diagnostics.
                summary["whale_mismatch_segments"] += len(append_rows)
            else:
                mismatched_whale_rows = []
                for row in segment_rows:
                    row_category = row.get("Category")
                    if row_category not in WHALE_CLASSES or row_category == corrected_class:
                        continue
                    updated_row = dict(row)
                    updated_row["Category"] = corrected_class
                    mismatched_whale_rows.append(updated_row)

                append_rows = mismatched_whale_rows
                summary["whale_mismatch_segments"] += len(mismatched_whale_rows)

            candidate_rows.extend(append_rows)

    if destination_samples_path is None:
        destination_samples_path = Path(
            DEFAULT_TRAINING_SAMPLES_CSV
            if for_training
            else DEFAULT_TESTING_SAMPLES_CSV
        )
    sample_duration = (
        TRAINING_SAMPLE_DURATION_SECONDS
        if for_training
        else TESTING_SAMPLE_DURATION_SECONDS
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
        comparison_paths=(
            (
                Path(
                    DEFAULT_TESTING_SAMPLES_CSV
                    if for_training
                    else DEFAULT_TRAINING_SAMPLES_CSV
                ),
                TESTING_SAMPLE_DURATION_SECONDS
                if for_training
                else TRAINING_SAMPLE_DURATION_SECONDS,
            ),
        ),
    )
    summary["overlaps"] = overlaps

    appended, duplicates = append_manual_samples(
        manual_samples_path,
        proposed_rows,
        existing_uris,
    )
    summary["appended"] = appended
    summary["duplicates"] += duplicates

    fieldnames = [
        "Category",
        "NodeName",
        "StartTimestamp",
        "URI",
        "Description",
        "Notes",
        "Confidence",
        "Tags",
    ]
    print(f"Proposed rows for {destination_samples_path.as_posix()}:")
    csv_writer = csv.DictWriter(
        sys.stdout,
        fieldnames=fieldnames,
        lineterminator="\n",
        extrasaction="ignore",
    )
    csv_writer.writeheader()
    csv_writer.writerows(proposed_rows)

    return summary


def main() -> int:
    """Run the false-positive processing CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Process rejected OrcaHello resident detections looking for new training "
            "or testing samples.  For training samples, re-run PODS-AI on the "
            "60-second WAV, and append mismatched whale-class sub-segments to "
            "new_manual_training_samples.csv with a corrected class.  For testing "
            "append mismatched whale-class segments to new_manual_testing_samples.csv."
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
        default="2026_03_17_00_00_00_PST",
        metavar="YYYY_MM_DD_HH_MM_SS_PST|now",
        help=(
            "Include only detections with timestamp <= this value. "
            "Use 'now' to remove the upper bound."
        ),
    )
    parser.add_argument(
        "--manual-samples-csv",
        help="Path to new manual samples csv.",
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
        "--detections-csv",
        default=DEFAULT_DETECTIONS_CSV,
        help="Path to detections.csv for add_samples.py metadata lookups.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        metavar="CATEGORY",
        help="Process only detections whose inferred actual category matches this value.",
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

    summary = process_false_positives(
        manual_samples_path=Path(manual_samples_csv),
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        detections_csv=args.detections_csv,
        feed_filter=args.feed,
        for_training=(result_set == "training"),
        actual_category_filter=args.category,
        start_time=start_time,
        end_time=end_time
    )

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
