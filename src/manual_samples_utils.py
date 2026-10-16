#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Shared helpers for reading and appending manual_samples.csv rows."""

import csv
from audio_utils import parse_timestamp_pst
from datetime import timedelta
from pathlib import Path
from typing import Iterable

CSV_FIELDNAMES = [
    "Category",
    "NodeName",
    "StartTimestamp",
    "URI",
    "Description",
    "Notes",
    "Confidence",
    "Tags",
]


def _sample_key(row: dict) -> tuple[str, str] | None:
    """Return a normalized node/timestamp key when both fields are present."""
    node_name = (row.get("NodeName") or "").strip().casefold()
    timestamp = (row.get("StartTimestamp") or "").strip()
    if not node_name or not timestamp:
        return None
    return node_name, timestamp


def filter_overlapping_samples(
    rows: list[dict],
    destination_path: Path,
    duration_seconds: int,
    comparison_paths: Iterable[tuple[Path, int]] = (),
) -> tuple[list[dict], int]:
    """Remove samples duplicated by or overlapping the destination manifests.

    Samples use half-open intervals, so adjacent clips are retained. Accepted
    rows are registered immediately to also reject overlaps within ``rows``.
    ``comparison_paths`` contains other manifests and their sample durations;
    it is used for cross-manifest validation without loading DCLDE data.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    destination_rows: list[dict] = []
    comparison_manifests: list[tuple[list[dict], int]] = []
    seen_paths: set[Path] = set()
    for manifest_index, (manifest_path, manifest_duration) in enumerate(
        [(destination_path, duration_seconds), *comparison_paths]
    ):
        manifest_path = Path(manifest_path)
        if manifest_path in seen_paths:
            continue
        seen_paths.add(manifest_path)
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                rows_from_manifest = list(csv.DictReader(handle))
                if manifest_index == 0:
                    destination_rows = rows_from_manifest
                else:
                    comparison_manifests.append((rows_from_manifest, manifest_duration))
        except (OSError, UnicodeError, csv.Error) as exc:
            print(
                f"Warning: Failed to read sample manifest from {manifest_path}: {exc}"
            )

    existing_entries = [
        (row, manifest_duration)
        for rows_from_manifest, manifest_duration in [
            (destination_rows, duration_seconds), *comparison_manifests
        ]
        for row in rows_from_manifest
    ]
    uris: set[str] = set()
    sample_keys: set[tuple[str, str]] = set()
    for row, _row_duration in existing_entries:
        uri = (row.get("URI") or "").strip()
        if uri:
            uris.add(uri)
        key = _sample_key(row)
        if key is not None:
            sample_keys.add(key)

    accepted_rows = []
    filtered = 0
    for row in rows:
        uri = (row.get("URI") or "").strip()
        key = _sample_key(row)
        duplicate = bool(uri and uri in uris) or bool(key and key in sample_keys)
        if not duplicate:
            duplicate = any(
                _samples_overlap(row, duration_seconds, other_row, other_duration)
                for other_row, other_duration in [
                    *existing_entries,
                    *((accepted_row, duration_seconds) for accepted_row in accepted_rows),
                ]
                if key is not None and _sample_key(other_row) is not None
                and _sample_key(other_row)[0] == key[0]
            )

        if duplicate:
            filtered += 1
            continue

        accepted_rows.append(row)
        if uri:
            uris.add(uri)
        if key is not None:
            sample_keys.add(key)

    return accepted_rows, filtered


def _samples_overlap(
    left: dict,
    left_duration: int,
    right: dict,
    right_duration: int,
) -> bool:
    """Compare sample windows using download_wavs validation rules."""
    from download_wavs import CSVRow, validate_no_overlaps

    left_key = _sample_key(left)
    right_key = _sample_key(right)
    if left_key is None or right_key is None or left_key[0] != right_key[0]:
        return False

    def as_csv_row(row: dict, key: tuple[str, str]) -> CSVRow:
        return CSVRow(
            category=row.get("Category") or "",
            node_name=key[0],
            timestamp_pst=(row.get("StartTimestamp") or "").strip(),
            uri=(row.get("URI") or "").strip(),
            description=row.get("Description") or "",
            notes=row.get("Notes") or "",
            confidence=row.get("Confidence") or "",
        )

    left_row = as_csv_row(left, left_key)
    right_row = as_csv_row(right, right_key)
    try:
        parse_timestamp_pst(left_row.timestamp_pst)
        parse_timestamp_pst(right_row.timestamp_pst)
    except ValueError:
        return False
    if left_duration == right_duration == 3:
        training_rows, testing_rows = [left_row, right_row], []
    elif left_duration == right_duration == 60:
        training_rows, testing_rows = [], [left_row, right_row]
    elif left_duration == 3 and right_duration == 60:
        training_rows, testing_rows = [left_row], [right_row]
    elif left_duration == 60 and right_duration == 3:
        training_rows, testing_rows = [right_row], [left_row]
    else:
        left_start = parse_timestamp_pst(left_row.timestamp_pst)
        right_start = parse_timestamp_pst(right_row.timestamp_pst)
        left_end = left_start + timedelta(seconds=left_duration)
        right_end = right_start + timedelta(seconds=right_duration)
        return max(left_start, right_start) < min(left_end, right_end)

    try:
        validate_no_overlaps(training_rows, testing_rows)
    except ValueError as exc:
        if "Detected overlapping sample windows:" in str(exc):
            return True
        raise
    return False


def load_existing_uris(manual_samples_path: Path) -> set[str]:
    """Load existing manual-sample URIs from a CSV file path."""
    if not manual_samples_path.exists():
        return set()

    try:
        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return {
                (row.get("URI") or "").strip()
                for row in reader
                if (row.get("URI") or "").strip()
            }
    except (OSError, UnicodeError, csv.Error) as exc:
        print(
            f"Warning: Failed to read existing manual sample URIs from "
            f"{manual_samples_path}: {exc}"
        )
        return set()


def append_manual_samples(
    manual_samples_path: Path,
    rows: list[dict],
    existing_uris: set[str],
    fieldnames: list[str] | None = None,
) -> tuple[int, int]:
    """Append rows to manual_samples.csv, skipping URIs already present."""
    fieldnames = fieldnames or CSV_FIELDNAMES
    rows_to_append = []
    duplicates = 0

    for row in rows:
        uri = (row.get("URI") or "").strip()
        if uri in existing_uris:
            duplicates += 1
            continue
        existing_uris.add(uri)
        rows_to_append.append({field: row.get(field, "") for field in fieldnames})

    if not rows_to_append:
        return 0, duplicates

    manual_samples_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not manual_samples_path.exists() or manual_samples_path.stat().st_size == 0
    )
    with open(manual_samples_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerows(rows_to_append)

    return len(rows_to_append), duplicates
