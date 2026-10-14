# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for shared manual-sample CSV helpers."""

import csv

from manual_samples_utils import filter_overlapping_samples


FIELDNAMES = ["Category", "NodeName", "StartTimestamp", "URI"]


def _row(timestamp: str, uri: str, node_name: str = "rpi_test") -> dict:
    return {
        "Category": "vessel",
        "NodeName": node_name,
        "StartTimestamp": timestamp,
        "URI": uri,
    }


def _write_rows(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_filters_duplicate_uri_in_destination(tmp_path):
    destination = tmp_path / "testing_60s_samples.csv"
    existing = _row("2026_10_02_12_16_52_PST", "https://example.com/existing")
    _write_rows(destination, [existing])
    candidate = _row(
        "2026_10_03_12_16_52_PST",
        "https://example.com/existing",
        node_name="rpi_other",
    )

    accepted, filtered = filter_overlapping_samples([candidate], destination, 60)

    assert accepted == []
    assert filtered == 1


def test_filters_duplicate_node_and_start_in_destination(tmp_path):
    destination = tmp_path / "testing_60s_samples.csv"
    existing = _row("2026_10_02_12_16_52_PST", "https://example.com/existing")
    _write_rows(destination, [existing])
    candidate = _row(
        "2026_10_02_12_16_52_PST",
        "https://example.com/candidate",
    )

    accepted, filtered = filter_overlapping_samples([candidate], destination, 60)

    assert accepted == []
    assert filtered == 1


def test_filters_overlap_on_same_node(tmp_path):
    destination = tmp_path / "testing_60s_samples.csv"
    _write_rows(
        destination,
        [_row("2025_01_01_00_00_00_PST", "https://example.com/existing")],
    )
    candidate = _row("2025_01_01_00_00_30_PST", "https://example.com/candidate")

    accepted, filtered = filter_overlapping_samples([candidate], destination, 60)

    assert accepted == []
    assert filtered == 1


def test_keeps_adjacent_clip_on_same_node(tmp_path):
    destination = tmp_path / "testing_60s_samples.csv"
    _write_rows(
        destination,
        [_row("2025_01_01_00_00_00_PST", "https://example.com/existing")],
    )
    candidate = _row("2025_01_01_00_01_00_PST", "https://example.com/candidate")

    accepted, filtered = filter_overlapping_samples([candidate], destination, 60)

    assert accepted == [candidate]
    assert filtered == 0


def test_keeps_overlap_on_different_node(tmp_path):
    destination = tmp_path / "testing_60s_samples.csv"
    _write_rows(
        destination,
        [_row("2025_01_01_00_00_00_PST", "https://example.com/existing")],
    )
    candidate = _row(
        "2025_01_01_00_00_30_PST",
        "https://example.com/candidate",
        node_name="rpi_other",
    )

    accepted, filtered = filter_overlapping_samples([candidate], destination, 60)

    assert accepted == [candidate]
    assert filtered == 0


def test_filters_intra_run_duplicates_and_overlaps(tmp_path):
    destination = tmp_path / "training_3s_samples.csv"
    first = _row("2025_01_01_00_00_00_PST", "https://example.com/first")
    exact_duplicate = _row("2025_01_01_00_00_00_PST", "https://example.com/duplicate")
    overlap = _row("2025_01_01_00_00_02_PST", "https://example.com/overlap")
    adjacent = _row("2025_01_01_00_00_03_PST", "https://example.com/adjacent")

    accepted, filtered = filter_overlapping_samples(
        [first, exact_duplicate, overlap, adjacent],
        destination,
        3,
    )

    assert accepted == [first, adjacent]
    assert filtered == 2


def test_accepts_rows_when_destination_does_not_exist(tmp_path):
    destination = tmp_path / "missing.csv"
    candidate = _row("2025_01_01_00_00_00_PST", "https://example.com/candidate")

    accepted, filtered = filter_overlapping_samples([candidate], destination, 60)

    assert accepted == [candidate]
    assert filtered == 0


def test_compares_intervals_safely_across_daylight_saving_transition(tmp_path):
    destination = tmp_path / "training_3s_samples.csv"
    _write_rows(
        destination,
        [_row("2025_03_09_01_59_59_PST", "https://example.com/existing")],
    )
    candidate = _row("2025_03_09_03_00_00_PST", "https://example.com/candidate")

    accepted, filtered = filter_overlapping_samples([candidate], destination, 3)

    assert accepted == []
    assert filtered == 1


def test_filters_training_candidate_overlapping_testing_manifest(tmp_path):
    testing = tmp_path / "testing_60s_samples.csv"
    _write_rows(
        testing,
        [_row("2025_01_01_00_00_00_PST", "https://example.com/testing")],
    )
    candidate = _row("2025_01_01_00_00_58_PST", "https://example.com/training")

    accepted, filtered = filter_overlapping_samples(
        [candidate],
        tmp_path / "training_3s_samples.csv",
        3,
        comparison_paths=((testing, 60),),
    )

    assert accepted == []
    assert filtered == 1


def test_filters_testing_candidate_overlapping_training_manifest(tmp_path):
    training = tmp_path / "training_3s_samples.csv"
    _write_rows(
        training,
        [_row("2025_01_01_00_00_00_PST", "https://example.com/training")],
    )
    candidate = _row("2025_01_01_00_00_00_PST", "https://example.com/testing")

    accepted, filtered = filter_overlapping_samples(
        [candidate],
        tmp_path / "testing_60s_samples.csv",
        60,
        comparison_paths=((training, 3),),
    )

    assert accepted == []
    assert filtered == 1


def test_cross_manifest_adjacent_clip_is_retained(tmp_path):
    training = tmp_path / "training_3s_samples.csv"
    _write_rows(
        training,
        [_row("2025_01_01_00_00_00_PST", "https://example.com/training")],
    )
    candidate = _row("2025_01_01_00_00_03_PST", "https://example.com/testing")

    accepted, filtered = filter_overlapping_samples(
        [candidate],
        tmp_path / "testing_60s_samples.csv",
        60,
        comparison_paths=((training, 3),),
    )

    assert accepted == [candidate]
    assert filtered == 0
