# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for process_false_negatives.py."""

import csv
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from add_samples import add_training_3s_samples
from make_csv import OrcaHelloDetection
from orcasite_feeds import OrcasiteFeed
from process_false_negatives import (
    append_manual_samples,
    is_orcahello_resident_prediction,
    process_false_negatives,
)


def _make_feed() -> OrcasiteFeed:
    """Build a minimal test feed."""
    return OrcasiteFeed(
        id="feed_1",
        name="Test Feed",
        node_name="rpi_test",
        slug="test-feed",
        bucket="audio-orcasound-net",
        bucket_region="us-west-2",
        visible=True,
        location=(47.0, -122.0),
    )


class TestOrcaHelloResidentPrediction:
    """Tests for OrcaHello resident-label normalization."""

    def test_resident_label_variants(self):
        """Both resident and whale labels should be treated as resident detections."""
        assert is_orcahello_resident_prediction("resident") is True
        assert is_orcahello_resident_prediction("whale") is True
        assert is_orcahello_resident_prediction("other") is False


class TestAppendManualSamples:
    """Tests for append_manual_samples."""

    def test_skips_duplicate_uris(self, tmp_path):
        """Rows with URIs already in the file should not be appended again."""
        manual_samples_path = tmp_path / "manual_samples.csv"
        manual_samples_path.write_text(
            "Category,NodeName,StartTimestamp,URI,Description,Notes,Confidence\n"
            "resident,rpi_test,2025_01_01_00_00_00_PST,https://example.com/existing,desc,notes,90.0\n",
            encoding="utf-8",
        )
        existing_uris = {"https://example.com/existing"}
        rows = [
            {
                "Category": "resident",
                "NodeName": "rpi_test",
                "StartTimestamp": "2025_01_01_00_00_00_PST",
                "URI": "https://example.com/existing",
                "Description": "desc",
                "Notes": "notes",
                "Confidence": "90.0",
            },
            {
                "Category": "resident",
                "NodeName": "rpi_test",
                "StartTimestamp": "2025_01_01_00_00_02_PST",
                "URI": "https://example.com/new",
                "Description": "desc",
                "Notes": "notes",
                "Confidence": "91.0",
            },
        ]

        appended, duplicates = append_manual_samples(manual_samples_path, rows, existing_uris)

        assert appended == 1
        assert duplicates == 1
        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            written_rows = list(csv.DictReader(handle))
        assert len(written_rows) == 2
        assert written_rows[-1]["URI"] == "https://example.com/new"


class TestProcessFalseNegatives:
    """Integration-style tests for process_false_negatives."""

    @pytest.mark.parametrize(
        ("corrected_class", "printed_category"),
        [("resident", "resident"), (None, "water")],
    )
    def test_prints_expected_category_and_returns_prediction(
        self, tmp_path, capsys, corrected_class, printed_category
    ):
        """CSV output uses a correction when supplied without changing returned predictions."""
        segment_path = tmp_path / "segment.wav"
        segment_path.write_bytes(b"segment")
        timestamp = "2025_01_01_04_00_00_PST"
        podsai_model = Mock()
        podsai_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.9,
        }

        with patch(
            "add_samples.split_wav_into_segments",
            return_value=[(segment_path, timestamp)],
        ), patch("add_samples.lookup_detection_in_csv", return_value=None), patch(
            "add_samples.generate_uri", return_value="https://example.com/segment"
        ):
            rows = add_training_3s_samples(
                wav_file="input.wav",
                node_name="rpi_test",
                start_timestamp=timestamp,
                output_dir=str(tmp_path),
                model=podsai_model,
                corrected_class=corrected_class,
            )

        assert rows[0]["Category"] == "water"
        output = capsys.readouterr().out
        csv_section = output.split("Segments in manual_samples.csv format:\n", 1)[1]
        printed_rows = list(csv.DictReader(csv_section.splitlines()))
        assert [row["Category"] for row in printed_rows] == [printed_category]

    def test_uses_retrying_feed_fetcher(self, tmp_path):
        """Processing should use the shared retrying feed-fetch helper."""
        feed = _make_feed()
        with patch(
            "process_false_negatives.get_orcasite_feeds_with_retry",
            return_value=[feed],
        ) as mock_get_feeds, patch(
            "process_false_negatives.get_model_inference",
            return_value=Mock(),
        ), patch("process_false_negatives.get_orcahello_detections", return_value=[]):
            summary = process_false_negatives(
                manual_samples_path=tmp_path / "manual_samples.csv",
                output_dir=tmp_path / "segments",
            )

        assert summary["confirmed"] == 0
        mock_get_feeds.assert_called_once_with()

    def test_appends_and_prints_orcahello_resident_segments_missed_by_podsai(self, tmp_path, capsys):
        """Segments predicted resident by OrcaHello and non-resident by PODS-AI are appended."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="confirmed",
            comments="",
            tags=""
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        segment_dir = tmp_path / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        manual_samples_path = tmp_path / "manual_samples.csv"

        segment_rows = [
            {
                "Category": "resident",
                "NodeName": "rpi_test",
                "StartTimestamp": "2025_01_01_04_00_00_PST",
                "URI": "https://example.com/resident",
                "Description": "desc",
                "Notes": "manual",
                "Confidence": "95.0",
                "Tags": "resident",
            },
            {
                "Category": "transient",
                "NodeName": "rpi_test",
                "StartTimestamp": "2025_01_01_04_00_02_PST",
                "URI": "https://example.com/transient",
                "Description": "desc",
                "Notes": "manual",
                "Confidence": "90.0",
                "Tags": "transient",
            },
            {
                "Category": "water",
                "NodeName": "rpi_test",
                "StartTimestamp": "2025_01_01_04_00_04_PST",
                "URI": "https://example.com/water",
                "Description": "desc",
                "Notes": "manual",
                "Confidence": "10.0",
                "Tags": "water",
            },
        ]

        for row in segment_rows:
            seg_name = f"rpi-test_{row['StartTimestamp']}.wav"
            (segment_dir / seg_name).write_bytes(b"segment")

        podsai_model = Mock()
        podsai_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.9,
        }

        orcahello_model = Mock()

        def _orcahello_predict(segment_path: str):
            if segment_path.endswith("2025_01_01_04_00_00_PST.wav"):
                return {"global_prediction_label": "whale", "global_confidence": 0.9}
            if segment_path.endswith("2025_01_01_04_00_02_PST.wav"):
                return {"global_prediction_label": "resident", "global_confidence": 0.9}
            if segment_path.endswith("2025_01_01_04_00_04_PST.wav"):
                return {"global_prediction_label": "whale", "global_confidence": 0.9}
            return {"global_prediction_label": "other", "global_confidence": 0.9}

        orcahello_model.predict.side_effect = _orcahello_predict

        def _get_model(*args, **kwargs):
            return podsai_model if kwargs.get("model_type") == "podsai" else orcahello_model

        with patch("process_false_negatives.get_model_inference", side_effect=_get_model), \
             patch("process_false_negatives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_negatives.get_orcahello_detections", return_value=[detection]), \
             patch("process_false_negatives.download_60s_audio", return_value=str(wav_path)), \
             patch("process_false_negatives.add_training_3s_samples", return_value=segment_rows) as mock_add_samples:
            summary = process_false_negatives(
                manual_samples_path=manual_samples_path,
                output_dir=segment_dir,
            )

        assert summary["confirmed"] == 1
        assert summary["not_false_negative"] == 0
        assert summary["mismatched_segments"] == 2
        assert summary["wrong_whale_class_segments"] == 1
        assert summary["appended"] == 1
        assert summary["duplicates"] == 0
        assert summary["overlaps"] == 1
        mock_add_samples.assert_called_once()
        assert mock_add_samples.call_args.kwargs["corrected_class"] == "resident"
        assert mock_add_samples.call_args.kwargs["fallback_description"] == detection.comments
        assert mock_add_samples.call_args.kwargs["fallback_notes"] == "tp_machine"

        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert [row["URI"] for row in rows] == [
            "https://example.com/transient",
        ]
        assert all(row["Category"] == "resident" for row in rows)

        output = capsys.readouterr().out
        marker = "Proposed rows for output/csv/training_3s_samples.csv:"
        assert marker in output
        csv_section = output.split(marker + "\n", 1)[1]
        proposed_rows = list(csv.DictReader(csv_section.splitlines()))
        assert [row["URI"] for row in proposed_rows] == [
            "https://example.com/transient",
        ]
        assert all(row["Category"] == "resident" for row in proposed_rows)

    def test_filters_training_candidate_in_testing_manifest(self, tmp_path):
        """False-negative training candidates overlapping testing are skipped."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="confirmed",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        segment_dir = tmp_path / "segments"
        segment_dir.mkdir()
        segment_timestamp = "2025_01_01_03_59_58_PST"
        (segment_dir / f"rpi-test_{segment_timestamp}.wav").write_bytes(b"segment")
        testing_path = tmp_path / "testing_60s_samples.csv"
        with open(testing_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Category", "NodeName", "StartTimestamp", "URI"])
            writer.writeheader()
            writer.writerow({
                "Category": "vessel",
                "NodeName": "rpi_test",
                "StartTimestamp": "2025_01_01_03_59_00_PST",
                "URI": "https://example.com/testing",
            })
        segment = {
            "Category": "transient",
            "NodeName": "rpi_test",
            "StartTimestamp": segment_timestamp,
            "URI": "https://example.com/training",
        }
        podsai_model = Mock()
        podsai_model.predict.return_value = {"global_prediction_label": "water"}
        orcahello_model = Mock()
        orcahello_model.predict.return_value = {"global_prediction_label": "resident"}

        def get_model(*args, **kwargs):
            return podsai_model if kwargs["model_type"] == "podsai" else orcahello_model

        with patch("process_false_negatives.DEFAULT_TESTING_SAMPLES_CSV", str(testing_path)), \
             patch("process_false_negatives.get_model_inference", side_effect=get_model), \
             patch("process_false_negatives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_negatives.get_orcahello_detections", return_value=[detection]), \
             patch("process_false_negatives.download_60s_audio", return_value=str(wav_path)), \
             patch("process_false_negatives.add_training_3s_samples", return_value=[segment]):
            summary = process_false_negatives(
                manual_samples_path=tmp_path / "manual.csv",
                output_dir=segment_dir,
            )

        assert summary["appended"] == 0
        assert summary["overlaps"] == 1

    def test_skips_when_full_clip_prediction_is_resident(self, tmp_path):
        """When full-clip PODS-AI prediction is resident, processing is skipped."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="confirmed",
            comments="",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")

        podsai_model = Mock()
        podsai_model.predict.return_value = {
            "global_prediction_label": "resident",
            "global_confidence": 0.95,
        }
        orcahello_model = Mock()

        def _get_model(*args, **kwargs):
            return podsai_model if kwargs.get("model_type") == "podsai" else orcahello_model

        with patch("process_false_negatives.get_model_inference", side_effect=_get_model), \
             patch("process_false_negatives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_negatives.get_orcahello_detections", return_value=[detection]), \
             patch("process_false_negatives.download_60s_audio", return_value=str(wav_path)), \
             patch("process_false_negatives.add_training_3s_samples") as mock_add_samples:
            summary = process_false_negatives(
                manual_samples_path=tmp_path / "manual_samples.csv",
                output_dir=tmp_path / "segments",
            )

        assert summary["confirmed"] == 1
        assert summary["not_false_negative"] == 1
        assert summary["mismatched_segments"] == 0
        assert summary["appended"] == 0
        mock_add_samples.assert_not_called()
        orcahello_model.predict.assert_not_called()

    def test_filters_by_predicted_category_before_segment_processing(self, tmp_path):
        """A predicted-category filter should skip non-matching detections after inference."""
        feed = _make_feed()
        matching_detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            status="confirmed",
            comments="",
        )
        non_matching_detection = OrcaHelloDetection(
            id="det_2",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="confirmed",
            comments="",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        segment_dir = tmp_path / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        manual_samples_path = tmp_path / "manual_samples.csv"

        segment_timestamp = "2025_01_01_04_05_02_PST"
        (segment_dir / f"rpi-test_{segment_timestamp}.wav").write_bytes(b"segment")

        podsai_model = Mock()
        podsai_model.predict.side_effect = [
            {"global_prediction_label": "WATER", "global_confidence": 0.9},
            {"global_prediction_label": "transient", "global_confidence": 0.9},
        ]

        orcahello_model = Mock()
        orcahello_model.predict.return_value = {
            "global_prediction_label": "resident",
            "global_confidence": 0.9,
        }

        def _get_model(*args, **kwargs):
            return podsai_model if kwargs.get("model_type") == "podsai" else orcahello_model

        with patch("process_false_negatives.get_model_inference", side_effect=_get_model), \
             patch("process_false_negatives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch(
                 "process_false_negatives.get_orcahello_detections",
                 return_value=[matching_detection, non_matching_detection],
             ), \
             patch("process_false_negatives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_negatives.add_training_3s_samples",
                 return_value=[
                     {
                         "Category": "transient",
                         "NodeName": "rpi_test",
                         "StartTimestamp": segment_timestamp,
                         "URI": "https://example.com/new",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "91.0",
                     },
                 ],
             ) as mock_add_samples:
            summary = process_false_negatives(
                manual_samples_path=manual_samples_path,
                output_dir=segment_dir,
                predicted_category_filter="water",
            )

        assert summary["confirmed"] == 2
        assert summary["mismatched_segments"] == 1
        assert summary["appended"] == 1
        assert podsai_model.predict.call_count == 2
        assert orcahello_model.predict.call_count == 1
        assert mock_add_samples.call_count == 1
        assert mock_add_samples.call_args.kwargs["start_timestamp"] == "2025_01_01_04_04_58_PST"
