# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for process_false_positives.py."""

import csv
from datetime import datetime, timezone
from unittest.mock import patch

from make_csv import OrcaHelloDetection
from orcasite_feeds import OrcasiteFeed
from process_false_positives import (
    append_manual_samples,
    get_corrected_class,
    process_false_positives,
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


class TestGetCorrectedClass:
    """Tests for get_corrected_class."""

    def test_detects_transient_keywords(self):
        """Transient-related comments should map to transient."""
        assert get_corrected_class("Likely transient calls from Bigg's whales.", None) == "transient"

    def test_detects_vessel_keywords(self):
        """Boat-like comments should map to vessel."""
        assert get_corrected_class("This is boat noise, not whales.", None) == "vessel"

    def test_returns_none_for_unsure_comments(self):
        """Ambiguous comments should be skipped."""
        assert get_corrected_class("Not sure what this is.", None) is None

    def test_strips_ai_prefix_line_before_inferring_class(self):
        """An 'AI: humpback' prefix line should be ignored so it does not produce 'humpback'."""
        # Only the auto-generated line is present; nothing else to infer from → None.
        assert get_corrected_class("AI: humpback", None) is None

    def test_no_humpback_alone_returns_water(self):
        """'No humpback' alone gives no positive signal → water."""
        assert get_corrected_class("No humpback", None) == "water"

    def test_ai_prefix_plus_no_humpback_returns_water(self):
        """'AI: humpback\\nNo humpback' is a false humpback with no other signal → water."""
        assert get_corrected_class("AI: humpback\nNo humpback", None) == "water"

    def test_no_humpback_nor_vessel_returns_water(self):
        """'No humpback nor vessel' with no other signal should resolve to water."""
        assert get_corrected_class("No humpback nor vessel", None) == "water"

    def test_ai_prefix_plus_no_humpback_nor_vessel_returns_water(self):
        """Full 'AI: humpback\\nNo humpback nor vessel' comment should resolve to water."""
        assert get_corrected_class("AI: humpback\nNo humpback nor vessel", None) == "water"

    def test_no_humpback_with_vessel_positive_returns_vessel(self):
        """'No humpback' with an unambiguous vessel keyword should still resolve to vessel."""
        assert get_corrected_class("No humpback. Boat noise.", None) == "vessel"

    def test_humpback_positive_without_negation_still_returns_humpback(self):
        """A plain 'humpback' mention (no negation) should still map to humpback."""
        assert get_corrected_class("Humpback whale song", None  ) == "humpback"

    def test_no_vessel_prevents_vessel_match(self):
        """'No vessel' should suppress the vessel keyword match → water."""
        assert get_corrected_class("No vessel here", None) == "water"

    def test_tags_override_comments(self):
        """Tags should take precedence over comments."""
        assert get_corrected_class("Humpback whale song", "vessel") == "vessel"

    def test_tags_case_and_whitespace_and_multiple(self):
        """Tags are split on ';', stripped, and matched case-insensitively; first matching tag returned."""
        assert get_corrected_class("", "  Humpback  ; other") == "humpback"

    def test_tags_unknown_returns_water(self):
        """If tags contain no known classes, return 'water'."""
        assert get_corrected_class("Boat noise", "unknown;maybe") == "water"

    def test_tags_first_match_returned_within_same_tier(self):
        """First matching tag should be returned when competing tags are in the same priority tier."""
        assert get_corrected_class("", "vessel;water") == "vessel"

    def test_tags_negative_tier_takes_priority_over_background_tier(self):
        """A sporadic-event tag (jingle, bird) wins over a constant background tag (vessel, etc.),
        regardless of position, matching the positive/negative/background priority used for live
        model predictions in podsai_inference.py."""
        assert get_corrected_class("", "vessel;jingle") == "jingle"
        assert get_corrected_class("", "water;bird") == "bird"

    def test_tags_background_hum_loses_to_negative_tier(self):
        """'hum' is background tier, so it loses to a sporadic-event tag like jingle or bird."""
        assert get_corrected_class("", "hum;jingle") == "jingle"

    def test_tags_whale_species_takes_priority_regardless_of_position(self):
        """A whale-species tag wins over a noise tag even when the noise tag is listed first."""
        assert get_corrected_class("", "vessel;resident") == "resident"

    def test_tags_whale_species_takes_priority_over_hum(self):
        """A whale-species tag wins over 'hum' even when hum appears first, since a real call was present."""
        assert get_corrected_class("", "hum;transient;orca;whale") == "transient"

    def test_tags_hum_returns_hum(self):
        """A bare 'hum' tag resolves to the 'hum' category."""
        assert get_corrected_class("", "hum") == "hum"

    def test_tags_60hz_alias_returns_hum(self):
        """The older '60Hz' tag is treated as an alias for 'hum'."""
        assert get_corrected_class("", "60Hz") == "hum"

    def test_tags_hum_and_60hz_together_returns_hum_once(self):
        """A detection tagged with both 'hum' and '60Hz' still resolves to 'hum'."""
        assert get_corrected_class("", "ship;60Hz;hum") == "hum"

    def test_comments_hum_word_returns_hum(self):
        """A standalone 'hum' mention in comments resolves to 'hum'."""
        assert get_corrected_class("Loud hum in background.", None) == "hum"

    def test_comments_electrical_keyword_returns_hum(self):
        """'Electrical' noise mentioned in comments resolves to 'hum'."""
        assert get_corrected_class("Electrical noise, nothing else.", None) == "hum"

    def test_comments_electric_keyword_returns_hum(self):
        """'Electric' (without the '-al' suffix) resolves to 'hum'."""
        assert get_corrected_class("Electric buzz, nothing else.", None) == "hum"

    def test_comments_60_hz_with_space_returns_hum(self):
        """'60 Hz' (with a space) resolves to 'hum'."""
        assert get_corrected_class("60 Hz interference, nothing else.", None) == "hum"

    def test_comments_60_hz_with_hyphen_returns_hum(self):
        """'60-Hz' (with a hyphen) resolves to 'hum'."""
        assert get_corrected_class("60-Hz interference, nothing else.", None) == "hum"

    def test_comments_hum_does_not_match_inside_humpback(self):
        """'hum' must not match as a substring of 'humpback' in free-text comments."""
        assert get_corrected_class("Humpback calls amid electrical hum.", None) == "humpback"

    def test_comments_hum_does_not_match_inside_human(self):
        """'hum' must not match as a substring of 'human' in free-text comments."""
        assert get_corrected_class("Human voices, electrical noise.", None) == "human"

    def test_comments_electrical_alongside_negated_humpback_returns_hum(self):
        """An explicit 'electrical' mention still resolves to 'hum' even alongside an unrelated, negated 'No humpback'."""
        assert get_corrected_class("Electrical noise. No humpback nor vessel.", None) == "hum"

    def test_comments_humming_returns_hum(self):
        """Inflected forms like 'humming' still resolve to 'hum'."""
        assert get_corrected_class("Constant humming in background.", None) == "hum"


class TestAppendManualSamples:
    """Tests for append_manual_samples."""

    def test_skips_duplicate_uris(self, tmp_path):
        """Rows with URIs already in the file should not be appended again."""
        manual_samples_path = tmp_path / "manual_samples.csv"
        manual_samples_path.write_text(
            "Category,NodeName,StartTimestamp,URI,Description,Notes,Confidence\n"
            "vessel,rpi_test,2025_01_01_00_00_00_PST,https://example.com/existing,desc,notes,90.0\n",
            encoding="utf-8",
        )
        existing_uris = {"https://example.com/existing"}
        rows = [
            {
                "Category": "vessel",
                "NodeName": "rpi_test",
                "StartTimestamp": "2025_01_01_00_00_00_PST",
                "URI": "https://example.com/existing",
                "Description": "desc",
                "Notes": "notes",
                "Confidence": "90.0",
            },
            {
                "Category": "vessel",
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


class TestProcessFalsePositives:
    """Integration-style tests for process_false_positives."""

    def test_uses_retrying_feed_fetcher(self, tmp_path):
        """Processing should use the shared retrying feed-fetch helper."""
        feed = _make_feed()
        with patch(
            "process_false_positives.get_orcasite_feeds_with_retry",
            return_value=[feed],
        ) as mock_get_feeds, patch(
            "process_false_positives.get_model_inference"
        ), patch("process_false_positives.get_machine_detections", return_value=[]):
            summary = process_false_positives(
                manual_samples_path=tmp_path / "manual_samples.csv",
                output_dir=tmp_path / "segments",
            )

        assert summary["rejected"] == 0
        mock_get_feeds.assert_called_once_with()

    def test_testing_mode_appends_and_prints_contiguous_proposal_csv(self, tmp_path, capsys):
        """Testing mode should append the single corrected 60-second row."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
            tags="vessel",
        )
        manual_samples_path = tmp_path / "manual_testing_samples.csv"
        testing_row = {
            "Category": "vessel",
            "NodeName": "rpi_test",
            "StartTimestamp": "2025_01_01_03_59_58_PST",
            "URI": "https://example.com/testing",
            "Description": "Boat noise from a nearby vessel.",
            "Notes": "fp_machine",
            "Confidence": 100,
            "Tags": "vessel",
        }

        def add_testing_sample(**kwargs):
            print("testing diagnostic")
            return testing_row

        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_positives.get_machine_detections", return_value=[detection]), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch("process_false_positives.add_training_3s_samples") as mock_add_training, \
             patch(
                 "process_false_positives.add_testing_60s_sample",
                 side_effect=add_testing_sample,
             ) as mock_add_testing:
            mock_get_model.return_value.predict.return_value = {
                "global_prediction_label": "resident",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
                for_training=False,
            )

        mock_get_model.assert_called_once()
        mock_get_model.return_value.predict.assert_called_once_with(str(wav_path))
        mock_add_training.assert_not_called()
        assert mock_add_testing.call_count == 1
        assert mock_add_testing.call_args.kwargs["corrected_class"] == "vessel"
        assert mock_add_testing.call_args.kwargs["start_timestamp"] == "2025_01_01_03_59_58_PST"
        assert mock_add_testing.call_args.kwargs["fallback_description"] == detection.comments
        assert mock_add_testing.call_args.kwargs["fallback_notes"] == "fp_machine"
        assert mock_add_testing.call_args.kwargs["fallback_tags"] == detection.tags
        assert summary["rejected"] == 1
        assert summary["whale_mismatch_segments"] == 1
        assert summary["appended"] == 1
        assert summary["duplicates"] == 0

        output = capsys.readouterr().out
        marker = "Proposed rows for output/csv/testing_60s_samples.csv:"
        assert "testing diagnostic" not in output
        csv_section = output.split(marker + "\n", 1)[1]
        proposed_rows = list(csv.DictReader(csv_section.splitlines()))
        assert len(proposed_rows) == 1
        assert proposed_rows[0]["URI"] == "https://example.com/testing"
        assert proposed_rows[0]["Tags"] == "vessel"

        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["Category"] == "vessel"
        assert rows[0]["URI"] == "https://example.com/testing"

    def test_testing_mode_filters_row_already_in_destination(self, tmp_path, capsys):
        """A row copied into testing_60s_samples.csv is not proposed again."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
            tags="vessel",
        )
        testing_row = {
            "Category": "vessel",
            "NodeName": "rpi_test",
            "StartTimestamp": "2025_01_01_03_59_58_PST",
            "URI": "https://example.com/testing",
            "Description": "Boat noise from a nearby vessel.",
            "Notes": "fp_machine",
            "Confidence": 100,
            "Tags": "vessel",
        }
        destination_path = tmp_path / "testing_60s_samples.csv"
        with open(destination_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=testing_row)
            writer.writeheader()
            writer.writerow(testing_row)

        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_positives.get_machine_detections", return_value=[detection]), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch("process_false_positives.add_testing_60s_sample", return_value=testing_row):
            mock_get_model.return_value.predict.return_value = {
                "global_prediction_label": "resident",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=tmp_path / "manual_testing_samples.csv",
                output_dir=tmp_path / "segments",
                for_training=False,
                destination_samples_path=destination_path,
            )

        mock_get_model.assert_called_once()
        mock_get_model.return_value.predict.assert_called_once_with(str(wav_path))
        assert summary["appended"] == 0
        assert summary["overlaps"] == 1
        assert not (tmp_path / "manual_testing_samples.csv").exists()
        output = capsys.readouterr().out
        marker = f"Proposed rows for {destination_path.as_posix()}:"
        proposed_rows = list(csv.DictReader(output.split(marker + "\n", 1)[1].splitlines()))
        assert proposed_rows == []

    def test_testing_mode_skips_when_current_model_is_not_resident(self, tmp_path):
        """Testing mode should skip rows that the current PODS-AI model no longer flags as resident."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
            tags="vessel",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_positives.get_machine_detections", return_value=[detection]), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch("process_false_positives.add_testing_60s_sample") as mock_add_testing:
            mock_get_model.return_value.predict.return_value = {
                "global_prediction_label": "water",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=tmp_path / "manual_testing_samples.csv",
                output_dir=tmp_path / "segments",
                for_training=False,
            )

        mock_add_testing.assert_not_called()
        assert summary["not_false_positive"] == 1
        assert summary["appended"] == 0
        assert not (tmp_path / "manual_testing_samples.csv").exists()

    def test_training_mode_filters_candidate_in_testing_manifest(self, tmp_path):
        """Training candidates overlapping testing samples are not appended."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
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
            "Category": "resident",
            "NodeName": "rpi_test",
            "StartTimestamp": "2025_01_01_03_59_58_PST",
            "URI": "https://example.com/training",
        }
        with patch("process_false_positives.DEFAULT_TESTING_SAMPLES_CSV", str(testing_path)), \
             patch("process_false_positives.get_model_inference") as get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_positives.get_machine_detections", return_value=[detection]), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch("process_false_positives.add_training_3s_samples", return_value=[segment]):
            get_model.return_value.predict.return_value = {"global_prediction_label": "resident"}
            summary = process_false_positives(
                manual_samples_path=tmp_path / "manual.csv",
                output_dir=tmp_path / "segments",
            )

        assert summary["appended"] == 0
        assert summary["overlaps"] == 1

    def test_appends_only_mismatched_whale_segments_with_corrected_class(self, tmp_path):
        """Whale-class segments should be rewritten unless they already match the corrected class."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        manual_samples_path = tmp_path / "manual_samples.csv"
        manual_samples_path.write_text(
            "Category,NodeName,StartTimestamp,URI,Description,Notes,Confidence\n"
            "vessel,rpi_test,2025_01_01_04_00_00_PST,https://example.com/existing,desc,manual,90.0\n",
            encoding="utf-8",
        )

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_positives.get_machine_detections", return_value=[detection]), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_positives.add_training_3s_samples",
                 return_value=[
                     {
                         "Category": "resident",
                         "NodeName": "rpi_test",
                         "StartTimestamp": "2025_01_01_04_00_00_PST",
                         "URI": "https://example.com/existing",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "90.0",
                     },
                     {
                         "Category": "resident",
                         "NodeName": "rpi_test",
                         "StartTimestamp": "2025_01_01_04_00_02_PST",
                         "URI": "https://example.com/new",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "91.0",
                     },
                     {
                         "Category": "water",
                         "NodeName": "rpi_test",
                         "StartTimestamp": "2025_01_01_04_00_04_PST",
                         "URI": "https://example.com/water",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "20.0",
                      },
                  ],
              ) as mock_add_samples:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = {
                "global_prediction_label": "resident",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
                start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                end_time=datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            )

        assert summary["rejected"] == 1
        assert summary["whale_mismatch_segments"] == 2
        assert summary["appended"] == 1
        assert summary["duplicates"] == 1
        assert mock_model.predict.call_count == 1
        assert mock_add_samples.call_args.kwargs["model"] is mock_model
        assert mock_add_samples.call_args.kwargs["start_timestamp"] == "2025_01_01_03_59_58_PST"
        assert mock_add_samples.call_args.kwargs["corrected_class"] == "vessel"
        assert mock_add_samples.call_args.kwargs["fallback_description"] == detection.comments
        assert mock_add_samples.call_args.kwargs["fallback_notes"] == "fp_machine"

        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 2
        assert rows[-1]["Category"] == "vessel"
        assert rows[-1]["URI"] == "https://example.com/new"

    def test_continues_when_global_prediction_is_not_resident(self, tmp_path):
        """A non-resident 60-second prediction should still process mismatched whale segments."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        manual_samples_path = tmp_path / "manual_samples.csv"

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_positives.get_machine_detections", return_value=[detection]), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_positives.add_training_3s_samples",
                 return_value=[
                     {
                         "Category": "resident",
                         "NodeName": "rpi_test",
                         "StartTimestamp": "2025_01_01_04_00_02_PST",
                         "URI": "https://example.com/new",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "91.0",
                     },
                     {
                         "Category": "water",
                         "NodeName": "rpi_test",
                         "StartTimestamp": "2025_01_01_04_00_04_PST",
                         "URI": "https://example.com/water",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "20.0",
                     },
                 ],
             ) as mock_add_samples:
            mock_get_model.return_value.predict.return_value = {
                "global_prediction_label": "water",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
            )

        assert summary["not_false_positive"] == 1
        assert summary["whale_mismatch_segments"] == 1
        assert summary["appended"] == 1
        mock_add_samples.assert_called_once()

        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["Category"] == "vessel"
        assert rows[0]["URI"] == "https://example.com/new"

    def test_skips_segments_that_already_match_corrected_whale_class(self, tmp_path):
        """Segments already predicted as the corrected whale class should not be appended."""
        feed = _make_feed()
        detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Likely transient calls from Bigg's whales.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        manual_samples_path = tmp_path / "manual_samples.csv"

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch("process_false_positives.get_machine_detections", return_value=[detection]), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_positives.add_training_3s_samples",
                 return_value=[
                     {
                         "Category": "transient",
                         "NodeName": "rpi_test",
                         "StartTimestamp": "2025_01_01_04_00_00_PST",
                         "URI": "https://example.com/correct",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "92.0",
                     },
                     {
                         "Category": "resident",
                         "NodeName": "rpi_test",
                         "StartTimestamp": "2025_01_01_04_00_02_PST",
                         "URI": "https://example.com/resident",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "91.0",
                     },
                     {
                         "Category": "humpback",
                         "NodeName": "rpi_test",
                         "StartTimestamp": "2025_01_01_04_00_05_PST",
                         "URI": "https://example.com/humpback",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "89.0",
                     },
                     {
                         "Category": "water",
                         "NodeName": "rpi_test",
                         "StartTimestamp": "2025_01_01_04_00_06_PST",
                         "URI": "https://example.com/water",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "20.0",
                     },
                 ],
             ):
            mock_get_model.return_value.predict.return_value = {
                "global_prediction_label": "resident",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
            )

        assert summary["whale_mismatch_segments"] == 2
        assert summary["appended"] == 2
        assert summary["duplicates"] == 0

        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert [row["URI"] for row in rows] == [
            "https://example.com/resident",
            "https://example.com/humpback",
        ]
        assert all(row["Category"] == "transient" for row in rows)

    def test_continues_after_processing_failure(self, tmp_path):
        """A processing error for one detection should be counted and not stop later detections."""
        feed = _make_feed()
        failed_detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        next_detection = OrcaHelloDetection(
            id="det_2",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        manual_samples_path = tmp_path / "manual_samples.csv"

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch(
                 "process_false_positives.get_machine_detections",
                 return_value=[next_detection, failed_detection],
             ), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_positives.add_training_3s_samples",
                 side_effect=[
                     RuntimeError("decode error"),
                     [
                         {
                             "Category": "resident",
                             "NodeName": "rpi_test",
                             "StartTimestamp": "2025_01_01_04_05_02_PST",
                             "URI": "https://example.com/new",
                             "Description": "desc",
                             "Notes": "manual",
                             "Confidence": "91.0",
                         },
                     ],
                 ],
             ):
            mock_get_model.return_value.predict.return_value = {
                "global_prediction_label": "resident",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
            )

        assert summary["rejected"] == 2
        assert summary["processing_failed"] == 1
        assert summary["whale_mismatch_segments"] == 1
        assert summary["appended"] == 1

        with open(manual_samples_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["Category"] == "vessel"
        assert rows[0]["URI"] == "https://example.com/new"

    def test_filters_by_actual_category_before_model_inference(self, tmp_path):
        """A category filter should skip non-matching detections before inference."""
        feed = _make_feed()
        matching_detection = OrcaHelloDetection(
            id="det_1",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Boat noise from a nearby vessel.",
        )
        non_matching_detection = OrcaHelloDetection(
            id="det_2",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            status="rejected",
            comments="Likely transient calls from Bigg's whales.",
        )
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")
        manual_samples_path = tmp_path / "manual_samples.csv"

        with patch("process_false_positives.get_model_inference") as mock_get_model, \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch(
                 "process_false_positives.get_machine_detections",
                 return_value=[matching_detection, non_matching_detection],
             ), \
             patch("process_false_positives.download_60s_audio", return_value=str(wav_path)), \
             patch(
                 "process_false_positives.add_training_3s_samples",
                 return_value=[
                     {
                         "Category": "resident",
                         "NodeName": "rpi_test",
                         "StartTimestamp": "2025_01_01_04_05_02_PST",
                         "URI": "https://example.com/new",
                         "Description": "desc",
                         "Notes": "manual",
                         "Confidence": "91.0",
                     },
                 ],
             ) as mock_add_samples:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = {
                "global_prediction_label": "resident",
                "global_confidence": 0.91,
            }
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
                actual_category_filter="VESSEL",
            )

        assert summary["rejected"] == 2
        assert summary["whale_mismatch_segments"] == 1
        assert summary["appended"] == 1
        assert mock_model.predict.call_count == 1
        assert mock_add_samples.call_count == 1
        assert mock_add_samples.call_args.kwargs["corrected_class"] == "vessel"

    def test_counts_unreviewed_and_confirmed_in_range(self, tmp_path):
        """Unreviewed and confirmed in-range detections should be counted for diagnostics."""
        feed = _make_feed()
        unreviewed = OrcaHelloDetection(
            id="det_unreviewed",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            status="unreviewed",
            comments="",
        )
        confirmed = OrcaHelloDetection(
            id="det_confirmed",
            feed=feed,
            timestamp=datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
            status="confirmed",
            comments="",
        )
        out_of_range = OrcaHelloDetection(
            id="det_old",
            feed=feed,
            timestamp=datetime(2024, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
            status="unreviewed",
            comments="",
        )
        manual_samples_path = tmp_path / "manual_samples.csv"

        with patch("process_false_positives.get_model_inference"), \
             patch("process_false_positives.get_orcasite_feeds_with_retry", return_value=[feed]), \
             patch(
                 "process_false_positives.get_machine_detections",
                 return_value=[confirmed, unreviewed, out_of_range],
             ):
            summary = process_false_positives(
                manual_samples_path=manual_samples_path,
                output_dir=tmp_path / "segments",
                start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                end_time=datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            )

        assert summary["rejected"] == 0
        assert summary["confirmed"] == 1
        assert summary["unreviewed"] == 1  # out_of_range not counted
