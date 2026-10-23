# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for testing sample download logic in download_wavs.py."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, Mock, patch
import os

import pytest
import download_wavs

from download_wavs import (
    CSVRow,
    add_seconds_to_timestamp_pst,
    download_testing_sample,
    is_external_humpback_training_wav,
    process_csv,
    process_testing_csv,
    run_download_wavs,
    validate_aligned_entries,
    validate_node_slug_in_uri,
    validate_no_overlaps,
    validate_uri_timestamps,
)


class TestDownloadTestingSample:
    """Tests for download_testing_sample routing behavior."""

    def test_tp_human_only_downloads_60s_audio_to_testing_directory(self):
        """tp_human_only rows should download from the start timestamp and save output."""
        row = CSVRow(
            category="resident",
            node_name="rpi_andrews_bay",
            timestamp_pst="2025_01_01_00_00_00_PST",
            uri="https://example.org/sample",
            description="sample",
            notes="tp_human_only",
            confidence="",
        )

        with TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "testing-wav"

            def _fake_download_audio_from_start_utc(
                node_name: str,
                start_time_utc: datetime,
                tmp_dir: str,
                duration_seconds: int,
            ):
                """Create and return a temporary fake 60-second WAV path."""
                wav_path = Path(tmp_dir) / "temp_60s.wav"
                wav_path.write_bytes(b"fake wav content")
                return str(wav_path)

            with patch(
                "audio_utils.download_audio_from_start_utc",
                side_effect=_fake_download_audio_from_start_utc,
            ) as mock_download:
                download_testing_sample(row, output_root)

            mock_download.assert_called_once_with(
                node_name="rpi_andrews_bay",
                start_time_utc=datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc),
                tmp_dir=ANY,
                duration_seconds=60,
            )
            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            assert expected.exists()

    def test_tp_machine_only_downloads_60s_clip_from_start_timestamp(self):
        """tp_machine_only rows should download from the start timestamp."""
        row = CSVRow(
            category="humpback",
            node_name="rpi_orcasound_lab",
            timestamp_pst="2025_01_01_00_00_03_PST",
            uri="https://example.org/sample",
            description="sample",
            notes="tp_machine_only",
            confidence="",
        )

        with TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "testing-wav"
            def _fake_download_audio_from_start_utc(
                node_name: str,
                start_time_utc: datetime,
                tmp_dir: str,
                duration_seconds: int,
            ):
                """Create and return a temporary fake 60-second WAV path."""
                wav_path = Path(tmp_dir) / "temp_60s.wav"
                wav_path.write_bytes(b"fake wav content")
                return str(wav_path)

            with patch(
                "audio_utils.download_audio_from_start_utc",
                side_effect=_fake_download_audio_from_start_utc,
            ) as mock_download:
                download_testing_sample(row, output_root)

            mock_download.assert_called_once_with(
                node_name="rpi_orcasound_lab",
                start_time_utc=datetime(2025, 1, 1, 8, 0, 3, tzinfo=timezone.utc),
                tmp_dir=ANY,
                duration_seconds=60,
            )
            expected = output_root / "humpback" / "rpi-orcasound-lab_2025_01_01_00_00_03_PST.wav"
            assert expected.exists()


class TestTimestampHelpers:
    """Tests for timestamp conversion helpers."""

    def test_add_seconds_to_timestamp_pst_adds_30_seconds(self):
        """add_seconds_to_timestamp_pst should add requested seconds in PST format."""
        assert add_seconds_to_timestamp_pst("2025_01_01_00_00_03_PST", 30) == "2025_01_01_00_00_33_PST"

    def test_add_seconds_to_timestamp_pst_adds_60_seconds(self):
        """add_seconds_to_timestamp_pst should support adding 60 seconds."""
        assert add_seconds_to_timestamp_pst("2025_01_01_00_00_03_PST", 60) == "2025_01_01_00_01_03_PST"


class TestOverlapValidation:
    def test_validate_no_overlaps_allows_non_overlapping_rows(self):
        training_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_01_00_00_PST", "", "", "", ""),
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_01_00_03_PST", "", "", "", ""),
        ]
        testing_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_59_00_PST", "", "", "tp_human_only", ""),
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_01_01_06_PST", "", "", "tp_human_only", ""),
        ]
        validate_no_overlaps(training_rows, testing_rows)

    def test_validate_no_overlaps_rejects_training_overlap(self):
        training_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_00_00_PST", "", "", "", ""),
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_00_02_PST", "", "", "", ""),
        ]
        with pytest.raises(ValueError, match="training overlap"):
            validate_no_overlaps(training_rows, [])

    def test_validate_no_overlaps_rejects_cross_file_overlap(self):
        training_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_00_00_PST", "", "", "", ""),
        ]
        testing_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2024_12_31_23_59_58_PST", "", "", "tp_machine_only", ""),
        ]
        with pytest.raises(ValueError, match="cross-file overlap"):
            validate_no_overlaps(training_rows, testing_rows)


class TestAlignedEntryValidation:
    @staticmethod
    def _clear_validation_caches():
        download_wavs._DETECTIONS_WINDOW_CACHE.clear()
        download_wavs._CORRECTED_TIMESTAMP_CACHE.clear()

    @staticmethod
    def _mock_detection_response(payload, total_pages: int | None = None):
        response = Mock()
        response.text = "[]"
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        response.headers = {}
        if payload:
            response.text = "[{\"id\":\"1\"}]"
        if total_pages is not None:
            response.headers["totalAmountPages"] = str(total_pages)
        return response

    def test_validate_aligned_entries_handles_paginated_items_payload(self):
        self._clear_validation_caches()
        testing_rows = [
            CSVRow(
                "human",
                "rpi_sunset_bay",
                "2025_12_01_00_00_00_PST",
                "https://live.orcasound.net/bouts/new/sunset-bay?time=2025-12-01T08%3A00%3A00.000Z",
                "Radio",
                "fp_machine_only",
                "100",
            ),
        ]
        first_page_payload = {
            "items": [
                {
                    "timestamp": "2025-12-01T07:00:00Z",
                    "comments": "not a match",
                    "found": "No",
                    "reviewed": False,
                }
                for _ in range(50)
            ],
        }
        second_page_payload = [
            {
                "timestamp": "2025-12-01T08:00:02Z",
                "comments": "Radio",
                "found": "No",
                "reviewed": True,
            },
        ]

        with patch(
            "download_wavs.requests.get",
            side_effect=[
                self._mock_detection_response(first_page_payload, total_pages=2),
                self._mock_detection_response(second_page_payload, total_pages=2),
            ],
        ) as mock_get, \
                patch("audio_utils.get_cached_folders", side_effect=AssertionError("should not query S3 for current epoch")):
            validate_aligned_entries(testing_rows)

        assert mock_get.call_count == 2

    def test_validate_aligned_entries_allows_current_epoch_false_positive_without_s3_lookup(self):
        self._clear_validation_caches()
        testing_rows = [
            CSVRow(
                "human",
                "rpi_sunset_bay",
                "2025_12_01_00_00_00_PST",
                "https://live.orcasound.net/bouts/new/sunset-bay?time=2025-12-01T08%3A00%3A00.000Z",
                "Radio",
                "fp_machine_only",
                "100",
            ),
        ]
        detections = [
            {
                "timestamp": "2025-12-01T08:00:02Z",
                "comments": "Radio",
                "found": "No",
                "reviewed": True,
            },
        ]

        with patch("download_wavs.requests.get", return_value=self._mock_detection_response(detections)) as mock_get, \
                patch("audio_utils.get_cached_folders", side_effect=AssertionError("should not query S3 for current epoch")):
            validate_aligned_entries(testing_rows)

        mock_get.assert_called_once()

    def test_validate_aligned_entries_rejects_current_epoch_misalignment_within_detection_window(self):
        self._clear_validation_caches()
        testing_rows = [
            CSVRow(
                "human",
                "rpi_sunset_bay",
                "2025_12_01_00_00_48_PST",
                "https://live.orcasound.net/bouts/new/sunset-bay?time=2025-12-01T08%3A00%3A48.000Z",
                "Radio",
                "fp_machine_only",
                "100",
            ),
        ]
        detections = [
            {
                "timestamp": "2025-12-01T08:00:02Z",
                "comments": "Radio",
                "found": "No",
                "reviewed": True,
            },
            {
                "timestamp": "2025-11-30T08:00:02Z",
                "comments": "Radio",
                "found": "No",
                "reviewed": True,
            },
        ]

        with patch("download_wavs.requests.get", return_value=self._mock_detection_response(detections)), \
                patch("audio_utils.get_cached_folders", side_effect=AssertionError("should not query S3 for current epoch")):
            with pytest.raises(
                ValueError,
                match=r"old testing_row: human,rpi_sunset_bay,2025_12_01_00_00_48_PST,https://live\.orcasound\.net/bouts/new/sunset-bay\?time=2025-12-01T08%3A00%3A48\.000Z,Radio,fp_machine_only,100\n"
                r"  new testing_row: human,rpi_sunset_bay,2025_12_01_00_00_00_PST,https://live\.orcasound\.net/bouts/new/sunset-bay\?time=2025-12-01T08%3A00%3A00\.000Z,Radio,fp_machine_only,100",
            ):
                validate_aligned_entries(testing_rows)

    def test_validate_aligned_entries_rejects_old_epoch_misalignment(self):
        self._clear_validation_caches()
        testing_rows = [
            CSVRow(
                "human",
                "rpi_sunset_bay",
                "2025_01_01_00_10_48_PST",
                "https://live.orcasound.net/bouts/new/sunset-bay?time=2025-01-01T08%3A10%3A48.000Z",
                "Radio",
                "fp_machine_only",
                "100",
            ),
        ]
        detections = [
            {
                "timestamp": "2025-01-01T08:11:05Z",
                "comments": "Radio",
                "found": "No",
                "reviewed": True,
            },
        ]
        folder_time = int(datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp())

        with patch("download_wavs.requests.get", return_value=self._mock_detection_response(detections)), \
                patch("audio_utils.get_cached_folders", return_value=[str(folder_time)]):
            with pytest.raises(
                ValueError,
                match=r"old testing_row: human,rpi_sunset_bay,2025_01_01_00_10_48_PST,https://live\.orcasound\.net/bouts/new/sunset-bay\?time=2025-01-01T08%3A10%3A48\.000Z,Radio,fp_machine_only,100\n"
                r"  new testing_row: human,rpi_sunset_bay,2025_01_01_00_10_00_PST,https://live\.orcasound\.net/bouts/new/sunset-bay\?time=2025-01-01T08%3A10%3A00\.000Z,Radio,fp_machine_only,100",
            ):
                validate_aligned_entries(testing_rows)

    def test_validate_aligned_entries_matches_reported_old_epoch_false_positive(self):
        self._clear_validation_caches()
        testing_rows = [
            CSVRow(
                "human",
                "rpi_sunset_bay",
                "2024_07_19_12_51_09_PST",
                "https://live.orcasound.net/bouts/new/sunset-bay?time=2024-07-19T19%3A51%3A09.000Z",
                "Human voices causing false positives despite significant noise from something contacting ladder and/or hydrophone.",
                "fp_machine_only",
                "54.4156",
            ),
        ]
        detections = [
            {
                "timestamp": "2024-07-19T21:07:09.102135Z",
                "comments": "Human voices causing false positives despite significant noise from something contacting ladder and/or hydrophone.",
                "found": "No",
                "reviewed": True,
            },
        ]
        folder_time = int(datetime(2024, 7, 19, 7, 0, 48, tzinfo=timezone.utc).timestamp())

        with patch("download_wavs.requests.get", return_value=self._mock_detection_response(detections)), \
                patch("audio_utils.get_cached_folders", return_value=[str(folder_time)]):
            with pytest.raises(
                ValueError,
                match=r"old testing_row: human,rpi_sunset_bay,2024_07_19_12_51_09_PST,https://live\.orcasound\.net/bouts/new/sunset-bay\?time=2024-07-19T19%3A51%3A09\.000Z,Human voices causing false positives despite significant noise from something contacting ladder and/or hydrophone\.,fp_machine_only,54\.4156\n"
                r"  new testing_row: human,rpi_sunset_bay,2024_07_19_12_50_08_PST,https://live\.orcasound\.net/bouts/new/sunset-bay\?time=2024-07-19T19%3A50%3A08\.000Z,Human voices causing false positives despite significant noise from something contacting ladder and/or hydrophone\.,fp_machine_only,54\.4156",
            ):
                validate_aligned_entries(testing_rows)


class TestCacheAndCleanup:
    def test_process_csv_copies_from_cache_without_downloading(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "training_3s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes,confidence\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,note,100\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-wav"
            cache_root = tmp_path / "cache-wav"
            cached_file = cache_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_bytes(b"cached")

            with patch("audio_utils.get_cached_folders", side_effect=AssertionError("should not download")):
                process_csv(csv_path, output_root, cache_root=cache_root)

            downloaded_file = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            assert downloaded_file.exists()
            assert downloaded_file.read_bytes() == b"cached"

    def test_process_csv_deletes_stale_wavs(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "training_3s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes,confidence\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,note,100\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-wav"
            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            stale = output_root / "resident" / "old.wav"
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_bytes(b"keep")
            stale.write_bytes(b"remove")

            process_csv(csv_path, output_root)

            assert expected.exists()
            assert not stale.exists()

    def test_process_csv_keeps_humpback_signal_wavs_and_rejects_noise(self):
        """signals-humpback segments stay in training; unlabeled noise is deleted."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "training_3s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes,confidence\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,note,100\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-wav"
            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            humpback_signal = (
                output_root / "humpback" / "signals-humpback_song_0000s.wav"
            )
            noise = output_root / "humpback" / "random_noise.wav"
            expected.parent.mkdir(parents=True, exist_ok=True)
            humpback_signal.parent.mkdir(parents=True, exist_ok=True)
            expected.write_bytes(b"keep")
            humpback_signal.write_bytes(b"humpback")
            noise.write_bytes(b"noise")

            process_csv(csv_path, output_root)

            assert expected.exists()
            assert humpback_signal.exists()
            assert not noise.exists()

    def test_is_external_humpback_training_wav_matches_signal_segments(self):
        assert is_external_humpback_training_wav(
            Path("humpback") / "signals-humpback_song_0000s.wav"
        )
        assert not is_external_humpback_training_wav(
            Path("humpback") / "rpi-orcasound-lab_2025_01_01_00_00_00_PST.wav"
        )
        assert not is_external_humpback_training_wav(
            Path("water") / "signals-humpback_song_0000s.wav"
        )

    def test_process_testing_csv_copies_from_cache_and_deletes_stale(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "testing_60s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes,confidence\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,tp_human_only,100\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-testing-wav"
            stale = output_root / "resident" / "old.wav"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_bytes(b"remove")

            cache_root = tmp_path / "cache-testing-wav"
            cached_file = cache_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_bytes(b"cached")

            with patch("audio_utils.download_60s_audio", side_effect=AssertionError("should not download")):
                process_testing_csv(csv_path, output_root, cache_root=cache_root)

            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            assert expected.exists()
            assert expected.read_bytes() == b"cached"
            assert not stale.exists()


class TestNodeSlugValidation:
    def test_validate_node_slug_uses_default_mapping_when_feed_lookup_fails(self):
        rows = [
            CSVRow(
                "resident",
                "rpi_andrews_bay",
                "2025_01_01_01_01_06_PST",
                "https://live.orcasound.net/bouts/new/andrews-bay?time=2025-01-01T09%3A01%3A06.000Z",
                "desc",
                "tp_human_only",
                "100",
            ),
        ]

        with patch("download_wavs.get_orcasite_feeds", side_effect=Exception("Read timed out")):
            validate_node_slug_in_uri(rows)

    def test_validate_node_slug_still_rejects_mismatch_with_default_mapping(self):
        rows = [
            CSVRow(
                "resident",
                "rpi_andrews_bay",
                "2025_01_01_01_01_06_PST",
                "https://live.orcasound.net/bouts/new/sunset-bay?time=2025-01-01T09%3A01%3A06.000Z",
                "desc",
                "tp_human_only",
                "100",
            ),
        ]

        with patch("download_wavs.get_orcasite_feeds", side_effect=Exception("Read timed out")), \
                pytest.raises(ValueError, match="Node slug not found in URI"):
            validate_node_slug_in_uri(rows)

    def test_validate_node_slug_uses_dclde_station_slug_when_feed_lookup_fails(self):
        rows = [
            CSVRow(
                "abiotic",
                "dclde_orcasound_lab_os_9_27_2017_08_03_00_0002",
                "2017_09_27_08_03_00_PST",
                "https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/orcasound/audio/orcasound_lab/OS_9_27_2017_08_03_00__0002.wav",
                "desc",
                "dclde_orcasound_full_recording",
                "100",
            ),
        ]

        with patch("download_wavs.get_orcasite_feeds", side_effect=Exception("Read timed out")):
            validate_node_slug_in_uri(rows)

    def test_validate_node_slug_raises_lookup_error_for_unknown_non_rpi_node(self):
        rows = [
            CSVRow(
                "resident",
                "unknown_station_name",
                "2025_01_01_01_01_06_PST",
                "https://live.orcasound.net/bouts/new/unknown-station-name?time=2025-01-01T09%3A01%3A06.000Z",
                "desc",
                "tp_human_only",
                "100",
            ),
        ]

        with patch("download_wavs.get_orcasite_feeds", side_effect=Exception("Read timed out")), \
                pytest.raises(ValueError, match="Unable to look up slug for node unknown_station_name"):
            validate_node_slug_in_uri(rows)


class TestUriTimestampValidation:
    def test_validate_uri_timestamps_does_not_require_feed_lookup(self):
        rows = [
            CSVRow(
                "resident",
                "rpi_andrews_bay",
                "2025_01_01_01_01_06_PST",
                "https://live.orcasound.net/bouts/new/andrews-bay?time=2025-01-01T09%3A01%3A06.000Z",
                "desc",
                "tp_human_only",
                "100",
            ),
        ]

        with patch("download_wavs.get_orcasite_feeds", side_effect=Exception("Read timed out")):
            validate_uri_timestamps(rows)

    def test_validate_uri_timestamps_matches_summer_dst_rows(self):
        rows = [
            CSVRow(
                "resident",
                "rpi_sunset_bay",
                "2024_07_19_12_50_08_PST",
                "https://live.orcasound.net/bouts/new/sunset-bay?time=2024-07-19T19%3A50%3A08.000Z",
                "desc",
                "tp_human_only",
                "100",
            ),
        ]

        validate_uri_timestamps(rows)


class TestValidateOnly:
    def test_run_download_wavs_validate_only_skips_download_processing(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_dir = tmp_path / "output" / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            (csv_dir / "training_3s_samples.csv").write_text(
                "category,node_name,timestamp_pst,uri,description,notes,confidence\n"
                "resident,rpi_andrews_bay,2025_01_01_01_00_00_PST,https://live.orcasound.net/bouts/new/andrews-bay?time=2025-01-01T09%3A00%3A00.000Z,desc,note,100\n",
                encoding="utf-8",
            )
            (csv_dir / "testing_60s_samples.csv").write_text(
                "category,node_name,timestamp_pst,uri,description,notes,confidence\n"
                "resident,rpi_andrews_bay,2025_01_01_01_01_06_PST,https://live.orcasound.net/bouts/new/andrews-bay?time=2025-01-01T09%3A01%3A06.000Z,desc,tp_human_only,100\n",
                encoding="utf-8",
            )

            original_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                with patch("download_wavs.process_csv") as mock_process_csv, \
                        patch("download_wavs.process_testing_csv") as mock_process_testing_csv, \
                        patch("download_wavs.validate_aligned_entries") as mock_validate_aligned_entries:
                    run_download_wavs(validate_only=True)
                mock_process_csv.assert_not_called()
                mock_process_testing_csv.assert_not_called()
                mock_validate_aligned_entries.assert_called_once()
            finally:
                os.chdir(original_cwd)

    def test_parse_args_custom_training_and_testing_paths_are_forwarded(self, monkeypatch, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        training_csv = custom_dir / "training.csv"
        training_csv.write_text(
            "category,node_name,timestamp_pst,uri,description,notes,confidence\n"
            "resident,rpi_andrews_bay,2025_01_01_01_00_00_PST,uri,desc,note,100\n",
            encoding="utf-8",
        )
        testing_csv = custom_dir / "testing.csv"
        testing_csv.write_text(
            "category,node_name,timestamp_pst,uri,description,notes,confidence\n"
            "resident,rpi_andrews_bay,2025_01_01_01_01_06_PST,uri,desc,tp_human_only,100\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "sys.argv",
            [
                "download_wavs.py",
                "--training-csv-path",
                str(training_csv),
                "--testing-csv-path",
                str(testing_csv),
            ],
        )

        args = download_wavs.parse_args()

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            with patch("download_wavs.process_csv") as mock_process_csv, \
                    patch("download_wavs.process_testing_csv") as mock_process_testing_csv, \
                    patch("download_wavs.validate_no_overlaps"), \
                    patch("download_wavs.validate_aligned_entries"), \
                    patch("download_wavs.validate_uri_timestamps"), \
                    patch("download_wavs.validate_node_slug_in_uri"):
                run_download_wavs(
                    training_csv_path=args.training_csv_path,
                    testing_csv_path=args.testing_csv_path,
                )
        finally:
            os.chdir(original_cwd)

        assert args.training_csv_path == training_csv
        assert args.testing_csv_path == testing_csv
        assert mock_process_csv.call_args.args[:2] == (training_csv, Path("output/wav"))
        assert mock_process_csv.call_args.kwargs == {"cache_root": None}
        assert mock_process_testing_csv.call_args.args[:2] == (
            testing_csv,
            Path("output/testing-wav"),
        )
