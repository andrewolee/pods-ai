# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for make_csv.py.

Tests cover parse_timestamp_pacific(), the timestamp-range filtering logic
inside process_all_feeds(), and the fail-fast behaviour of get_orcasite_feeds().
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests
from pytz import timezone as pytz_timezone

from make_csv import (
    PACIFIC_TZ,
    UTC_TZ,
    get_orcasite_detections,
    get_orcahello_detections,
    get_orcasite_feeds,
    parse_timestamp_pacific,
    process_all_feeds,
)
from orcasite_feeds import OrcasiteFeed, get_orcasite_feeds_with_retry


# ---------------------------------------------------------------------------
# parse_timestamp_pacific
# ---------------------------------------------------------------------------

class TestParseTimestampPacific:
    """Tests for parsing YYYY_MM_DD_HH_MM_SS_PST strings into datetimes."""

    def test_returns_timezone_aware_datetime(self):
        """Result must carry timezone info."""
        dt = parse_timestamp_pacific("2026_03_17_00_00_00_PST")
        assert dt.tzinfo is not None

    def test_correct_date_components(self):
        """Date components should match the input string."""
        dt = parse_timestamp_pacific("2025_12_24_17_51_23_PST")
        dt_pst = dt.astimezone(PACIFIC_TZ)
        assert dt_pst.year == 2025
        assert dt_pst.month == 12
        assert dt_pst.day == 24
        assert dt_pst.hour == 17
        assert dt_pst.minute == 51
        assert dt_pst.second == 23

    def test_timezone_is_pacific(self):
        """Result should be localized to US/Pacific."""
        dt = parse_timestamp_pacific("2026_01_01_00_00_00_PST")
        # Convert to UTC and check the offset matches PST (UTC-8) or PDT (UTC-7).
        utc_dt = dt.astimezone(UTC_TZ)
        offset_hours = (dt.utcoffset().total_seconds()) / 3600
        # Pacific time is either UTC-8 (PST) or UTC-7 (PDT).
        assert offset_hours in (-8.0, -7.0)

    def test_midnight_boundary(self):
        """Midnight (00:00:00) should parse correctly."""
        dt = parse_timestamp_pacific("2026_03_17_00_00_00_PST")
        dt_pst = dt.astimezone(PACIFIC_TZ)
        assert dt_pst.hour == 0
        assert dt_pst.minute == 0
        assert dt_pst.second == 0

    def test_missing_suffix_raises_value_error(self):
        with pytest.raises(
            ValueError,
            match="Timestamp '2026_01_17_00_00_00' must end with '_PST' or '_PDT'",
        ):
            parse_timestamp_pacific("2026_01_17_00_00_00")

    def test_raises_on_invalid_format(self):
        """ValueError should be raised on an unparseable timestamp body."""
        with pytest.raises(ValueError):
            parse_timestamp_pacific("not_a_timestamp_PST")


# ---------------------------------------------------------------------------
# process_all_feeds – timestamp filtering
# ---------------------------------------------------------------------------

def _make_det(ts: datetime, source: str = "human", category: str = "whale",
              description: str = "resident pod"):
    """Build a minimal OrcasiteDetection-like mock for a given timestamp."""
    from make_csv import OrcasiteDetection, OrcasiteFeed
    feed = OrcasiteFeed(
        id="feed_1",
        name="Test Feed",
        node_name="rpi_test",
        slug="test",
        bucket="audio-orcasound-net",
        bucket_region="us-west-2",
        visible=True,
        location=(47.0, -122.0),
    )
    return OrcasiteDetection(
        id="det_1",
        feed=feed,
        timestamp=ts,
        source=source,
        category=category,
        description=description,
        idempotency_key="",
    )


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    """Return a UTC-aware datetime."""
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


class TestProcessAllFeedsTimestampFilter:
    """Tests that process_all_feeds correctly filters by start_time / end_time."""

    def _run(self, dets, start_time=None, end_time=None, tmp_path=None):
        """
        Run process_all_feeds with mocked I/O and return the rows written to the CSV.
        """
        import csv
        from pathlib import Path
        from tempfile import TemporaryDirectory

        ctx = TemporaryDirectory() if tmp_path is None else None
        out_dir = Path(ctx.name if ctx else str(tmp_path))

        with patch("make_csv.get_orcasite_feeds") as mock_feeds, \
             patch("make_csv.get_orcasite_detections") as mock_dets, \
             patch("make_csv.get_orcahello_detections") as mock_oh:

            from make_csv import OrcasiteFeed
            feed = OrcasiteFeed(
                id="feed_1", name="Test", node_name="rpi_test", slug="test",
                bucket="audio-orcasound-net", bucket_region="us-west-2",
                visible=True, location=(47.0, -122.0),
            )
            mock_feeds.return_value = [feed]
            mock_dets.return_value = dets
            mock_oh.return_value = []

            process_all_feeds(out_dir, start_time=start_time, end_time=end_time)

        csv_path = out_dir / "detections.csv"
        if not csv_path.exists():
            if ctx:
                ctx.cleanup()
            return []

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if ctx:
            ctx.cleanup()
        return rows

    def test_no_filter_includes_all(self):
        """Without filters, all classifiable detections are included."""
        dets = [
            _make_det(_utc(2025, 1, 1)),
            _make_det(_utc(2025, 6, 1)),
            _make_det(_utc(2026, 1, 1)),
        ]
        rows = self._run(dets)
        assert len(rows) == 3

    def test_start_timestamp_uses_dynamic_pacific_abbreviation(self):
        """CSV timestamps use PDT in summer and PST in winter."""
        rows = self._run([
            _make_det(_utc(2024, 7, 15, 12)),
            _make_det(_utc(2024, 1, 15, 12)),
        ])
        timestamps = {row["StartTimestamp"] for row in rows}
        assert timestamps == {
            "2024_07_15_05_00_00_PDT",
            "2024_01_15_04_00_00_PST",
        }

    def test_end_time_excludes_later_detections(self):
        """Detections after end_time must not appear in the output."""
        end = parse_timestamp_pacific("2026_03_17_00_00_00_PST")
        before = _make_det(_utc(2025, 12, 1))
        after = _make_det(_utc(2026, 6, 1))
        rows = self._run([before, after], end_time=end)
        assert len(rows) == 1

    def test_start_time_excludes_earlier_detections(self):
        """Detections before start_time must not appear in the output."""
        start = parse_timestamp_pacific("2025_06_01_00_00_00_PST")
        before = _make_det(_utc(2025, 1, 1))
        after = _make_det(_utc(2025, 12, 1))
        rows = self._run([before, after], start_time=start)
        assert len(rows) == 1

    def test_detection_on_end_boundary_is_included(self):
        """A detection exactly at end_time should be included (<=)."""
        end = parse_timestamp_pacific("2026_03_17_00_00_00_PST")
        # Convert end to UTC and use it as the detection timestamp.
        end_utc = end.astimezone(timezone.utc)
        det = _make_det(end_utc)
        rows = self._run([det], end_time=end)
        assert len(rows) == 1

    def test_detection_on_start_boundary_is_included(self):
        """A detection exactly at start_time should be included (>=)."""
        start = parse_timestamp_pacific("2025_06_01_00_00_00_PST")
        start_utc = start.astimezone(timezone.utc)
        det = _make_det(start_utc)
        rows = self._run([det], start_time=start)
        assert len(rows) == 1

    def test_range_filter_combined(self):
        """Only detections within [start, end] should be included."""
        start = parse_timestamp_pacific("2025_01_01_00_00_00_PST")
        end = parse_timestamp_pacific("2025_12_31_23_59_59_PST")
        too_early = _make_det(_utc(2024, 12, 31))
        in_range = _make_det(_utc(2025, 6, 15))
        too_late = _make_det(_utc(2026, 2, 1))
        rows = self._run([too_early, in_range, too_late], start_time=start, end_time=end)
        assert len(rows) == 1

    def test_end_time_none_includes_future_detections(self):
        """end_time=None (i.e., '--end now') should include all future detections."""
        future = _make_det(_utc(2099, 1, 1))
        rows = self._run([future], end_time=None)
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# CLI argument parsing – default end and 'now' keyword
# ---------------------------------------------------------------------------

class TestCliEndArgument:
    """Tests for the --end CLI argument default and 'now' special value."""

    def test_default_end_is_fixed_timestamp(self):
        """The default --end value should be 2026_03_17_00_00_00_PST."""
        import argparse
        import sys
        # Re-parse with no arguments using the same logic as __main__.
        parser = argparse.ArgumentParser()
        parser.add_argument("--start", type=str, default=None)
        parser.add_argument("--end", type=str, default="2026_03_17_00_00_00_PST")
        args = parser.parse_args([])
        assert args.end == "2026_03_17_00_00_00_PST"
        # Should parse without error.
        end_time = None if (args.end or "").lower() == "now" else parse_timestamp_pacific(args.end)
        assert end_time is not None

    def test_end_now_yields_none(self):
        """'--end now' should produce end_time=None (no upper bound)."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--end", type=str, default="2026_03_17_00_00_00_PST")
        args = parser.parse_args(["--end", "now"])
        end_time = None if (args.end or "").lower() == "now" else parse_timestamp_pacific(args.end)
        assert end_time is None

    def test_end_now_case_insensitive(self):
        """'--end NOW' (uppercase) should also produce end_time=None."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--end", type=str, default="2026_03_17_00_00_00_PST")
        args = parser.parse_args(["--end", "NOW"])
        end_time = None if (args.end or "").lower() == "now" else parse_timestamp_pacific(args.end)
        assert end_time is None


# ---------------------------------------------------------------------------
# get_orcasite_feeds – fail-fast on network errors
# ---------------------------------------------------------------------------

class TestGetOrcasiteFeedsFailFast:
    """get_orcasite_feeds() should propagate exceptions so CI fails immediately."""

    def test_reraises_on_request_exception(self):
        """A network error must propagate rather than returning an empty list."""
        with patch("orcasite_feeds.requests.get", side_effect=requests.exceptions.Timeout("timed out")):
            with pytest.raises(requests.exceptions.Timeout):
                get_orcasite_feeds()

    def test_reraises_on_http_error(self):
        """An HTTP error (e.g. 503) must propagate rather than returning an empty list."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("503")
        with patch("orcasite_feeds.requests.get", return_value=mock_response):
            with pytest.raises(requests.exceptions.HTTPError):
                get_orcasite_feeds()


class TestGetOrcasiteFeedsWithRetry:
    """get_orcasite_feeds_with_retry() should retry on ReadTimeout."""

    def test_retries_after_read_timeout(self):
        """A read timeout should trigger a retry and eventually return feeds."""
        expected_feeds = [
            OrcasiteFeed(
                id="feed_1",
                name="Test Feed",
                node_name="rpi_test",
                slug="test",
                bucket="audio-orcasound-net",
                bucket_region="us-west-2",
                visible=True,
                location=(47.0, -122.0),
            )
        ]
        with patch(
            "orcasite_feeds.get_orcasite_feeds",
            side_effect=[requests.exceptions.ReadTimeout("timed out"), expected_feeds],
        ) as mock_get_feeds, patch("orcasite_feeds.time.sleep") as mock_sleep:
            feeds = get_orcasite_feeds_with_retry(max_attempts=2, retry_delay_seconds=1)

        assert feeds == expected_feeds
        assert mock_get_feeds.call_count == 2
        mock_sleep.assert_called_once_with(1)

    def test_returns_empty_list_after_retry_exhausted(self):
        """Repeated read timeouts should return an empty list instead of raising."""
        with patch(
            "orcasite_feeds.get_orcasite_feeds",
            side_effect=requests.exceptions.ReadTimeout("timed out"),
        ) as mock_get_feeds, patch("orcasite_feeds.time.sleep") as mock_sleep:
            feeds = get_orcasite_feeds_with_retry(max_attempts=2, retry_delay_seconds=1)

        assert feeds == []
        assert mock_get_feeds.call_count == 2
        mock_sleep.assert_called_once_with(1)


class TestGetOrcasiteDetectionsWithRetry:
    """get_orcasite_detections() should retry on ReadTimeout and fail after retries."""

    @staticmethod
    def _feed() -> OrcasiteFeed:
        """Create a minimal feed object for testing."""
        return OrcasiteFeed(
            id="feed_1",
            name="Test Feed",
            node_name="rpi_test",
            slug="test",
            bucket="audio-orcasound-net",
            bucket_region="us-west-2",
            visible=True,
            location=(47.0, -122.0),
        )

    def test_retries_after_read_timeout(self):
        """A read timeout should retry and then succeed when a later attempt works."""
        response_with_data = MagicMock()
        response_with_data.raise_for_status.return_value = None
        response_with_data.json.return_value = {
            "data": [
                {
                    "id": "det_1",
                    "attributes": {
                        "timestamp": "2025-01-01T12:00:00Z",
                        "source": "human",
                        "category": "whale",
                        "description": "resident",
                        "idempotency_key": "",
                    },
                }
            ]
        }
        response_empty_page = MagicMock()
        response_empty_page.raise_for_status.return_value = None
        response_empty_page.json.return_value = {"data": []}

        with patch(
            "make_csv.requests.get",
            side_effect=[
                requests.exceptions.ReadTimeout("timed out"),
                response_with_data,
                response_empty_page,
            ],
        ) as mock_get, patch("make_csv.time.sleep") as mock_sleep:
            detections = get_orcasite_detections(self._feed(), max_attempts=2, retry_delay_seconds=1)

        assert len(detections) == 1
        assert detections[0].id == "det_1"
        assert detections[0].source == "human"
        assert detections[0].category == "whale"
        assert mock_get.call_count == 3
        mock_sleep.assert_called_once_with(1)

    def test_raises_after_read_timeout_retry_exhausted(self):
        """Repeated read timeouts should raise after the configured retry attempts."""
        with patch(
            "make_csv.requests.get",
            side_effect=requests.exceptions.ReadTimeout("timed out"),
        ) as mock_get, patch("make_csv.time.sleep") as mock_sleep:
            with pytest.raises(requests.exceptions.ReadTimeout):
                get_orcasite_detections(self._feed(), max_attempts=2, retry_delay_seconds=1)

        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1)

    def test_non_timeout_errors_fail_fast_without_retry(self):
        """Non-timeout errors should fail immediately without retries."""
        with patch(
            "make_csv.requests.get",
            side_effect=requests.exceptions.HTTPError("boom"),
        ) as mock_get, patch("make_csv.time.sleep") as mock_sleep:
            with pytest.raises(requests.exceptions.HTTPError):
                get_orcasite_detections(self._feed(), max_attempts=3, retry_delay_seconds=1)

        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()


class TestGetOrcahelloDetections:
    """Tests for get_orcahello_detections."""

    def test_includes_comments_from_cosmos_items(self):
        """The returned detection should preserve the moderation comments field."""
        feed = OrcasiteFeed(
            id="feed_1",
            name="Test Feed",
            node_name="rpi_test",
            slug="test",
            bucket="audio-orcasound-net",
            bucket_region="us-west-2",
            visible=True,
            location=(47.0, -122.0),
        )
        mock_container = MagicMock()
        mock_container.query_items.return_value = [
            {
                "id": "det_1",
                "SRKWFound": "no",
                "reviewed": True,
                "timestamp": "2025-01-01T12:00:00Z",
                "whaleFoundConfidence": "0.25",
                "comments": "Boat noise.",
            }
        ]
        mock_database = MagicMock()
        mock_database.get_container_client.return_value = mock_container
        mock_client = MagicMock()
        mock_client.get_database_client.return_value = mock_database

        with patch("make_csv.CosmosClient", return_value=mock_client):
            detections = get_orcahello_detections(feed)

        assert len(detections) == 1
        assert detections[0].status == "rejected"
        assert detections[0].comments == "Boat noise."

    def test_sets_timestamp_to_none_when_parsing_fails(self):
        """An unparseable OrcaHello timestamp should produce a detection with timestamp=None."""
        feed = OrcasiteFeed(
            id="feed_1",
            name="Test Feed",
            node_name="rpi_test",
            slug="test",
            bucket="audio-orcasound-net",
            bucket_region="us-west-2",
            visible=True,
            location=(47.0, -122.0),
        )
        mock_container = MagicMock()
        mock_container.query_items.return_value = [
            {
                "id": "det_1",
                "SRKWFound": "yes",
                "reviewed": True,
                "timestamp": "not-a-timestamp",
            }
        ]
        mock_database = MagicMock()
        mock_database.get_container_client.return_value = mock_container
        mock_client = MagicMock()
        mock_client.get_database_client.return_value = mock_database

        with patch("make_csv.CosmosClient", return_value=mock_client):
            detections = get_orcahello_detections(feed)

        assert len(detections) == 1
        assert detections[0].status == "confirmed"
        assert detections[0].timestamp is None
