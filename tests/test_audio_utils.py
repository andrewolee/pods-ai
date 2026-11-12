# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

import os
import time
from datetime import datetime, timezone

import pytest

from audio_utils import get_difference_between_times_in_seconds


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="requires time.tzset")
def test_difference_between_times_across_dst_transition(monkeypatch):
    original_timezone = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()

    try:
        earlier = datetime(2024, 3, 10, 9, 30, tzinfo=timezone.utc).timestamp()
        later = datetime(2024, 3, 10, 10, 30, tzinfo=timezone.utc).timestamp()

        assert get_difference_between_times_in_seconds(later, earlier) == 3600
    finally:
        if original_timezone is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original_timezone)
        time.tzset()
