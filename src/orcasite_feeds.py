# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Lightweight helpers for fetching Orcasite feed metadata.

This module is intentionally kept free of azure-cosmos and other heavy
dependencies so that any script needing only the feeds API (e.g.
add_samples.py) can import it.
"""

from dataclasses import dataclass
import time
from typing import Optional, List

import requests

FEED_FETCH_MAX_ATTEMPTS = 3
FEED_FETCH_RETRY_DELAY_SECONDS = 2


@dataclass
class OrcasiteFeed:
    id: str                     # e.g., "feed_02u8r4EPgmlYQmh6gzlGIL"
    name: str                   # "Beach Camp at Sunset Bay"
    node_name: str              # "rpi_sunset_bay"
    slug: str                   # "sunset-bay"
    bucket: str                 # "audio-orcasound-net"
    bucket_region: str          # "us-west-2"
    visible: bool               # True/False
    location: tuple[float, float]  # (lat, lng)
    image_url: Optional[str] = None
    cloudfront_url: Optional[str] = None


def get_orcasite_feeds() -> List[OrcasiteFeed]:
    """
    Fetch feeds from the Orcasite API and parse them into a list of OrcasiteFeed objects.

    Each feed includes metadata such as id, name, node_name, slug, storage
    bucket info, visibility, geographic location (latitude, longitude), and
    optional image and CloudFront URLs.

    Returns:
        List[OrcasiteFeed]: A list of parsed feed objects.

    Raises:
        Exception: Re-raises any exception encountered during the HTTP request
            or response parsing so the caller can detect and report the failure.
    """
    url = "https://live.orcasound.net/api/json/feeds"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        feeds = []
        for item in data.get("data", []):
            attrs = item.get("attributes", {})

            lat = attrs.get("lat_lng", {}).get("lat")
            lng = attrs.get("lat_lng", {}).get("lng")

            feed = OrcasiteFeed(
                id=item.get("id"),
                name=attrs.get("name"),
                node_name=attrs.get("node_name"),
                slug=attrs.get("slug"),
                bucket=attrs.get("bucket"),
                bucket_region=attrs.get("bucket_region"),
                visible=attrs.get("visible", True),
                location=(lat, lng),
                image_url=attrs.get("image_url"),
                cloudfront_url=attrs.get("cloudfront_url"),
            )

            feeds.append(feed)

        return feeds

    except Exception as e:
        print("Error fetching Orcasite feeds:", e)
        raise


def get_orcasite_feeds_with_retry(
    max_attempts: int = FEED_FETCH_MAX_ATTEMPTS,
    retry_delay_seconds: int = FEED_FETCH_RETRY_DELAY_SECONDS,
) -> List[OrcasiteFeed]:
    """Fetch Orcasite feeds, retrying when the API read times out.

    Args:
        max_attempts: Maximum number of fetch attempts before giving up.
        retry_delay_seconds: Seconds to wait between retry attempts.

    Returns:
        List[OrcasiteFeed]: Feed metadata records, or an empty list when retries are exhausted.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return get_orcasite_feeds()
        except requests.exceptions.ReadTimeout as exc:
            if attempt == max_attempts:
                print(f"Error fetching Orcasite feeds after {max_attempts} attempts: {exc}")
                break
            print(
                "Read timeout fetching Orcasite feeds "
                f"(attempt {attempt}/{max_attempts}); retrying in {retry_delay_seconds} seconds."
            )
            time.sleep(retry_delay_seconds)
    return []
