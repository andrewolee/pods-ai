#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Investigate prevalence of "hum" and "60Hz" tags among rejected OrcaHello detections.

For issue #482: determine whether enough false positives are tagged "hum" (or the
older "60Hz" tag) to justify adding "hum" as a new training/testing category.

Queries Cosmos DB for every known feed's rejected detections, tallies tag frequency,
and reports the "hum"/"60Hz" share specifically alongside a few example detections
for manual spot-checking.
"""

import argparse
from collections import Counter
from datetime import datetime
from typing import Optional

from audio_utils import get_orcahello_detections, parse_timestamp_pst
from orcasite_feeds import get_orcasite_feeds_with_retry

HUM_TAGS = {"hum", "60hz"}


def investigate_hum_prevalence(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    feed_filter: Optional[str] = None,
) -> dict:
    """Tally tag frequency among rejected OrcaHello detections across all feeds.

    Returns a summary dict with total rejected-detection count, a Counter of
    normalized tag frequency, a count of detections with no tags at all, and
    example detections for any tag in HUM_TAGS.
    """
    feeds = get_orcasite_feeds_with_retry()
    if feed_filter:
        feeds = [feed for feed in feeds if feed.node_name == feed_filter]

    tag_counter: Counter = Counter()
    no_tag_count = 0
    rejected_count = 0
    hum_examples = []

    for feed in feeds:
        print(f"Querying {feed.node_name}...")
        detections = get_orcahello_detections(feed, start_time, end_time)
        for detection in detections:
            if detection.status.lower() != "rejected":
                continue
            rejected_count += 1

            raw_tags = [t.strip().lower() for t in (detection.tags or "").split(";") if t.strip()]
            if not raw_tags:
                no_tag_count += 1
            for tag in raw_tags:
                tag_counter[tag] += 1

            if any(tag in HUM_TAGS for tag in raw_tags):
                hum_examples.append({
                    "feed": feed.node_name,
                    "timestamp": detection.timestamp,
                    "tags": detection.tags,
                    "comments": detection.comments,
                })

    return {
        "rejected_count": rejected_count,
        "no_tag_count": no_tag_count,
        "tag_counter": tag_counter,
        "hum_examples": hum_examples,
    }


def print_report(summary: dict) -> None:
    """Print a tag-frequency report, highlighting the hum/60Hz share."""
    rejected_count = summary["rejected_count"]
    tag_counter: Counter = summary["tag_counter"]

    print()
    print("=" * 72)
    print("Tag Frequency Among Rejected OrcaHello Detections")
    print("=" * 72)
    print(f"{'Tag':<20}{'Count':>10}{'% of rejected':>18}")
    print("-" * 72)
    for tag, count in tag_counter.most_common():
        pct = 100 * count / rejected_count if rejected_count else 0.0
        print(f"{tag:<20}{count:>10}{pct:>17.2f}%")
    print("-" * 72)
    print(f"Total rejected detections: {rejected_count}")
    print(f"Rejected detections with no tags at all: {summary['no_tag_count']}")

    # Count each detection once, even if it carries both "hum" and "60hz" tags
    # (summing tag_counter directly would double-count those detections).
    hum_count = len(summary["hum_examples"])
    hum_pct = 100 * hum_count / rejected_count if rejected_count else 0.0
    print()
    print(f"'hum'/'60Hz' tagged: {hum_count} / {rejected_count} ({hum_pct:.2f}%)")
    print("=" * 72)

    if summary["hum_examples"]:
        print("\nExample hum/60Hz-tagged detections:")
        for ex in summary["hum_examples"][:10]:
            print(f"  [{ex['feed']}] {ex['timestamp']} tags={ex['tags']!r} comments={ex['comments']!r}")


def main() -> int:
    """Run the hum-category investigation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Investigate prevalence of 'hum'/'60Hz'-tagged false positives in the "
            "OrcaHello detection history, to inform issue #482."
        )
    )
    parser.add_argument(
        "--feed",
        type=str,
        default=None,
        help="Restrict to a single feed by node_name (e.g., rpi_sunset_bay).",
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
        default=None,
        metavar="YYYY_MM_DD_HH_MM_SS_PST",
        help="Include only detections with timestamp <= this value.",
    )
    args = parser.parse_args()

    start_time = parse_timestamp_pst(args.start) if args.start else None
    end_time = parse_timestamp_pst(args.end) if args.end else None

    summary = investigate_hum_prevalence(start_time, end_time, args.feed)
    print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
