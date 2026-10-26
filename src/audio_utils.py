# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Common utilities for audio downloading and S3 operations.

This module contains shared functions used by both extract_training_samples.py
and download_wavs.py to avoid code duplication.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import math
import http.client
import os
import shutil
import time
import urllib.error

import boto3
from azure.cosmos import CosmosClient
from botocore import UNSIGNED
from botocore.config import Config
import ffmpeg
import m3u8
try:
    from orcasite_feeds import OrcasiteFeed
except ImportError:  # pragma: no cover
    from src.orcasite_feeds import OrcasiteFeed
from pytz import timezone
import requests


# Simple in-memory cache for S3 folder listings keyed by "bucket::prefix"
_FOLDERS_CACHE = {}

# Number of times to retry a download on transient connection errors.
MAX_DOWNLOAD_RETRIES = 3

# Seconds to wait between download retry attempts.
DOWNLOAD_RETRY_DELAY_SECONDS = 2
PACIFIC_TZ = timezone('US/Pacific')
UTC_TZ = timezone('UTC')
COSMOS_URL = os.environ.get("COSMOS_URL", "").strip() or "https://aifororcasmetadatastore.documents.azure.com:443/"
COSMOS_KEY = os.environ.get("COSMOS_KEY", "").strip()
COSMOS_DB = os.environ.get("COSMOS_DB", "predictions")
COSMOS_CONTAINER = os.environ.get("COSMOS_CONTAINER", "metadata")


@dataclass
class OrcaHelloDetection:
    id: str
    feed: OrcasiteFeed
    timestamp: Optional[datetime]
    status: str
    confidence: Optional[float] = None
    tags: str = ""
    comments: str = ""


# Terms in a detection description that indicate the label cannot be determined with confidence.
SKIP_TERMS = {'?', 'not sure', 'unsure', 'possibl', 'sounds like', 'sounded like', 'ould be'}


def get_all_folders(bucket: str, prefix: str) -> List[str]:
    """
    Get all folder names from an S3 bucket with the given prefix.
    
    Parameters:
        bucket (str): Name of the S3 bucket.
        prefix (str): Prefix to filter objects.
    
    Returns:
        List[str]: List of folder names (without the prefix).
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}

    all_keys = []
    for page in paginator.paginate(**kwargs):
        try:
            common_prefixes = page["CommonPrefixes"]
            prefixes = [
                prefix["Prefix"].split("/")[-2] for prefix in common_prefixes
            ]
            all_keys.extend(prefixes)
        except KeyError:
            break

    return all_keys


def get_cached_folders(bucket: str, prefix: str) -> List[str]:
    """
    Return cached folder list for (bucket, prefix). If absent, call get_all_folders()
    to populate the cache, then return the cached value.
    """
    key = f"{bucket}::{prefix}"
    if key not in _FOLDERS_CACHE:
        _FOLDERS_CACHE[key] = get_all_folders(bucket, prefix)
    return _FOLDERS_CACHE[key]


def get_folders_between_timestamp(bucket_list: List[str], start_time: int, end_time: int) -> List[int]:
    """
    Filter bucket list to only include folders between start_time and end_time.
    
    Parameters:
        bucket_list (List[str]): List of folder names (as strings).
        start_time (int): Start unix timestamp.
        end_time (int): End unix timestamp.
    
    Returns:
        List[int]: Filtered list of folder names as integers.
    """
    bucket_list = [int(bucket) for bucket in bucket_list]
    start_index = 0
    end_index = len(bucket_list) - 1

    while start_index < len(bucket_list) and bucket_list[start_index] < start_time:
        start_index += 1

    while end_index >= 0 and bucket_list[end_index] > end_time:
        end_index -= 1

    # Include the folder before start_time to ensure we have data.
    return bucket_list[max(0, start_index - 1) : end_index + 1]


def get_difference_between_times_in_seconds(unix_time1: int, unix_time2: int) -> float:
    """
    Calculate the difference between two unix timestamps in seconds.
    
    Parameters:
        unix_time1 (int): First unix timestamp.
        unix_time2 (int): Second unix timestamp.
    
    Returns:
        float: Difference in seconds.
    """
    dt1 = datetime.fromtimestamp(int(unix_time1))
    dt2 = datetime.fromtimestamp(int(unix_time2))
    return (dt1 - dt2).total_seconds()


def download_from_url(dl_url: str, dl_dir: str):
    """
    Download a file from URL to a directory, with retry on transient connection errors.

    Retries up to MAX_DOWNLOAD_RETRIES times on ConnectionError or ChunkedEncodingError
    before re-raising the exception.

    Parameters:
        dl_url (str): URL to download from.
        dl_dir (str): Directory to save the file.
    """
    file_name = os.path.basename(dl_url)
    dl_path = os.path.join(dl_dir, file_name)

    if os.path.isfile(dl_path):
        return

    last_exception: Exception = RuntimeError("No attempts made")
    for attempt in range(MAX_DOWNLOAD_RETRIES + 1):
        try:
            response = requests.get(dl_url, timeout=30)
            response.raise_for_status()
            with open(dl_path, 'wb') as f:
                f.write(response.content)
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            last_exception = e
            if attempt < MAX_DOWNLOAD_RETRIES:
                print(f"  Retry {attempt + 1} of {MAX_DOWNLOAD_RETRIES} for {dl_url}: {e}")
                time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)
    raise last_exception


def load_m3u8_with_retry(stream_url: str) -> m3u8.M3U8:
    """
    Load an m3u8 playlist from a URL, retrying on transient network errors.

    Retries up to MAX_DOWNLOAD_RETRIES times on IncompleteRead, URLError,
    or ConnectionError before re-raising the exception.

    Parameters:
        stream_url (str): URL of the m3u8 playlist to load.

    Returns:
        m3u8.M3U8: Parsed m3u8 object.
    """
    last_exception: Exception = RuntimeError("No attempts made")
    for attempt in range(MAX_DOWNLOAD_RETRIES + 1):
        try:
            return m3u8.load(stream_url)
        except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError) as e:
            last_exception = e
            if attempt < MAX_DOWNLOAD_RETRIES:
                print(f"  Retry {attempt + 1} of {MAX_DOWNLOAD_RETRIES} for {stream_url}: {e}")
                time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)
    raise last_exception


def format_timestamp_pst(dt: datetime) -> str:
    """
    Format a datetime object as PST timestamp string in the format YYYY_MM_DD_HH_MM_SS_PST.
    """
    dt_pst = dt.astimezone(PACIFIC_TZ)
    return dt_pst.strftime("%Y_%m_%d_%H_%M_%S_PST")


def parse_timestamp_pst(ts_str: str) -> datetime:
    """
    Parse a PST timestamp string in the format YYYY_MM_DD_HH_MM_SS_PST into a timezone-aware datetime.
    """
    if not ts_str.endswith("_PST"):
        raise ValueError(f"Timestamp '{ts_str}' must end with '_PST'")
    body = ts_str[:-4]
    dt_naive = datetime.strptime(body, "%Y_%m_%d_%H_%M_%S")
    return PACIFIC_TZ.localize(dt_naive)


def get_node_name_for_feed(feed: OrcasiteFeed) -> str:
    """Retrieve the node name associated with an OrcasiteFeed."""
    return feed.node_name


def get_machine_detections(
    feed: OrcasiteFeed,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[OrcaHelloDetection]:
    """
    Retrieve machine detections associated with the given feed's node name.
    """
    if not COSMOS_KEY:
        raise ValueError("COSMOS_KEY environment variable must be set and non-empty to fetch machine detections")

    node_name = get_node_name_for_feed(feed)

    cosmos_client = CosmosClient(COSMOS_URL, credential=COSMOS_KEY)
    cosmos_database = cosmos_client.get_database_client(COSMOS_DB)
    container = cosmos_database.get_container_client(COSMOS_CONTAINER)

    query = """
        SELECT * FROM c
        WHERE c.location.id = @node_name
    """
    params = [{"name": "@node_name", "value": node_name}]
    if start_time:
        query += " AND c.timestamp >= @start_time"
        params.append({"name": "@start_time", "value": start_time.isoformat()})
    if end_time:
        query += " AND c.timestamp <= @end_time"
        params.append({"name": "@end_time", "value": end_time.isoformat()})
    query += " ORDER BY c.timestamp DESC"

    items = container.query_items(
        query=query,
        parameters=params,
        enable_cross_partition_query=True
    )

    results = []
    for item in items:
        found = item.get("SRKWFound", "").lower()
        reviewed = item.get("reviewed", False)
        if reviewed and found == "yes":
            status = "confirmed"
        elif reviewed and found == "no":
            status = "rejected"
        else:
            status = "unreviewed"

        ts_raw = item.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            ts = None

        confidence = item.get("whaleFoundConfidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = None

        results.append(
            OrcaHelloDetection(
                id=item.get("id"),
                feed=feed,
                timestamp=ts,
                status=status,
                confidence=confidence,
                tags=(item.get("tags") or "").strip(),
                comments=(item.get("comments") or "").strip()
            )
        )

    return results


def _get_aligned_end_time(timestamp_pst_str: str) -> datetime:
    raw_end = parse_timestamp_pst(timestamp_pst_str)
    snapped_sec = ((raw_end.second + 9) // 10) * 10
    if snapped_sec == 60:
        # roll over to next minute.
        raw_end = raw_end + timedelta(minutes=1)
        snapped_sec = 0
    return raw_end.replace(second=snapped_sec, microsecond=0)


def download_60s_audio(node_name: str, min_end_timestamp_pst_str: str, tmp_dir: str) -> Optional[str]:
    """
    Download 60 seconds of audio ending at the next 10-second boundary after min_end_timestamp_str.
    Note that min_end_timestamp_str is the Orcasite time, which is off by ~2 seconds from real time.
    """
    # Compute the aligned end time and derive the 60s clip start UTC to reuse
    # the download_audio_from_start_utc implementation which handles the
    # HLS folder/segment logic. This reduces duplication and keeps behavior
    # consistent between callers that request clips by end-time vs start-time.
    end_time = _get_aligned_end_time(min_end_timestamp_pst_str)
    audio_offset = 2 # startup delay
    start_time = end_time - timedelta(seconds=60) - timedelta(seconds=audio_offset)
    start_time_utc = start_time.astimezone(UTC_TZ)

    return download_audio_from_start_utc(node_name, start_time_utc, tmp_dir, 60)


MIN_SEGMENT_DURATION = 0.001
FLOAT_TOLERANCE = 1e-9

def _build_clip_id(start_time_utc: datetime) -> str:
    return start_time_utc.astimezone(PACIFIC_TZ).strftime("%Y_%m_%d_%H_%M_%S_PST")


def format_utc_iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC_TZ).strftime("%Y-%m-%dT%H:%M:%SZ")


def download_audio_from_start_utc(
    node_name: str,
    start_time_utc: datetime,
    tmp_dir: str,
    duration_seconds: float,
) -> Optional[str]:
    """Download a clip beginning at start_time_utc and lasting for duration_seconds."""
    end_time_utc = start_time_utc + timedelta(seconds=duration_seconds)
    start_unix_time = int(start_time_utc.timestamp())
    end_unix_time = int(end_time_utc.timestamp())

    hydrophone_stream_url = f"https://s3-us-west-2.amazonaws.com/audio-orcasound-net/{node_name}"
    bucket_folder = hydrophone_stream_url.split("https://s3-us-west-2.amazonaws.com/")[1]
    tokens = bucket_folder.split("/")
    s3_bucket = tokens[0]
    folder_name = tokens[1]
    prefix = folder_name + "/hls/"

    try:
        all_hydrophone_folders = get_cached_folders(s3_bucket, prefix=prefix)
        print(f"  Found {len(all_hydrophone_folders)} folders in total for {node_name}")
        valid_folders = get_folders_between_timestamp(all_hydrophone_folders, start_unix_time, end_unix_time)
        print(f"  Found {len(valid_folders)} folders in date range")
        if not valid_folders:
            print(f"  Warning: No folders found for timestamp {start_time_utc}")
            return None
        current_folder = int(valid_folders[0])
    except Exception as e:
        print(f"  ERROR: Failed to query S3 bucket: {e}")
        return None

    stream_url = f"{hydrophone_stream_url}/hls/{current_folder}/live.m3u8"
    try:
        stream_obj = load_m3u8_with_retry(stream_url)
    except Exception as e:
        print(f"  ERROR: Failed to load m3u8 file: {e}")
        return None

    num_total_segments = len(stream_obj.segments)
    if num_total_segments == 0:
        print("  ERROR: No segments found in m3u8 file")
        return None

    target_duration_exact = sum(item.duration for item in stream_obj.segments) / num_total_segments
    target_duration = max(target_duration_exact, MIN_SEGMENT_DURATION)

    time_since_folder_start_for_start = get_difference_between_times_in_seconds(start_unix_time, current_folder)
    time_since_folder_start_for_end = get_difference_between_times_in_seconds(end_unix_time, current_folder)

    segment_start_index = max(
        0,
        math.floor((time_since_folder_start_for_start + FLOAT_TOLERANCE) / target_duration),
    )
    segment_end_index = min(
        num_total_segments,
        math.ceil((time_since_folder_start_for_end - FLOAT_TOLERANCE) / target_duration),
    )
    if segment_end_index <= segment_start_index:
        segment_end_index = min(num_total_segments, segment_start_index + 1)

    print(
        f"Segment: folder={current_folder}, indices=[{segment_start_index}:{segment_end_index}), "
        f"start={format_utc_iso_z(start_time_utc)}, duration={duration_seconds:.1f}s"
    )

    try:
        file_names = []
        for i in range(segment_start_index, segment_end_index):
            audio_segment = stream_obj.segments[i]
            base_path = audio_segment.base_uri
            file_name = audio_segment.uri
            audio_url = base_path + file_name
            download_from_url(audio_url, tmp_dir)
            file_names.append(file_name)

        if not file_names:
            print("  ERROR: No segments were successfully downloaded")
            return None

        clip_id = _build_clip_id(start_time_utc)
        clipname = f"temp_{duration_seconds:g}s_{node_name}_{clip_id}"
        if len(file_names) > 1:
            hls_file = str(Path(tmp_dir) / f"{clipname}.ts")
            with open(hls_file, "wb") as wfd:
                for f in file_names:
                    with open(Path(tmp_dir) / f, "rb") as fd:
                        shutil.copyfileobj(fd, wfd)
        else:
            hls_file = str(Path(tmp_dir) / file_names[0])

        wav_file_path = str(Path(tmp_dir) / f"{clipname}.wav")
        ss_offset = time_since_folder_start_for_start - (segment_start_index * target_duration)
        if ss_offset < 0:
            ss_offset = 0.0

        stream = ffmpeg.input(hls_file, ss=ss_offset)
        stream = ffmpeg.output(
            stream,
            wav_file_path,
            t=duration_seconds,
            acodec="pcm_s16le",
            ar=44100,
            ac=1,
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        print(f"  Downloaded {duration_seconds:g}s audio: {wav_file_path}")
        return wav_file_path
    except Exception as e:
        print(f"  Warning: Unable to retrieve audio clip: {e}")
        return None
