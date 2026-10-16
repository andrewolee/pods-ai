#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Live inference orchestrator for multiclass PODS-AI models."""

import argparse
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import requests

from model_inference import get_model_inference
from pytz import timezone as pytz_tz
from run_inference import build_tags_list


AZURE_STORAGE_ACCOUNT_NAME = "livemlaudiospecstorage"
AZURE_STORAGE_AUDIO_CONTAINER_NAME = "audiowavs"
AZURE_STORAGE_SPECTROGRAM_CONTAINER_NAME = "spectrogramspng"

COSMOSDB_ACCOUNT_NAME = "aifororcasmetadatastore"
COSMOSDB_DATABASE_NAME = "predictions"
COSMOSDB_CONTAINER_NAME = "metadata"

ORCASOUND_S3_BUCKET = "audio-orcasound-net"

NEGATIVE_LABELS = {"abiotic", "other", "water", "vessel", "jingle", "human", "bird"}

PODSAI_MODEL_ID = "davethaler/whale-call-detector"
from model_revisions import LATEST_PODSAI_MODEL_REVISION

def get_hydro_attributes_from_feed() -> dict:
    hydrophone_location_data = {}
    """Get the API data from the Orcasound feed."""
    try:
        response = requests.get("https://live.orcasound.net/api/json/feeds")
        response.raise_for_status()
        data = response.json().get("data")
        for location in data:
            attributes = location.get("attributes", {})
            node_name = attributes.get("node_name")
            name = attributes.get("name")
            coordinates = attributes.get("location_point", {}).get("coordinates", [])
            if len(coordinates) == 2:
                longitude, latitude = coordinates
            hydrophone_location_data[node_name] = {
                "id": node_name,
                "name": name,
                "longitude": float(longitude),
                "latitude": float(latitude)
            }
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error occurred: {err}")
        # Print the raw text/JSON from the server which often contains the real error logs
        print("Server Response context:")
        print(response.text)
    return hydrophone_location_data

source_guid_to_location = get_hydro_attributes_from_feed()

def assemble_blob_uri(container_name: str, item_name: str) -> str:
    """Assemble a blob URI from account/container/item."""
    return "https://{acct}.blob.core.windows.net/{cont}/{item}".format(
        acct=AZURE_STORAGE_ACCOUNT_NAME, cont=container_name, item=item_name
    )


def prediction_to_label(prediction: Any, id2label: Optional[dict[int, str]]) -> str:
    """Return a string label for a local/global prediction value."""
    if isinstance(prediction, str):
        return prediction
    if isinstance(prediction, int) and id2label is not None:
        return id2label.get(prediction, str(prediction))
    return str(prediction)


def is_positive_label(label: str, negative_labels: Optional[set[str]] = None) -> bool:
    """Return True when a predicted class label should count as a whale detection."""
    effective_negative_labels = negative_labels if negative_labels is not None else NEGATIVE_LABELS
    return label not in effective_negative_labels


def build_prediction_list(
    result: dict[str, Any],
    id2label: Optional[dict[int, str]] = None,
    negative_labels: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Build CosmosDB prediction rows from PODS-AI per-segment output."""
    prediction_list: list[dict[str, Any]] = []

    local_predictions = result.get("local_predictions", [])
    local_confidences = result.get("local_confidences", [])
    hop_duration = float(result.get("hop_duration", 2.0))
    segment_duration = float(result.get("segment_duration", 3.0))

    for idx, (local_prediction, local_confidence) in enumerate(
        zip(local_predictions, local_confidences)
    ):
        label = prediction_to_label(local_prediction, id2label)
        if is_positive_label(label, negative_labels=negative_labels):
            prediction_list.append(
                {
                    "id": idx,
                    "label": label,
                    "startTime": idx * hop_duration,
                    "duration": segment_duration,
                    "confidence": float(local_confidence),
                }
            )

    return prediction_list


def build_cosmosdb_metadata(
    audio_uri: str,
    image_uri: str,
    result: dict[str, Any],
    timestamp_in_iso: str,
    source_guid: str,
    model_id: str,
    id2label: Optional[dict[int, str]] = None,
    negative_labels: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Build a CosmosDB metadata document for PODS-AI multiclass output."""
    prediction_list = build_prediction_list(
        result,
        id2label=id2label,
        negative_labels=negative_labels,
    )

    location = source_guid_to_location.get(
        source_guid,
        {
            "id": source_guid,
            "name": source_guid,
            "longitude": None,
            "latitude": None,
        },
    )

    global_label = result.get("global_prediction_label", "")
    local_labels = [
        prediction_to_label(p, id2label)
        for p in result.get("local_predictions", [])
    ]

    tags = build_tags_list(result, id2label=id2label, negative_labels=negative_labels)

    return {
        "id": str(uuid.uuid4()),
        "modelId": model_id,
        "audioUri": audio_uri,
        "imageUri": image_uri,
        "reviewed": False,
        "timestamp": timestamp_in_iso,
        "whaleFoundConfidence": float(result.get("global_confidence", 0.0)) * 100.0,
        "globalPredictionLabel": global_label,
        "location": location,
        "source_guid": source_guid,
        "predictions": prediction_list,
        "tags": ";".join(tags),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orch_config",
        type=str,
        required=False,
        help="Path to orchestrator config YAML (default: /config/config.yml).",
    )
    parser.add_argument(
        "--max_live_iterations",
        type=int,
        default=None,
        help="Maximum number of LiveHLS poll cycles (ignored for DateRangeHLS).",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=None,
        help="Maximum number of segments to process (for DateRangeHLS testing).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Log level (default: DEBUG).",
    )
    args, _ = parser.parse_known_args()

    if args.orch_config:
        print(
            f"Using orchestrator config from command line argument: {args.orch_config}"
        )
    else:
        args.orch_config = "/config/config.yml"
        print(f"Using orchestrator config from ConfigMap: {args.orch_config}")

    return args


def setup_logger(connection_string: Optional[str], log_level: str = "DEBUG") -> logging.Logger:
    """Configure and return orchestrator logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))

    if connection_string is not None:
        try:
            from opencensus.ext.azure.log_exporter import AzureEventHandler, AzureLogHandler

            logger.addHandler(AzureLogHandler(connection_string=connection_string))
            logger.addHandler(AzureEventHandler(connection_string=connection_string))
        except Exception as e:
            logger.warning(f"Could not initialize Azure log handlers: {e}")

    return logger


def load_model(orch_config: dict[str, Any], logger: logging.Logger) -> Any:
    """Load PODS-AI model using the model_inference factory."""
    model_path = orch_config.get("model_hf_repo_id", PODSAI_MODEL_ID)
    model_revision = orch_config.get("model_hf_repo_revision", LATEST_PODSAI_MODEL_REVISION)
    threshold = float(orch_config.get("threshold", 0.5))
    min_num_positive_calls_threshold = int(
        orch_config.get("min_num_positive_calls_threshold", 2)
    )
    device = orch_config.get("device")

    kwargs: dict[str, Any] = {
        "threshold": threshold,
        "min_num_positive_calls_threshold": min_num_positive_calls_threshold,
        "model_revision": model_revision,
    }
    if device:
        kwargs["device"] = device

    logger.debug(f"Loading PODS-AI model from {model_path}")
    return get_model_inference(
        model_type="podsai",
        model_path=model_path,
        **kwargs,
    )


def setup_azure_clients(orch_config: dict[str, Any]) -> tuple[Any, Any]:
    """Return (blob_service_client, cosmos_client), or (None, None)."""
    if not orch_config["upload_to_azure"]:
        return None, None

    from azure.cosmos import CosmosClient
    from azure.storage.blob import BlobServiceClient

    blob_service_client = BlobServiceClient.from_connection_string(
        os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    cosmos_client = CosmosClient(
        f"https://{COSMOSDB_ACCOUNT_NAME}.documents.azure.com:443/",
        os.getenv("AZURE_COSMOSDB_PRIMARY_KEY"),
    )
    return blob_service_client, cosmos_client


def build_orcasound_client(orch_config: dict[str, Any]) -> Any:
    """Return an OrcasoundHLSClient from orch_config."""
    external_src = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "orcahello"
        / "InferenceSystem"
        / "src"
    )
    if external_src.exists() and str(external_src) not in sys.path:
        sys.path.insert(0, str(external_src))

    from orcasound_hls import OrcasoundHLSClient

    return OrcasoundHLSClient(
        bucket=ORCASOUND_S3_BUCKET,
        hydrophone_id=orch_config["hls_hydrophone_id"],
    )


def upload_detection_to_azure(
    clip_path: str,
    spectrogram_path: str,
    result: dict[str, Any],
    start_timestamp: str,
    hls_hydrophone_id: str,
    model_id: str,
    blob_service_client: Any,
    cosmos_client: Any,
    logger: logging.Logger,
    id2label: Optional[dict[int, str]] = None,
    negative_labels: Optional[set[str]] = None,
) -> tuple[str, str, str]:
    """Upload detection assets and metadata for a positive multiclass detection."""

    def _upload_or_skip_existing(blob_client: Any, file_path: str, blob_name: str) -> None:
        try:
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data)
        except Exception as exc:
            if getattr(exc, "error_code", None) == "BlobAlreadyExists":
                logger.info(f"Blob already exists, skipping upload: {blob_name}")
                return
            raise

    audio_clip_name = os.path.basename(clip_path)
    audio_blob_client = blob_service_client.get_blob_client(
        container=AZURE_STORAGE_AUDIO_CONTAINER_NAME,
        blob=audio_clip_name,
    )
    _upload_or_skip_existing(audio_blob_client, clip_path, audio_clip_name)
    audio_uri = assemble_blob_uri(AZURE_STORAGE_AUDIO_CONTAINER_NAME, audio_clip_name)

    spectrogram_name = os.path.basename(spectrogram_path)
    spectrogram_blob_client = blob_service_client.get_blob_client(
        container=AZURE_STORAGE_SPECTROGRAM_CONTAINER_NAME,
        blob=spectrogram_name,
    )
    _upload_or_skip_existing(
        spectrogram_blob_client, spectrogram_path, spectrogram_name
    )
    spectrogram_uri = assemble_blob_uri(
        AZURE_STORAGE_SPECTROGRAM_CONTAINER_NAME,
        spectrogram_name,
    )

    metadata = build_cosmosdb_metadata(
        audio_uri,
        spectrogram_uri,
        result,
        start_timestamp,
        hls_hydrophone_id,
        model_id,
        id2label=id2label,
        negative_labels=negative_labels,
    )
    database = cosmos_client.get_database_client(COSMOSDB_DATABASE_NAME)
    container = database.get_container_client(COSMOSDB_CONTAINER_NAME)
    container.create_item(body=metadata)
    logger.info(
        f"Uploaded detection to Azure: audio={audio_clip_name}, "
        f"spectrogram={spectrogram_name}, cosmos_id={metadata['id']}, "
        f"timestamp={start_timestamp}, tags={metadata.get('tags', [])}"
    )
    return audio_clip_name, spectrogram_name, metadata["id"]


def count_positive_segments(
    result: dict[str, Any],
    id2label: Optional[dict[int, str]] = None,
    negative_labels: Optional[set[str]] = None,
) -> int:
    """Count local predictions that map to positive (non-background) labels."""
    local_predictions = result.get("local_predictions", [])
    count = 0
    for local_prediction in local_predictions:
        label = prediction_to_label(local_prediction, id2label)
        if is_positive_label(label, negative_labels=negative_labels):
            count += 1
    return count


def _process_segment(
    segment: Any,
    model: Any,
    orch_config: dict[str, Any],
    blob_service_client: Any,
    cosmos_client: Any,
    logger: logging.Logger,
    model_id: str,
    local_dir: str,
) -> None:
    """Process a single HLS segment: download, inference, optional upload."""
    hls_hydrophone_id = orch_config["hls_hydrophone_id"]

    logger.info(
        f"Segment: folder={segment.folder_epoch}, "
        f"indices=[{segment.start_index}:{segment.end_index}), "
        f"start={segment.start_iso}, duration={segment.duration_s:.1f}s"
    )

    try:
        clip_path = segment.download_as_wav(local_dir)
    except Exception as e:
        logger.warning(f"Failed to download segment: {e}")
        return

    start_timestamp = segment.start_iso
    result = model.predict(
        clip_path,
        segment_duration=int(orch_config.get("segment_duration", 3)),
        hop_duration=int(orch_config.get("hop_duration", 2)),
    )

    id2label = getattr(model, "id2label", None)
    global_prediction_label = prediction_to_label(
        result.get("global_prediction_label", ""),
        id2label,
    )
    global_confidence = float(result.get("global_confidence", 0.0))
    positive_segments = count_positive_segments(result, id2label=id2label)
    detected = is_positive_label(global_prediction_label)

    logger.info(
        f"Inference: prediction={global_prediction_label}, "
        f"confidence={global_confidence:.3f}, "
        f"positive_segments={positive_segments}/{len(result.get('local_predictions', []))}",
        extra={"custom_dimensions": {"Hydrophone ID": hls_hydrophone_id}},
    )

    if detected or not orch_config["delete_local_wavs"]:
        import spectrogram_visualizer

        spectrogram_path = spectrogram_visualizer.write_spectrogram(clip_path)
    else:
        spectrogram_path = None

    if detected:
        logger.info(
            f"Whale call detected ({global_prediction_label}, confidence={global_confidence:.3f})",
            extra={"custom_dimensions": {"Hydrophone ID": hls_hydrophone_id}},
        )
        if orch_config["upload_to_azure"] and spectrogram_path is not None:
            upload_detection_to_azure(
                clip_path,
                spectrogram_path,
                result,
                start_timestamp,
                hls_hydrophone_id,
                model_id,
                blob_service_client,
                cosmos_client,
                logger,
                id2label=id2label,
            )

    if orch_config["delete_local_wavs"]:
        os.remove(clip_path)
        if spectrogram_path is not None:
            os.remove(spectrogram_path)


def run_loop(
    orcasound_client: Any,
    model: Any,
    orch_config: dict[str, Any],
    blob_service_client: Any,
    cosmos_client: Any,
    logger: logging.Logger,
    model_id: str,
    max_live_iterations: Optional[int] = None,
    max_segments: Optional[int] = None,
) -> None:
    """Main inference loop using Orcasound HLS segments and PODS-AI model."""
    local_dir = "wav_dir"
    os.makedirs(local_dir, exist_ok=True)

    hls_stream_type = orch_config["hls_stream_type"]
    segment_size = float(orch_config.get("inference_segment_size", 60.0))
    live_delay_buffer = float(orch_config.get("hls_live_delay_buffer", 60.0))

    if hls_stream_type == "DateRangeHLS":
        hls_start_time_pst = orch_config["hls_start_time_pst"]
        hls_end_time_pst = orch_config["hls_end_time_pst"]

        start_dt = datetime.strptime(hls_start_time_pst, "%Y-%m-%d %H:%M")
        start_unix = int(pytz_tz("US/Pacific").localize(start_dt).timestamp())

        end_dt = datetime.strptime(hls_end_time_pst, "%Y-%m-%d %H:%M")
        end_unix = int(pytz_tz("US/Pacific").localize(end_dt).timestamp())

        logger.debug(
            f"DateRange: start_unix={start_unix}, end_unix={end_unix}, "
            f"start_pst={hls_start_time_pst}, end_pst={hls_end_time_pst}"
        )
        logger.info(
            f"Fetching DateRange segments: start_unix={start_unix}, "
            f"end_unix={end_unix}, segment_size={segment_size}"
        )

        segments = orcasound_client.get_segments(
            start_unix=start_unix,
            end_unix=end_unix,
            segment_size=segment_size,
        )
        if max_segments is not None:
            segments = segments[:max_segments]

        logger.info(f"Got {len(segments)} segments from date range")

        for segment in segments:
            _process_segment(
                segment,
                model,
                orch_config,
                blob_service_client,
                cosmos_client,
                logger,
                model_id,
                local_dir,
            )

    elif hls_stream_type == "LiveHLS":

        def _next_aligned_time(now: float, interval: float) -> float:
            return _align(now, interval) + interval

        def _align(ts: float, interval: float) -> float:
            return (ts // interval) * interval

        live_iteration_count = 0
        while True:
            now = _align(datetime.now(timezone.utc).timestamp(), segment_size)
            end_unix = now - live_delay_buffer
            start_unix = end_unix - segment_size

            logger.info(
                f"--- [iter {live_iteration_count}] LiveHLS poll: fetching segments in "
                f"[{start_unix:.0f}, {end_unix:.0f}] "
                f"(now={now:.0f}, delay={live_delay_buffer}s)"
            )

            segments = orcasound_client.get_segments(
                start_unix=start_unix,
                end_unix=end_unix,
                segment_size=segment_size,
            )

            logger.info(
                f"[iter {live_iteration_count}] LiveHLS poll: got {len(segments)} segments"
            )

            for segment in segments:
                _process_segment(
                    segment,
                    model,
                    orch_config,
                    blob_service_client,
                    cosmos_client,
                    logger,
                    model_id,
                    local_dir,
                )

            live_iteration_count += 1
            if max_live_iterations is not None and live_iteration_count >= max_live_iterations:
                break

            sleep_time = _next_aligned_time(now, segment_size) - time.time()
            logger.debug(
                f"Sleeping for {sleep_time:.1f}s until "
                f"{_next_aligned_time(now, segment_size):.0f}"
            )
            if sleep_time > 0:
                time.sleep(sleep_time)

    else:
        raise ValueError("hls_stream_type should be one of LiveHLS or DateRangeHLS")


def main() -> int:
    """Run the live inference orchestrator entry point."""
    try:
        from dotenv import load_dotenv
        import yaml
    except Exception as e:
        print(f"Missing dependencies for orchestrator runtime: {e}", file=sys.stderr)
        return 1

    load_dotenv()

    args = parse_args()
    with open(args.orch_config, encoding="utf-8") as f:
        orch_config = yaml.safe_load(f)

    app_insights_connection_string = os.getenv(
        "INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING"
    )
    logger = setup_logger(app_insights_connection_string, log_level=args.log_level)

    model_id = orch_config.get("model_id", orch_config.get("model_hf_repo_id", PODSAI_MODEL_ID))
    model = load_model(orch_config, logger)

    blob_service_client, cosmos_client = setup_azure_clients(orch_config)
    orcasound_client = build_orcasound_client(orch_config)

    run_loop(
        orcasound_client,
        model,
        orch_config,
        blob_service_client,
        cosmos_client,
        logger,
        model_id,
        max_live_iterations=args.max_live_iterations,
        max_segments=args.max_segments,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
