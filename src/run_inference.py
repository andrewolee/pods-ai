#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Run inference on a wav file and output per-class probabilities.

Usage:
    python run_inference.py sample.wav
    python run_inference.py sample.wav --model podsai --model-path /path/to/podsai-model
    python run_inference.py sample.wav --model fastai --model-path ../model
    python run_inference.py --node-name rpi_sunset_bay --end-timestamp-str 2025_01_15_12_30_00_PST
"""

import argparse
import math
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

import ffmpeg

from audio_utils import (
    format_utc_iso_z,
    download_audio_from_start_utc,
    download_from_url,
    get_cached_folders,
    get_difference_between_times_in_seconds,
    get_folders_between_timestamp,
    load_m3u8_with_retry,
    PACIFIC_TZ,
)
UTC_TZ = timezone.utc
from model_inference import get_model_inference

PODSAI_MODEL_ID = "davethaler/whale-call-detector"
# renovate: datasource=git-refs depName=https://huggingface.co/davethaler/whale-call-detector versioning=git.
from model_revisions import LATEST_PODSAI_MODEL_REVISION, PODSAI_WAV2VEC2_MODEL_REVISION

# Preserve the existing exported constant name for compatibility.
PODSAI_MODEL_REVISION = LATEST_PODSAI_MODEL_REVISION
PROPOSED_DESCRIPTION_EXTRA_CLASSES = {"vessel", "human", "jingle", "bird"}
NEGATIVE_LABELS = {"abiotic", "other", "water", "vessel", "jingle", "human", "bird"}


def parse_pst_end_timestamp(timestamp_str: str) -> datetime:
    """Parse PST end timestamp format YYYY_MM_DD_HH_MM_SS_PST."""
    dt = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S_PST")
    return PACIFIC_TZ.localize(dt)


def parse_utc_start_timestamp(timestamp_str: str) -> datetime:
    """Parse UTC start timestamp format YYYY-MM-DDTHH:MM:SSZ."""
    return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC_TZ)


def _format_pacific_timestamp(dt: datetime) -> str:
    return dt.astimezone(PACIFIC_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


# Get global positive tags plus the most common non-whale context tag.
def build_tags_list(
    result: dict[str, Any],
    id2label: Optional[dict[int, str]] = None,
    negative_labels: Optional[set[str]] = None,
) -> list[str]:
    """Build deduplicated event tags from global and local predictions.

    Multi-label model results contribute every supported global whale label,
    followed by the most common non-whale local context label. Older single-label
    results continue to use their primary label.

    Args:
        result: Inference result returned by ``run_inference`` or a model backend.
        id2label: Optional mapping for integer local prediction IDs.
        negative_labels: Optional labels treated as non-whale context labels.

    Returns:
        Ordered, duplicate-free event tags.
    """
    from LiveInferenceOrchestrator import is_positive_label

    effective_negative_labels = (
        negative_labels if negative_labels is not None else NEGATIVE_LABELS
    )
    local_prediction_labels = []

    # Collect tags in output order.
    tags = []

    global_prediction_label = result.get("global_prediction_label", "")
    global_prediction_labels = result.get("global_prediction_labels")
    if global_prediction_labels is None:
        global_prediction_labels = (
            [global_prediction_label]
            if global_prediction_label
            and is_positive_label(
                global_prediction_label,
                negative_labels=effective_negative_labels,
            )
            else []
        )
    for label in global_prediction_labels:
        if label and is_positive_label(label, negative_labels=effective_negative_labels):
            if label not in tags:
                tags.append(label)

    # Add labels from local predictions regardless if positive or negative.
    local_predictions = result.get("local_predictions", [])
    for local_prediction in local_predictions:
        label = prediction_to_label(local_prediction, id2label)
        local_prediction_labels.append(label)

    # If no local tags, return only global tag.
    if not local_prediction_labels:
        return tags

    # Add the most common non-whale local context label.
    most_common_counts = Counter(local_prediction_labels).most_common()
    for label, _count in most_common_counts:
        if not is_positive_label(label, negative_labels=effective_negative_labels):
            if label not in tags:
                tags.append(label)
            break
    # Map 'resident' to 'srkw' for AIForOrcas compatibility.
    tags = ["srkw" if tag == "resident" else tag for tag in tags]

    return tags


def prediction_to_label(prediction: Any, id2label: Optional[dict[int, str]]) -> str:
    """Return a string label for a local/global prediction value."""
    if isinstance(prediction, str):
        return prediction
    if isinstance(prediction, int) and id2label is not None:
        return id2label.get(prediction, str(prediction))
    return str(prediction)


def calculate_positive_segments(
    local_predictions: list[Any],
    local_confidences: list[Any],
    hop_duration: float,
    segment_duration: float,
    id2label: Optional[dict[int, str]] = None,
    negative_labels: Optional[set[str]] = None,
    threshold: Optional[float] = None,
    start_time_utc: Optional[datetime] = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Count positive segments and optionally include UTC/Pacific timestamps."""
    effective_negative_labels = (
        negative_labels if negative_labels is not None else NEGATIVE_LABELS
    )
    confidence_threshold = float(threshold) if threshold is not None else 0.0
    positive_segments: list[dict[str, Any]] = []

    for idx, (local_prediction, local_confidence) in enumerate(
        zip(local_predictions, local_confidences)
    ):
        label = prediction_to_label(local_prediction, id2label)
        confidence = float(local_confidence)
        if label in effective_negative_labels or confidence < confidence_threshold:
            continue

        start_seconds = idx * float(hop_duration)
        segment_info: dict[str, Any] = {
            "index": idx,
            "label": label,
            "confidence": confidence,
            "start_time_seconds": start_seconds,
            "duration_seconds": float(segment_duration),
        }
        if start_time_utc is not None:
            segment_start_time_utc = start_time_utc + timedelta(seconds=start_seconds)
            segment_info["start_time_utc"] = format_utc_iso_z(segment_start_time_utc)
            segment_info["start_time_pacific"] = _format_pacific_timestamp(
                segment_start_time_utc
            )
        positive_segments.append(segment_info)

    return len(positive_segments), positive_segments


def run_inference(wav_path: str, model_type: str = "podsai",
                  model_path: Optional[str] = None,
                  model_revision: Optional[str] = None,
                  model_variant: str = "ast",
                  inference_model: Optional[Any] = None,
                  start_time_utc: Optional[datetime] = None,
                  feature_extractor: Optional[str] = None) -> dict:
    """
    Run inference on a wav file and return per-class probabilities.

    Args:
        wav_path: Path to the wav file.
        model_type: Type of model to use ('podsai', 'fastai', or 'orcahello').
        model_path: Path to the model directory or HuggingFace Hub model ID.
                    Required for podsai. Defaults to './model' for fastai,
                    'orcasound/orcahello-srkw-detector-v1' for orcahello,
                    and PODSAI_MODEL_ID for podsai.
        model_revision: Git commit hash to pin the HuggingFace Hub model revision.
                        Only used when model_path is a Hub model ID (not a local path).
                        Defaults to LATEST_PODSAI_MODEL_REVISION when model_path is the default
                        PODS-AI Hub model.
        feature_extractor: Feature extractor/config path or Hub ID when model_path is an ONNX file.
        model_variant: PODS-AI model variant to use when model_type is "podsai"
                       and the default PODS-AI Hub model/revision is being used
                       (model_path and model_revision are not explicitly set).
                       Supported values are "ast" (default) and "wav2vec2".
                       Ignored when model_path or model_revision is explicitly provided.
        inference_model: Optional preloaded model inference instance to reuse.
                         When provided, model loading via get_model_inference is skipped.

    Returns:
        Dictionary with:
            - probabilities: dict mapping class label to probability (0.0-1.0).
              Each value is the mean local_confidence for windows that predicted
              that class and whose confidence exceeds the model's threshold.
            - global_prediction_label: predicted class label for the whole file
            - global_prediction_labels: ordered class labels supported
              by the clip; older model results fall back to the single label
            - global_confidence: confidence score (0.0-1.0) for the global prediction
            - predict_time: time in seconds spent in the model's predict() method
            - positive_segments_count: number of positive PODS-AI segments above threshold
            - positive_segments: list of positive segment details (label/confidence/timestamps)
    """
    local_predictions: list[Any] = []
    local_confidences: list[Any] = []
    hop_duration = 2.0
    segment_duration = 3.0
    positive_segments_count = 0
    positive_segments: list[dict[str, Any]] = []
    global_prediction_labels: list[str] = []

    if model_type == "fastai":
        if model_path is None:
            model_path = "./model"
        model = (
            inference_model
            if inference_model is not None
            else get_model_inference(model_type="fastai", model_path=model_path)
        )

        start_time = time.perf_counter()
        result = model.predict(wav_path)
        predict_time = time.perf_counter() - start_time
        local_predictions = result.get("local_predictions", [])
        local_confidences = result.get("local_confidences", [])
        hop_duration = float(result.get("hop_duration", 1.0))
        segment_duration = float(result.get("segment_duration", 3.0))
        # local_confidences that exceed the threshold (resident windows).
        resident_prob = float(result.get("global_confidence", 0.0))
        other_prob = round(1.0 - resident_prob, 4)

        probabilities: dict[str, float] = {
            "other": other_prob,
            "resident": round(resident_prob, 4),
        }
        local_prediction_labels = [
            "resident" if int(local_prediction) == 1 else "other"
            for local_prediction in result.get("local_predictions", [])
        ]
        global_prediction = result.get("global_prediction", 0)
        global_prediction_label = "resident" if global_prediction else "other"
        global_prediction_labels = [global_prediction_label] if global_prediction_label == "resident" else []
        global_confidence = resident_prob

    elif model_type == "orcahello":
        if model_path is None:
            model_path = "orcasound/orcahello-srkw-detector-v1"
        model = (
            inference_model
            if inference_model is not None
            else get_model_inference(
                model_type="orcahello",
                model_path=model_path,
                model_revision=model_revision,
            )
        )

        start_time = time.perf_counter()
        result = model.predict(wav_path)
        predict_time = time.perf_counter() - start_time
        if result.get("error"):
            raise RuntimeError(
                f"OrcaHello inference failed for {wav_path}: {result['error']}"
            )
        local_predictions = result.get("local_predictions", [])
        local_confidences = result.get("local_confidences", [])
        hop_duration = float(result.get("hop_duration", 1.0))
        segment_duration = float(result.get("segment_duration", 2.0))

        # The OrcaHello SRKW detector is a binary classifier (other vs resident).
        resident_prob = float(result.get("global_confidence", 0.0))
        other_prob = round(1.0 - resident_prob, 4)

        probabilities = {
            "other": other_prob,
            "resident": round(resident_prob, 4),
        }
        local_prediction_labels = [
            "resident" if int(local_prediction) == 1 else "other"
            for local_prediction in result.get("local_predictions", [])
        ]
        global_prediction = result.get("global_prediction", 0)
        global_prediction_label = "resident" if global_prediction else "other"
        global_prediction_labels = [global_prediction_label] if global_prediction_label == "resident" else []
        global_confidence = resident_prob

    elif model_type == "podsai":
        if model_variant not in {"ast", "wav2vec2"}:
            raise ValueError(
                f"Unknown PODS-AI model variant: {model_variant!r}. Use 'ast' or 'wav2vec2'."
            )
        if model_path is None:
            model_path = PODSAI_MODEL_ID
            if model_revision is None:
                if model_variant == "wav2vec2":
                    model_revision = PODSAI_WAV2VEC2_MODEL_REVISION
                else:
                    model_revision = LATEST_PODSAI_MODEL_REVISION

        if inference_model is not None:
            model = inference_model
        elif isinstance(model_path, str) and model_path.lower().endswith(".onnx"):
            effective_feature_extractor = feature_extractor or PODSAI_MODEL_ID
            effective_revision = model_revision
            if feature_extractor is None and effective_revision is None:
                effective_revision = LATEST_PODSAI_MODEL_REVISION
            model = get_model_inference(
                model_type="podsai",
                model_path=model_path,
                feature_extractor=effective_feature_extractor,
                model_revision=effective_revision,
            )
        else:
            model = get_model_inference(
                model_type="podsai",
                model_path=model_path,
                model_revision=model_revision,
            )

        start_time = time.perf_counter()
        result = model.predict(wav_path)
        predict_time = time.perf_counter() - start_time
        local_predictions = result.get("local_predictions", [])
        local_confidences = result.get("local_confidences", [])
        hop_duration = float(result.get("hop_duration", 2.0))
        segment_duration = float(result.get("segment_duration", 3.0))

        probabilities = result["per_class_probabilities"]
        local_prediction_labels = []
        id2label = getattr(model, "id2label", {})
        for local_prediction in local_predictions:
            label = prediction_to_label(local_prediction, id2label)
            if isinstance(label, str):
                local_prediction_labels.append(label)
        global_prediction_label = result.get("global_prediction_label", "")
        global_prediction_labels = result.get("global_prediction_labels")
        if global_prediction_labels is None:
            from LiveInferenceOrchestrator import is_positive_label

            global_prediction_labels = (
                [global_prediction_label]
                if (
                    global_prediction_label
                    and is_positive_label(global_prediction_label)
                )
                else []
            )
        global_confidence = float(result.get("global_confidence", 0.0))
        positive_segments_count, positive_segments = calculate_positive_segments(
            local_predictions=local_predictions,
            local_confidences=local_confidences,
            hop_duration=hop_duration,
            segment_duration=segment_duration,
            id2label=id2label,
            threshold=getattr(model, "threshold", None),
            start_time_utc=start_time_utc,
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Use 'podsai', 'fastai', or 'orcahello'."
        )

    return {
        "probabilities": probabilities,
        "global_prediction_label": global_prediction_label,
        "global_prediction_labels": global_prediction_labels,
        "global_confidence": global_confidence,
        "predict_time": predict_time,
        "local_predictions": local_predictions,
        "local_confidences": local_confidences,
        "local_prediction_labels": local_prediction_labels,
        "hop_duration": hop_duration,
        "segment_duration": segment_duration,
        "positive_segments_count": positive_segments_count,
        "positive_segments": positive_segments,
    }


def print_results(results: dict, model_type: str) -> None:
    """Print inference results to stdout.

    Args:
        results: Dictionary returned by run_inference().
        model_type: Model type string, printed for context.
    """
    probabilities = results["probabilities"]
    label = results["global_prediction_label"]
    confidence = results["global_confidence"]
    predict_time = results.get("predict_time", 0.0)

    print(f"Model type: {model_type}")
    print(f"Global prediction: {label} (confidence: {confidence:.4f})")
    global_labels = results.get("global_prediction_labels", [])
    if len(global_labels) > 1:
        print(f"Global predictions: {', '.join(global_labels)}")
    print(f"Prediction time: {predict_time:.2f}s")

    local_predictions = results.get("local_predictions", [])
    local_confidences = results.get("local_confidences", [])
    local_prediction_labels = results.get("local_prediction_labels", [])
    hop_duration = results.get("hop_duration", 2.0)

    if local_predictions and local_confidences:
        print(f"\nSegments: {len(local_predictions)} total")

        if model_type == "podsai":
            positive_segments_count = results.get("positive_segments_count", 0)
            print(f"Positive segments: {positive_segments_count}/{len(local_predictions)}")

        print("\nAll segment predictions:")

        # Use local_prediction_labels if available, otherwise convert predictions.
        if local_prediction_labels:
            for idx, (pred_label, conf) in enumerate(zip(local_prediction_labels, local_confidences)):
                start_seconds = idx * hop_duration
                print(f"  +{start_seconds:.1f}s: {pred_label} (whale confidence: {conf:.3f})")
        else:
            # FastAI and OrcaHello use binary predictions (0=other, 1=resident).
            for idx, (pred, conf) in enumerate(zip(local_predictions, local_confidences)):
                start_seconds = idx * hop_duration
                pred_label = "resident" if int(pred) == 1 else "other"
                print(f"  +{start_seconds:.1f}s: {pred_label} (whale confidence: {conf:.3f})")

    print()
    print("Average whale likelihood by predicted class:")
    for class_name, prob in sorted(probabilities.items()):
        print(f"  {class_name}: {prob:.4f}")


def main() -> int:
    """Entry point for the run_inference CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run model inference on a wav file and output per-class probabilities. "
            "Provide either a local wav file path, or --node-name plus exactly one of "
            "--end-timestamp-str (PST end time) or --start-timestamp-utc."
        )
    )
    parser.add_argument(
        "wav_file",
        nargs="?",
        help="Path to the wav file to score.",
    )
    parser.add_argument(
        "--node-name",
        default=None,
        help="Feed node name (e.g., rpi_sunset_bay) used with timestamp arguments to download audio.",
    )
    parser.add_argument(
        "--end-timestamp-str",
        default=None,
        help=(
            "PST end timestamp used with --node-name to download audio "
            "(format: YYYY_MM_DD_HH_MM_SS_PST)."
        ),
    )
    parser.add_argument(
        "--start-timestamp-utc",
        default=None,
        help=(
            "UTC start timestamp used with --node-name to download audio "
            "(format: YYYY-MM-DDTHH:MM:SSZ)."
        ),
    )
    parser.add_argument(
        "--set",
        default="testing",
        type=str.lower,
        choices=("training", "testing"),
        help=(
            "Sample set used when downloading audio with --node-name: "
            "training downloads a 3-second clip, testing downloads a 60-second clip "
            "(default: testing). Ignored when wav_file is provided."
        ),
    )
    parser.add_argument(
        "--model",
        default="podsai",
        help=(
            "Model type to use (default: podsai). "
            "podsai: 8-class model (water, resident, transient, humpback, vessel, jingle, human, bird). "
            "fastai: 2-class model (other, resident). "
            "orcahello: 2-class SRKW detector (other, resident) using the OrcaHello ResNet50 model."
        ),
    )
    parser.add_argument(
        "--type",
        default="ast",
        choices=("ast", "wav2vec2"),
        help=(
            "PODS-AI model type to use with --model podsai (default: ast). "
            "ast selects the AST-based checkpoint, wav2vec2 selects the older Wav2Vec2 checkpoint."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to model directory or HuggingFace Hub model ID. "
            "Defaults to ./model for --model fastai. "
            "Defaults to orcasound/orcahello-srkw-detector-v1 for --model orcahello. "
            f"Defaults to {PODSAI_MODEL_ID!r} for --model podsai."
        ),
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Git commit hash to pin the HuggingFace Hub model revision. "
            "Only used when --model-path is a Hub model ID. "
            f"Defaults to the pinned revision ({LATEST_PODSAI_MODEL_REVISION}) when using "
            f"the default PODS-AI model ({PODSAI_MODEL_ID!r})."
        ),
    )
    parser.add_argument(
        "--feature-extractor",
        default=None,
        metavar="PATH_OR_HF_ID",
        help="Feature extractor/config to use when --model-path is an ONNX file.",
    )

    args = parser.parse_args()
    end_timestamp_str = args.end_timestamp_str
    timestamp_args_provided = sum(
        [
            end_timestamp_str is not None,
            args.start_timestamp_utc is not None,
        ]
    )
    if args.node_name is None and timestamp_args_provided > 0:
        print(
            "Error: --node-name is required when using timestamp arguments.",
            file=sys.stderr,
        )
        return 1
    if args.node_name is not None and timestamp_args_provided != 1:
        print(
            "Error: with --node-name, provide exactly one of --end-timestamp-str "
            "or --start-timestamp-utc.",
            file=sys.stderr,
        )
        return 1

    with ExitStack() as stack:
        start_time_utc: Optional[datetime] = None
        if args.wav_file:
            if args.node_name is not None:
                print(
                    "Error: provide either wav_file or --node-name with one timestamp argument, not both.",
                    file=sys.stderr,
                )
                return 1
            wav_path = args.wav_file
            if not Path(wav_path).exists():
                print(f"Error: wav file not found: {wav_path}", file=sys.stderr)
                return 1
        else:
            if args.node_name is None:
                print(
                    "Error: either provide wav_file, or provide --node-name with one timestamp argument.",
                    file=sys.stderr,
                )
                return 1

            temp_dir = stack.enter_context(TemporaryDirectory())
            duration_seconds = 3.0 if args.set == "training" else 60.0
            try:
                if args.start_timestamp_utc is not None:
                    start_time_utc = parse_utc_start_timestamp(args.start_timestamp_utc)
                else:
                    end_time_pst = parse_pst_end_timestamp(end_timestamp_str)
                    start_time_utc = end_time_pst.astimezone(UTC_TZ) - timedelta(seconds=duration_seconds)
                wav_path = download_audio_from_start_utc(
                    args.node_name, start_time_utc, temp_dir, duration_seconds=duration_seconds
                )
            except ValueError as e:
                print(f"Failed to download wav: {e}", file=sys.stderr)
                return 1
            except Exception as e:
                print(f"Failed to download wav: {e}", file=sys.stderr)
                return 1
            if not wav_path:
                print("Error: failed to download wav file.", file=sys.stderr)
                return 1
            if not Path(wav_path).exists():
                print(f"Error: downloaded wav file not found: {wav_path}", file=sys.stderr)
                return 1

        try:
            results = run_inference(
                wav_path,
                model_type=args.model,
                model_path=args.model_path,
                model_revision=args.model_revision,
                model_variant=args.type,
                start_time_utc=start_time_utc,
                feature_extractor=args.feature_extractor,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Inference failed: {e}", file=sys.stderr)
            return 1

    print_results(results, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
