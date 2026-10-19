#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Compare multiple models on a test set of audio samples.

Usage:
    python compare_models.py [options]

Loads a test set from WAV files under output/testing-wav, then runs each enabled model
(fastai, orcahello, wav2vec2 (old PODS-AI with Wav2Vec2)), oldpodsai or podsai (AST) on the corresponding
60-second WAV files and reports correct identifications, whale-class F1, and
per-whale-class false-positive/false-negative rates per model.

A "correct" identification means:
  - For fastai and orcahello, model predicted "resident" (SRKW) when the label is
    "resident", or anything other than "resident" when the label is not "resident".
  - For custom, podsai, oldpodsai, and wav2vec2, the actual category is present in the model's
    multi-label prediction set (with a single-label fallback for older models).
    Abiotic samples also accept water, vessel, or jingle predictions.

For each whale class X (resident, transient, humpback):
  - X false positives are samples predicted as X when the correct label was not X.
  - X false negatives are samples whose correct label is X but the model predicted something else.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from model_inference import get_model_inference
from model_revisions import (
    LATEST_PODSAI_MODEL_REVISION,
    PODSAI_WAV2VEC2_MODEL_REVISION,
    PREVIOUS_PODSAI_MODEL_REVISION,
)
from run_inference import run_inference

RESIDENT_LABEL = "resident"
ABIOTIC_LABELS = {"abiotic", "water", "vessel", "jingle"}
WHALE_CLASS_NAMES = {"humpback", "resident", "transient"}
BACKGROUND_LABELS = {
    "abiotic", "bird", "human", "jingle", "other", "vessel", "water"
}
SUMMARY_LABELS = [
    ("resident", "R"),
    ("transient", "T"),
    ("humpback", "H"),
]
MATRIX_CELL_PADDING = 2
PODSAI_MODEL_ID = "davethaler/whale-call-detector"

# Maps user-facing model names to inference backends. wav2vec2 and oldpodsai reuse podsai
# inference with a different pinned model revision.
MODEL_TYPE_TO_INFERENCE_TYPE = {
    "fastai": "fastai",
    "orcahello": "orcahello",
    "wav2vec2": "podsai",
    "oldpodsai": "podsai",
    "podsai": "podsai",
}
CUSTOM_MODEL_TYPE = "podsai"
DEFAULT_MODELS = "fastai,orcahello,wav2vec2,oldpodsai,podsai"
MODEL_LOAD_ERRORS = (
    AttributeError,
    EOFError,
    ImportError,
    IndexError,
    KeyError,
    NotImplementedError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class ModelResult:
    """Accumulated results for a single model across all test samples."""

    model_type: str
    total: int = 0
    correct: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    skipped: int = 0
    predict_times: list[float] = field(default_factory=list)
    # Maps actual_label -> {predicted_label -> count} for each evaluated sample.
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    # Full prediction sets used for multi-label metrics. The confusion matrix
    # remains a primary-label view for backwards-compatible display output.
    prediction_labels: dict[str, list[set[str]]] = field(default_factory=dict)

    @property
    def evaluated(self) -> int:
        """Number of samples actually evaluated (not skipped)."""
        return self.total - self.skipped

    @property
    def accuracy(self) -> Optional[float]:
        """Fraction of evaluated samples correctly identified."""
        if self.evaluated == 0:
            return None
        return self.correct / self.evaluated

    @property
    def false_positive_rate(self) -> Optional[float]:
        """Fraction of evaluated samples that are false positives."""
        if self.evaluated == 0:
            return None
        return self.false_positives / self.evaluated

    @property
    def false_negative_rate(self) -> Optional[float]:
        """Fraction of evaluated samples that are false negatives."""
        if self.evaluated == 0:
            return None
        return self.false_negatives / self.evaluated

    @property
    def avg_predict_time(self) -> Optional[float]:
        """Average time in seconds spent in predict() method per WAV file."""
        if not self.predict_times:
            return None
        return sum(self.predict_times) / len(self.predict_times)

    @property
    def whale_f1(self) -> Optional[float]:
        """Macro F1 across whale classes in actual labels or prediction sets."""
        labels_seen = _labels_seen_in_confusion_matrix(self.confusion_matrix)
        for prediction_sets in self.prediction_labels.values():
            for labels in prediction_sets:
                labels_seen.update(labels)
        whale_labels = sorted(label for label in labels_seen if label in WHALE_CLASS_NAMES)
        if not whale_labels:
            return None

        f1_scores = []
        for label in whale_labels:
            actual_count = self.actual_count_for_label(label)
            false_negatives = self.false_negative_count_for_label(label)
            true_positives = actual_count - false_negatives
            false_positives = self.false_positive_count_for_label(label)

            precision_denominator = true_positives + false_positives
            recall_denominator = true_positives + false_negatives
            precision = (
                true_positives / precision_denominator
                if precision_denominator else 0.0
            )
            recall = (
                true_positives / recall_denominator
                if recall_denominator else 0.0
            )
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append((2 * precision * recall) / (precision + recall))

        return sum(f1_scores) / len(f1_scores)

    def actual_count_for_label(self, label: str) -> int:
        """Return how many evaluated samples have the given ground-truth label."""
        return sum(self.confusion_matrix.get(label, {}).values())

    def false_positive_count_for_label(self, label: str) -> int:
        """Return the number of evaluated samples incorrectly predicted as the given label."""
        if self.prediction_labels:
            return sum(
                label in labels
                for actual_label, prediction_sets in self.prediction_labels.items()
                if actual_label != label
                for labels in prediction_sets
            )
        return sum(
            predicted_counts.get(label, 0)
            for actual_label, predicted_counts in self.confusion_matrix.items()
            if actual_label != label
        )

    def false_negative_count_for_label(self, label: str) -> int:
        """Return the number of evaluated samples with the given label predicted as something else."""
        if self.prediction_labels:
            return sum(
                label not in labels
                for labels in self.prediction_labels.get(label, [])
            )
        return sum(
            count
            for predicted_label, count in self.confusion_matrix.get(label, {}).items()
            if predicted_label != label
        )

    def false_positive_rate_for_label(self, label: str) -> Optional[float]:
        """Return the fraction of non-label samples incorrectly predicted as the given label.

        Returns None when there are no evaluated samples whose actual label differs from
        the given label.
        """
        negative_count = self.evaluated - self.actual_count_for_label(label)
        if negative_count == 0:
            return None
        return self.false_positive_count_for_label(label) / negative_count

    def false_negative_rate_for_label(self, label: str) -> Optional[float]:
        """Return the fraction of actual label samples predicted as something else.

        Returns None when there are no evaluated samples whose actual label matches the
        given label.
        """
        actual_count = self.actual_count_for_label(label)
        if actual_count == 0:
            return None
        return self.false_negative_count_for_label(label) / actual_count


def get_category_from_path(wav_path: Path, wav_dir: Path) -> str:
    relative_path = wav_path.relative_to(wav_dir)
    if len(relative_path.parts) != 2:
        return None

    category = relative_path.parts[0]
    return category


def load_test_samples(wav_dir: Path, max_samples: Optional[int] = None,
                      category_filter: Optional[str] = None) -> list[Path]:
    """
    Load test samples from WAV files in the testing directory.

    Args:
        wav_dir: Root directory containing category subdirectories with WAV files.
        max_samples: Maximum number of samples to load. If None, load all samples.
        category_filter: If specified, only load samples matching this category.
                        If None, load samples from all categories.

    Returns:
        List of WAV file paths, or an empty list on error.
    """
    samples = []
    try:
        wav_paths = sorted(
            path for path in wav_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        )
        for wav_path in wav_paths:
            category = get_category_from_path(wav_path, wav_dir)
            if category is None:
                continue
            if category_filter is not None and category != category_filter:
                continue

            samples.append(wav_path)

            if max_samples is not None and len(samples) >= max_samples:
                break
    except OSError as e:
        print(f"Error reading WAV files from {wav_dir}: {e}", file=sys.stderr)
    return samples


def is_exact_match_model(model_type: str) -> bool:
    """Return whether a model uses exact category membership for correctness."""
    return model_type in {"custom", "podsai", "oldpodsai", "wav2vec2"}


def parse_model_spec(spec: str) -> tuple[str, str, Optional[str]]:
    """Parse one model name from ``--models``.

    Args:
        spec: A built-in model name or the literal ``custom``.

    Returns:
        A tuple containing the display name, inference backend, and an empty
        custom path slot. The custom path is supplied separately by
        ``--custom-model-path``.

    Raises:
        ValueError: If ``spec`` is unknown or uses the removed inline custom
            path syntax.
    """
    model = spec.strip()
    if model == "custom":
        return model, CUSTOM_MODEL_TYPE, None
    if model not in MODEL_TYPE_TO_INFERENCE_TYPE:
        valid = ", ".join(sorted((*MODEL_TYPE_TO_INFERENCE_TYPE, "custom")))
        raise ValueError(
            f"unknown model type {model!r}; valid models: {valid}"
        )
    return model, MODEL_TYPE_TO_INFERENCE_TYPE[model], None


def resolve_custom_model_path(model_path: str) -> tuple[str, bool]:
    """Validate and normalize a custom model path or Hugging Face ID.

    Args:
        model_path: Local file/directory path or Hugging Face repository ID.

    Returns:
        A tuple of the normalized model value and whether it is a local path.

    Raises:
        ValueError: If the value is empty, names an invalid local path, or
            cannot be expanded.
    """
    if not model_path or not model_path.strip():
        raise ValueError("--custom-model-path must not be empty")
    model_path = model_path.strip()
    try:
        path = Path(model_path).expanduser()
    except (RuntimeError, OSError, ValueError) as exc:
        raise ValueError(f"invalid custom model path: {model_path}") from exc

    try:
        path_exists = path.exists()
        valid_local_path = (path.is_file() or path.is_dir()) if path_exists else False
    except OSError as exc:
        raise ValueError(f"unable to access local custom model path: {model_path}") from exc
    if path_exists:
        if not valid_local_path:
            raise ValueError(f"local custom model path is not a file or directory: {model_path}")
        return str(path), True
    looks_like_local_path = (
        path.is_absolute()
        or model_path.startswith((".", "~", "/", "\\"))
        or (len(model_path) > 1 and model_path[1] == ":")
    )
    if looks_like_local_path:
        raise ValueError(f"local custom model path does not exist: {model_path}")
    return model_path, False


def validate_custom_model_path(model_path: str) -> None:
    """Validate a custom model path without changing the existing API.

    Args:
        model_path: Local file/directory path or Hugging Face repository ID.

    Raises:
        ValueError: If the value is empty or names an invalid local path.
    """
    resolve_custom_model_path(model_path)


def validate_inference_model(model: Any, model_name: str) -> None:
    """Validate the prediction method required by a custom model.

    Args:
        model: Object returned by the inference model factory.
        model_name: Display name used in the validation error.

    Raises:
        ValueError: If the object does not expose a callable ``predict``.
    """
    if not callable(getattr(model, "predict", None)):
        raise ValueError(
            f"{model_name!r} did not load a PODS-AI inference model with predict()"
        )


def is_correct_prediction(
    actual_label: str,
    predicted_label: str,
    model_type: str,
    predicted_labels: Optional[list[str]] = None,
) -> bool:
    """Return whether a prediction should count toward the Correct column.

    Args:
        actual_label: Ground-truth category for the sample.
        predicted_label: Model-predicted category for the sample.
        model_type: Model family used to interpret correctness.
        predicted_labels: Optional complete set of positive labels emitted by a
                          multi-label model. When provided, membership in this set
                          determines correctness for exact-match models and resident
                          membership determines correctness for binary models. When
                          omitted or empty, the primary ``predicted_label`` is used
                          as a single-label fallback.
                          Abiotic ground truth also accepts water, vessel, or jingle.

    Returns:
        True when the prediction is correct under the model-specific summary rules.
    """
    labels = predicted_labels or [predicted_label]
    if is_exact_match_model(model_type):
        if actual_label == "abiotic":
            return bool(ABIOTIC_LABELS.intersection(labels))
        return actual_label in labels
    if actual_label in BACKGROUND_LABELS:
        return RESIDENT_LABEL not in labels
    return (RESIDENT_LABEL in labels) == (actual_label == RESIDENT_LABEL)


def _labels_seen_in_confusion_matrix(confusion_matrix: dict[str, dict[str, int]]) -> set[str]:
    """Return all labels that appear as actual or predicted in a confusion matrix.

    Args:
        confusion_matrix: Mapping of actual labels to per-predicted-label counts.

    Returns:
        Set of unique labels appearing either as actual labels or predicted labels.
    """
    labels = set(confusion_matrix)
    for predicted_counts in confusion_matrix.values():
        labels.update(predicted_counts)
    return labels


def evaluate_model(
    model_type: str,
    model_path: Optional[str],
    wav_paths: list[Path],
    wav_dir: Path,
    model_revision: Optional[str] = None,
    result_model_type: Optional[str] = None,
    inference_model: Optional[Any] = None,
) -> ModelResult:
    """
    Run a model against all test samples and accumulate results.

    Args:
        model_type: One of 'fastai', 'orcahello', 'wav2vec2', 'oldpodsai', 'podsai', or
                    'custom'. 'oldpodsai', 'wav2vec2', and 'custom' are mapped to
                    'podsai' inference internally.
        model_path: Path to the model (or HuggingFace Hub model ID).
        wav_paths: List of WAV file paths.
        wav_dir: Root directory containing testing WAV files.
        model_revision: Git commit hash to pin the HuggingFace Hub model revision.
                        Only used when model_path is a Hub model ID (not a local path).
        result_model_type: Optional display name to store in ModelResult.model_type.
                           If omitted, model_type is used.
        inference_model: Optional preloaded model inference instance to reuse.

    Returns:
        ModelResult with counts of correct, false positive, and false negative predictions,
        plus timing information for predict() calls.
    """
    result = ModelResult(model_type=result_model_type or model_type, total=len(wav_paths))
    display_model_type = result.model_type

    for wav_path in wav_paths:
        category = get_category_from_path(wav_path, wav_dir)
        if category is None:
            continue
        expected_resident = (category == RESIDENT_LABEL)
        relative_path = wav_path.relative_to(wav_dir)

        try:
            inference_result = run_inference(str(wav_path), model_type=model_type,
                                             model_path=model_path,
                                             model_revision=model_revision,
                                             inference_model=inference_model)
            predict_time = inference_result.get("predict_time", 0.0)
            result.predict_times.append(predict_time)
        except Exception as e:
            print(f"  [{display_model_type}] Error on {wav_path.name}: {e}")
            result.skipped += 1
            continue

        predicted_label = inference_result.get("global_prediction_label", "")
        predicted_labels = inference_result.get("global_prediction_labels")
        if not predicted_labels:
            predicted_labels = [predicted_label] if predicted_label else []
        predicted_resident = RESIDENT_LABEL in predicted_labels

        if is_correct_prediction(
            category, predicted_label, model_type, predicted_labels
        ):
            result.correct += 1
            status = "correct"
        elif predicted_resident and not expected_resident:
            result.false_positives += 1
            status = "false_positive"
        elif expected_resident and not predicted_resident:
            result.false_negatives += 1
            status = "false_negative"
        else:
            status = "incorrect"

        # Keep a primary-label matrix for display and the complete set for metrics.
        actual_label = category
        result.prediction_labels.setdefault(actual_label, []).append(set(predicted_labels))
        if actual_label not in result.confusion_matrix:
            result.confusion_matrix[actual_label] = {}
        preds = result.confusion_matrix[actual_label]
        preds[predicted_label] = preds.get(predicted_label, 0) + 1

        print(
            f"  [{display_model_type}] {relative_path}: "
            f"predicted={predicted_label!r} -> {status} ({predict_time:.2f}s)"
        )

    return result


def print_confusion_matrix(result: ModelResult) -> None:
    """
    Print a per-class confusion matrix for a single model result.

    Rows are actual (ground-truth) labels; columns are predicted labels.
    Only labels with at least one non-zero entry in their row (for actuals) or
    column (for predicted) are shown; all-zero rows and columns are omitted.

    Args:
        result: ModelResult whose confusion_matrix to display.
    """
    matrix = result.confusion_matrix
    if not matrix:
        return

    # Collect actual labels (rows) that have at least one non-zero prediction.
    actual_labels = sorted(
        actual for actual in matrix
        if any(count > 0 for count in matrix[actual].values())
    )

    # Collect predicted labels (columns) that have at least one non-zero count.
    predicted_labels = sorted(
        set(
            predicted
            for preds in matrix.values()
            for predicted, count in preds.items()
            if count > 0
        )
    )

    if not actual_labels or not predicted_labels:
        return

    row_totals = {actual: sum(matrix.get(actual, {}).values()) for actual in actual_labels}
    all_labels = sorted(set(actual_labels) | set(predicted_labels))
    widest_total = max(len("total"), max(len(str(total)) for total in row_totals.values()))
    col_width = max(max(len(label) for label in predicted_labels), widest_total) + MATRIX_CELL_PADDING
    row_label_width = max(len(label) for label in all_labels) + MATRIX_CELL_PADDING

    print(f"Confusion Matrix for {result.model_type} (rows=actual, cols=predicted):")
    print(f"{'':>{row_label_width}}", end="")
    for label in predicted_labels:
        print(f"{label:>{col_width}}", end="")
    print(f"{'total':>{col_width}}", end="")
    print()

    for actual in actual_labels:
        print(f"{actual:>{row_label_width}}", end="")
        for predicted in predicted_labels:
            count = matrix.get(actual, {}).get(predicted, 0)
            print(f"{count:>{col_width}}", end="")
        print(f"{row_totals[actual]:>{col_width}}", end="")
        print()


def print_summary(results: list[ModelResult]) -> None:
    """
    Print a formatted comparison table for all model results.

    Args:
        results: List of ModelResult objects, one per model.
    """
    class_column_format = " {:>7} {:>7}"
    header = (
        f"{'Model':<15} {'Evaluated':>9} {'Correct':>9} {'Accuracy':>9} {'F1':>7}"
        f"{class_column_format.format('RFP%', 'RFN%')}"
        f"{class_column_format.format('TFP%', 'TFN%')}"
        f"{class_column_format.format('HFP%', 'HFN%')}"
        f" {'Avg Time':>10}"
    )
    separator = "=" * len(header)
    print()
    print(separator)
    print("Model Comparison Summary")
    print(separator)
    print(header)
    print("-" * len(header))

    for r in results:
        evaluated = r.evaluated
        accuracy = f"{r.accuracy:.1%}" if r.accuracy is not None else "N/A"
        whale_f1 = f"{r.whale_f1:.3f}" if r.whale_f1 is not None else "N/A"
        avg_time = f"{r.avg_predict_time:.2f}s" if r.avg_predict_time is not None else "N/A"
        class_stats = []
        for label, _ in SUMMARY_LABELS:
            false_positive_rate = r.false_positive_rate_for_label(label)
            false_negative_rate = r.false_negative_rate_for_label(label)
            class_stats.append(
                class_column_format.format(
                    f"{false_positive_rate:.1%}" if false_positive_rate is not None else "N/A",
                    f"{false_negative_rate:.1%}" if false_negative_rate is not None else "N/A",
                )
            )

        print(
            f"{r.model_type:<15} {evaluated:>9} {r.correct:>9} {accuracy:>9} {whale_f1:>7}"
            f"{''.join(class_stats)} {avg_time:>10}"
        )
        if r.skipped:
            print(f"  ({r.skipped} skipped due to missing WAV or inference error)")

    print(separator)
    print()
    print("Definitions:")
    print("  Accuracy     = Correct / Evaluated")
    print("  Correct      = fastai/orcahello: resident vs other; wav2vec2/oldpodsai/podsai/custom: category in prediction set")
    print("                 abiotic also accepts water, vessel, or jingle")
    print("  F1           = macro F1 over humpback, resident, and transient classes that are present")
    print("  [R|T|H]FP%   = among non-[R|T|H] samples, fraction predicted as that class")
    print("  [R|T|H]FN%   = among actual samples of that class, fraction predicted as another class")
    print("  Avg Time     = average time spent in model predict() per 60-second WAV file")
    print("  Note         = compares end-to-end 60-second inference on all WAV files in --wav-dir")

    for r in results:
        print()
        print_confusion_matrix(r)


def main() -> int:
    """Entry point for the compare_models CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare model predictions on a test set discovered from WAV files in --wav-dir. "
            "Runs each enabled model against 60-second WAV files "
            "and reports correct identifications, false positives, and false negatives."
        )
    )
    parser.add_argument(
        "--wav-dir",
        default="output/testing-wav",
        help="Root directory containing testing WAV files (default: output/testing-wav).",
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated list of models to evaluate. "
            "Include custom for an additional PODS-AI-compatible model. "
            f"(default: {DEFAULT_MODELS})."
        ),
    )
    parser.add_argument(
        "--custom-model-path",
        default=None,
        metavar="PATH_OR_HF_ID",
        help=(
            "Local path or Hugging Face model ID for the custom model. "
            "Providing this option automatically adds custom to --models."
        ),
    )
    parser.add_argument(
        "--custom-model-revision",
        default=None,
        metavar="REVISION",
        help=(
            "Optional revision to use when loading a custom Hugging Face model. "
            "Ignored for local paths."
        ),
    )
    parser.add_argument(
        "--fastai-model-path",
        default="model",
        help=(
            "Path to FastAI model directory. "
            "Defaults to model when not specified."
        ),
    )
    parser.add_argument(
        "--orcahello-model-path",
        default="orcasound/orcahello-srkw-detector-v1",
        help=(
            "Path or HuggingFace Hub ID for the OrcaHello model. "
            "Defaults to orcasound/orcahello-srkw-detector-v1 when not specified."
        ),
    )
    parser.add_argument(
        "--orcahello-model-revision",
        dest="orcahello_model_revision",
        default=None,
        metavar="REVISION",
        help=(
            "Hugging Face revision for the OrcaHello model. "
            "Only used when --orcahello-model-path is a Hub model ID."
        ),
    )
    parser.add_argument(
        "--podsai-model-path",
        default=PODSAI_MODEL_ID,
        help=(
            "Path to PODS-AI model directory or HuggingFace Hub ID. "
            "Used by wav2vec2 (Wav2Vec2), oldpodsai (AST), and podsai (AST). "
            f"Defaults to {PODSAI_MODEL_ID!r} when not specified."
        ),
    )
    parser.add_argument(
        "--podsai-model-revision",
        default=LATEST_PODSAI_MODEL_REVISION,
        help=(
            "Git commit hash to pin the PODS-AI HuggingFace Hub model revision. "
            "Only used when --podsai-model-path is a Hub model ID. "
            f"Defaults to the pinned revision ({LATEST_PODSAI_MODEL_REVISION})."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Maximum number of test samples to process. "
            "If not specified, all samples are processed."
        ),
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Only test samples from this category. "
            "If not specified, all categories are tested. "
            "Categories: abiotic, water, resident, transient, humpback, vessel, jingle, human, bird."
        ),
    )

    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    if not wav_dir.exists():
        print(f"Error: WAV directory not found: {wav_dir}", file=sys.stderr)
        print(
            "Run download_wavs.py first to download testing WAV files.",
            file=sys.stderr,
        )
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    try:
        if args.custom_model_path is not None and "custom" not in models:
            models.append("custom")
        if args.custom_model_revision is not None and "custom" not in models:
            raise ValueError(
                "--custom-model-revision requires custom in --models or "
                "--custom-model-path"
            )
        parsed_models = [parse_model_spec(model) for model in models]
        seen_models: set[str] = set()
        for display_name, _, _ in parsed_models:
            if display_name in seen_models:
                raise ValueError(
                    f"{display_name} may be listed only once in --models"
                )
            seen_models.add(display_name)
        custom_count = sum(display_name == "custom" for display_name, _, _ in parsed_models)
        if custom_count and args.custom_model_path is None:
            raise ValueError(
                "custom is listed in --models but --custom-model-path was not provided"
            )
        custom_path = None
        custom_path_is_local = False
        if args.custom_model_path is not None:
            custom_path, custom_path_is_local = resolve_custom_model_path(
                args.custom_model_path
            )
        model_specs = [
            (display_name, inference_type, custom_path if display_name == "custom" else None)
            for display_name, inference_type, _ in parsed_models
        ]
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    model_paths: dict[str, Optional[str]] = {
        "fastai": args.fastai_model_path,
        "orcahello": args.orcahello_model_path,
        "wav2vec2": args.podsai_model_path,
        "oldpodsai": args.podsai_model_path,
        "podsai": args.podsai_model_path,
    }
    model_revisions: dict[str, Optional[str]] = {
        "fastai": None,
        "orcahello": args.orcahello_model_revision,
        "wav2vec2": PODSAI_WAV2VEC2_MODEL_REVISION,
        "oldpodsai": PREVIOUS_PODSAI_MODEL_REVISION,
        "podsai": args.podsai_model_revision,
    }

    # Validate max_samples if specified.
    if args.max_samples is not None and args.max_samples <= 0:
        print(f"Error: --max-samples must be a positive integer, got {args.max_samples}", file=sys.stderr)
        return 1

    wav_paths = load_test_samples(wav_dir, max_samples=args.max_samples,
                                category_filter=args.category)
    if not wav_paths:
        if args.category:
            print(f"Error: no test samples found for category '{args.category}'.", file=sys.stderr)
        else:
            print("Error: no test samples found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(wav_paths)} test samples from WAV files in {wav_dir}")
    if args.category:
        print(f"  (filtered to category: {args.category})")
    if args.max_samples is not None:
        print(f"  (limited to first {args.max_samples} samples)")
    print(f"WAV directory: {wav_dir}")
    print(f"Models to evaluate: {', '.join(models)}")
    print()

    results = []
    for model_type, inference_model_type, custom_path in model_specs:
        print(f"Evaluating model: {model_type}")
        effective_model_path = custom_path or model_paths[model_type]
        if custom_path:
            effective_revision = (
                None if custom_path_is_local else args.custom_model_revision
            )
        else:
            effective_revision = model_revisions[model_type]
        try:
            inference_model = get_model_inference(
                model_type=inference_model_type,
                model_path=effective_model_path,
                model_revision=effective_revision,
            )
            if custom_path:
                validate_inference_model(inference_model, model_type)
        except MODEL_LOAD_ERRORS as exc:
            if custom_path:
                print(
                    f"Error loading custom model {custom_path!r}: "
                    "the model must be compatible with the PODS-AI inference "
                    f"contract ({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                )
            else:
                print(f"Error loading {model_type} model: {exc}", file=sys.stderr)
            return 1
        model_result = evaluate_model(
            model_type=inference_model_type,
            model_path=effective_model_path,
            wav_paths=wav_paths,
            wav_dir=wav_dir,
            model_revision=effective_revision,
            result_model_type=model_type,
            inference_model=inference_model,
        )
        results.append(model_result)
        print()

    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
