#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Train a PODS-AI audio classification model for orca call detection.

This script fine-tunes a HuggingFace audio classification model on orca call
audio. The default base model uses spectrogram features rather than raw-audio
Wav2Vec2 embeddings. The trained model can be pushed to HuggingFace Hub or
saved locally.

Usage:
    # Binary classification (other vs any call)
    python train_podsai_model.py --num_classes 2 --output_dir ./model/binary

    # Multi-class classification (water, resident, transient, humpback, vessel, jingle, human, bird, hum)
    python train_podsai_model.py --num_classes 9 --output_dir ./model/multiclass
"""

import argparse
from collections import Counter
from functools import partial
import math
import os
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

# Configure datasets to use soundfile for audio decoding BEFORE importing datasets components.
import datasets.config
datasets.config.AUDIO_BACKENDS_USE_TORCH = False
datasets.config.AUDIOCODEC_DEFAULT_DECODER = "soundfile"

from datasets import Dataset, Audio, DatasetDict, ClassLabel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
import evaluate

# Verify audio decoding dependencies are available.
try:
    import librosa
    import soundfile
except ImportError as e:
    print("Error: Missing required audio decoding libraries.")
    print("Please install the required dependencies:")
    print("  pip install -r requirements.txt")
    print(f"\nSpecific error: {e}")
    raise

# Get repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Label mappings (will be set based on num_classes).
LABEL2ID = {}
ID2LABEL = {}


class _SklearnMetricFallback:
    """Fallback metric implementation used when evaluate metrics are unavailable."""

    def __init__(self, metric_name: str):
        self.metric_name = metric_name

    def compute(
        self,
        predictions: list[int] | np.ndarray,
        references: list[int] | np.ndarray,
        average: str | None = None,
        labels: list[int] | None = None,
    ) -> dict:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

        if self.metric_name == "accuracy":
            return {"accuracy": float(accuracy_score(references, predictions))}

        precision, recall, f1, _ = precision_recall_fscore_support(
            references,
            predictions,
            average=average,
            labels=labels,
            zero_division=0,
        )
        metric_value = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }[self.metric_name]
        return {self.metric_name: metric_value}


def _load_metric(metric_name: str):
    """Load an evaluate metric with sklearn fallback if unavailable."""
    try:
        return evaluate.load(metric_name)
    except FileNotFoundError:
        print(
            f"Warning: evaluate metric '{metric_name}' unavailable; using sklearn fallback.",
        )
        return _SklearnMetricFallback(metric_name)


# Load metrics once at module scope.
ACCURACY_METRIC = _load_metric("accuracy")
PRECISION_METRIC = _load_metric("precision")
RECALL_METRIC = _load_metric("recall")
F1_METRIC = _load_metric("f1")

# Whale classes for optimization in multi-class mode.
WHALE_CLASS_NAMES = {"humpback", "resident", "transient"}
CHECKPOINT_SAVE_LIMIT = 6
DEFAULT_MAX_PREPROCESSING_WORKERS = 8
TRAIN_TEST_SPLIT_TEST_SIZE = 0.2


class FeatureExtractorProtocol(Protocol):
    """Protocol for audio feature extractors used during PODS-AI training."""

    def __call__(self, processed_audio: list[np.ndarray], *, sampling_rate: int, padding: bool) -> dict:
        """Convert audio arrays into model inputs.

        Args:
            processed_audio: Fixed-length audio clips for the current batch.
            sampling_rate: Sampling rate of the provided clips.
            padding: Whether the extractor should apply its batch padding logic.

        Returns:
            Dictionary of model-ready features such as input_values or attention_mask.
        """


def setup_label_mappings(num_classes: int) -> None:
    """
    Set up label mappings based on number of classes.

    Args:
        num_classes: 2 for binary (other vs call), 9 for multi-class including "hum"
    """
    global LABEL2ID, ID2LABEL

    if num_classes == 2:
        # Binary: other (0) vs any whale (1).
        LABEL2ID = {"other": 0, "whale": 1}
        ID2LABEL = {0: "other", 1: "whale"}
        print("Using BINARY classification: other vs whale")
    elif num_classes == 9:
        # Multi-class: water, resident, transient, humpback, vessel, jingle, human, bird, hum.
        LABEL2ID = {"water": 0, "resident": 1, "transient": 2, "humpback": 3, "vessel": 4, "jingle": 5, "human": 6, "bird": 7, "hum": 8}
        ID2LABEL = {0: "water", 1: "resident", 2: "transient", 3: "humpback", 4: "vessel", 5: "jingle", 6: "human", 7: "bird", 8: "hum"}
        print("Using MULTI-CLASS classification: water, resident, transient, humpback, vessel, jingle, human, bird, hum")
    else:
        raise ValueError(f"num_classes must be 2 or 9, got {num_classes}")


def can_stratify_split(num_unique_labels: int, min_samples_per_label: int, estimated_test_size: int) -> bool:
    """
    Decide whether a stratified train/test split is feasible.

    Args:
        num_unique_labels: Number of distinct labels present in the dataset.
        min_samples_per_label: Smallest per-label sample count across all labels.
        estimated_test_size: Expected number of samples in the test split.

    Returns:
        True if stratification is possible, i.e. there are at least 2 unique
        labels, each label has at least 2 samples, and the test split itself
        is large enough to hold at least one sample per label.  A freshly-added
        category (e.g. "hum") may start with very few samples, and with many
        classes a small test split can end up smaller than the number of
        classes, which ``train_test_split``'s ``stratify_by_column`` rejects
        with a ``ValueError``.
    """
    return (
        num_unique_labels > 1
        and min_samples_per_label >= 2
        and estimated_test_size >= num_unique_labels
    )


def load_audio_dataset(data_dir: Path, num_classes: int) -> DatasetDict:
    """
    Load audio files from the output/wav directory structure.

    Args:
        data_dir: Path to output/wav directory
        num_classes: 2 for binary, 9 for multi-class including "hum"

    Returns:
        DatasetDict with train and test splits

    Raises:
        ValueError: If no audio files are found or dataset is too small for training
    """
    audio_files = []
    labels = []

    for category in ["water", "resident", "transient", "humpback", "vessel", "jingle", "human", "bird", "hum"]:
        category_dir = data_dir / category
        if not category_dir.exists():
            print(f"Warning: {category_dir} does not exist")
            continue

        # Get all wav files in this category.
        wav_files = list(category_dir.glob("**/*.wav"))
        print(f"Found {len(wav_files)} files for {category}")

        for wav_file in wav_files:
            # Map labels based on num_classes.
            if num_classes == 2:
                # Binary: map resident/transient/humpback to "whale" (1), everything else to "other" (0).
                label = LABEL2ID["whale"] if category in WHALE_CLASS_NAMES else LABEL2ID["other"]
            else:
                # Multi-class: use original category.
                label = LABEL2ID[category]

            audio_files.append(str(wav_file))
            labels.append(label)

    # Validate that dataset is not empty.
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {data_dir}. Please ensure wav files exist in category subdirectories.")

    # Check label distribution for stratification.
    label_counts = Counter(labels)
    min_samples_per_label = min(label_counts.values()) if label_counts else 0

    print(f"Total samples: {len(audio_files)}")
    print(f"Label distribution: {dict(label_counts)}")

    # Create dataset.
    dataset = Dataset.from_dict({
        "audio": audio_files,
        "label": labels
    })

    # Cast audio column to Audio type.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Cast label column to ClassLabel type for stratification support.
    dataset = dataset.cast_column("label", ClassLabel(names=list(ID2LABEL.values())))

    # Split into train/test.
    # Use stratification only if each label has at least 2 samples and total > 1.
    if len(audio_files) < 2:
        raise ValueError(f"Dataset too small for train/test split: only {len(audio_files)} sample(s) found. Need at least 2 samples.")

    num_unique_labels = len(label_counts)
    estimated_test_size = round(TRAIN_TEST_SPLIT_TEST_SIZE * len(audio_files))
    can_stratify = can_stratify_split(
        num_unique_labels, min_samples_per_label, estimated_test_size
    )

    if can_stratify:
        # Fall back if the split still raises (e.g. a rounding edge case).
        try:
            print("Using stratified train/test split (80/20)")
            dataset = dataset.train_test_split(
                test_size=TRAIN_TEST_SPLIT_TEST_SIZE, seed=42, stratify_by_column="label"
            )
        except ValueError as exc:
            print(f"Warning: stratified split failed ({exc}). Using non-stratified split.")
            dataset = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT_TEST_SIZE, seed=42)
    else:
        # Cannot use stratification - use random split.
        if num_unique_labels == 1:
            print("Warning: Dataset has only 1 unique label. Using non-stratified split.")
        elif min_samples_per_label < 2:
            print(f"Warning: Some labels have fewer than 2 samples (min={min_samples_per_label}). Using non-stratified split.")
        else:
            print(
                f"Warning: Test split (~{estimated_test_size} samples) is smaller than the "
                f"number of classes ({num_unique_labels}). Using non-stratified split."
            )
        dataset = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT_TEST_SIZE, seed=42)

    return dataset


def preprocess_function(examples: dict, feature_extractor: FeatureExtractorProtocol, max_duration: float = 3.0) -> dict:
    """
    Preprocess audio files for the model.

    Args:
        examples: Batch of examples with audio data
        feature_extractor: HuggingFace audio feature extractor instance
        max_duration: Maximum audio duration in seconds

    Returns:
        Processed inputs for the model (as NumPy arrays for serialization)
    """
    audio_arrays = [x["array"] for x in examples["audio"]]

    # Pad or truncate to max_duration so every training example has the same clip length.
    target_length = int(max_duration * 16000)  # 16kHz sample rate.
    processed_audio = []
    for audio in audio_arrays:
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        processed_audio.append(audio)

    # The audio is already padded/truncated above, so we only need the extractor's
    # feature conversion here.
    inputs = feature_extractor(
        processed_audio,
        sampling_rate=16000,
        padding=True,
    )

    # Ensure input_values is a NumPy array (feature_extractor returns this by default).
    # Convert to list of arrays for proper serialization in datasets library.
    inputs["labels"] = examples["label"]

    return inputs


def get_preprocessing_workers(dataset: DatasetDict, requested_workers: int) -> int:
    """Determine a safe number of dataset preprocessing workers.

    Args:
        dataset: Dataset splits that will be preprocessed.
        requested_workers: User-requested worker count.

    Returns:
        Effective worker count capped by the smallest dataset split.
    """
    if requested_workers < 1:
        raise ValueError(f"preprocessing_workers must be at least 1, got {requested_workers}")

    split_sizes = [len(split_dataset) for split_dataset in dataset.values()]
    if not split_sizes:
        return 1

    return max(1, min(requested_workers, min(split_sizes)))


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    Compute evaluation metrics with per-class breakdown.

    Args:
        eval_pred: Predictions and labels

    Returns:
        Dictionary of metrics
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    # Overall metrics.
    accuracy = ACCURACY_METRIC.compute(predictions=predictions, references=labels)
    precision = PRECISION_METRIC.compute(predictions=predictions, references=labels, average="weighted")
    recall = RECALL_METRIC.compute(predictions=predictions, references=labels, average="weighted")
    f1 = F1_METRIC.compute(predictions=predictions, references=labels, average="weighted")

    # Per-class metrics with explicit labels to ensure alignment with ID2LABEL ordering.
    all_labels = list(ID2LABEL.keys())
    precision_per_class = PRECISION_METRIC.compute(predictions=predictions, references=labels, average=None, labels=all_labels)
    recall_per_class = RECALL_METRIC.compute(predictions=predictions, references=labels, average=None, labels=all_labels)
    f1_per_class = F1_METRIC.compute(predictions=predictions, references=labels, average=None, labels=all_labels)

    # Confusion matrix analysis.
    print("\n" + "="*60)
    print("DETAILED EVALUATION METRICS")
    print("="*60)
    print("Dataset: trainer test split from output/wav (80/20 split of training samples).")

    # Class distribution in predictions vs ground truth.
    print("\nClass Distribution:")
    for class_id, class_name in ID2LABEL.items():
        true_count = np.sum(labels == class_id)
        pred_count = np.sum(predictions == class_id)
        print(f"  {class_name:12s} - True: {true_count:3d}, Predicted: {pred_count:3d}")

    # Per-class performance.
    print("\nPer-Class Performance:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)
    for class_id, class_name in ID2LABEL.items():
        prec = precision_per_class['precision'][class_id] if len(precision_per_class['precision']) > class_id else 0
        rec = recall_per_class['recall'][class_id] if len(recall_per_class['recall']) > class_id else 0
        f1_score = f1_per_class['f1'][class_id] if len(f1_per_class['f1']) > class_id else 0
        print(f"{class_name:<12} {prec:<12.3f} {rec:<12.3f} {f1_score:<12.3f}")

    # Confusion matrix.
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>12}", end="")
    for class_name in ID2LABEL.values():
        print(f"{class_name[:8]:>10}", end="")
    print()

    for true_class_id, true_class_name in ID2LABEL.items():
        print(f"{true_class_name:>12}", end="")
        for pred_class_id in range(len(ID2LABEL)):
            count = np.sum((labels == true_class_id) & (predictions == pred_class_id))
            print(f"{count:>10}", end="")
        print()

    print("="*60 + "\n")

    # Optimize model selection for whale classes by using macro F1 over
    # humpback, resident, and transient when those labels are present.
    whale_class_ids = sorted(class_id for class_id, class_name in ID2LABEL.items() if class_name in WHALE_CLASS_NAMES)
    if whale_class_ids:
        f1_whale = F1_METRIC.compute(
            predictions=predictions,
            references=labels,
            average="macro",
            labels=whale_class_ids,
        )
        f1_for_training = f1_whale["f1"]
    else:
        f1_for_training = f1["f1"]

    # Return metrics for training logs.
    metrics = {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1_for_training,
    }

    # Add per-class F1 to tracking.
    for class_id, class_name in ID2LABEL.items():
        if len(f1_per_class['f1']) > class_id:
            metrics[f"f1_{class_name}"] = f1_per_class['f1'][class_id]

    return metrics


def analyze_dataset(dataset: DatasetDict) -> None:
    """
    Analyze dataset statistics and distribution.

    Args:
        dataset: DatasetDict with train and test splits
    """
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)

    for split_name in ["train", "test"]:
        split_data = dataset[split_name]
        labels = split_data["label"]

        print(f"\n{split_name.upper()} Split ({len(labels)} samples):")

        # Count per class.
        label_counts = Counter(labels)
        for class_id in sorted(label_counts.keys()):
            class_name = ID2LABEL[class_id]
            count = label_counts[class_id]
            percentage = 100 * count / len(labels)
            print(f"  {class_name:12s}: {count:4d} samples ({percentage:5.1f}%)")

        # Check for severe imbalance.
        if len(label_counts) > 0:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
            if imbalance_ratio > 10:
                print("  WARNING: Severe class imbalance detected!")

    print("="*60 + "\n")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train PODS-AI audio classification model for orca calls"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        choices=[2, 9],
        default=9,
        help="Number of classes: 2 for binary (other vs whale), 9 for multi-class including \"hum\" (default: 9)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="output/wav",
        help="Path to wav files directory (default: output/wav)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model/podsai",
        help="Directory to save the trained model (default: model/podsai)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MIT/ast-finetuned-audioset-10-10-0.4593",
        help=(
            "Base model to fine-tune "
            "(default: MIT/ast-finetuned-audioset-10-10-0.4593)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)",
    )
    parser.add_argument(
        "--preprocessing_workers",
        type=int,
        default=max(1, min(DEFAULT_MAX_PREPROCESSING_WORKERS, os.cpu_count() or 1)),
        help="Number of parallel workers for AST feature preprocessing (default: min of 8 or available CPU cores)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="orca-call-detector",
        help="HuggingFace Hub model ID (default: orca-call-detector)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to a specific checkpoint to resume training from",
    )
    args = parser.parse_args()

    # Set up label mappings based on num_classes.
    setup_label_mappings(args.num_classes)

    # Set up paths.
    data_dir = REPO_ROOT / args.data_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_dir}...")
    dataset = load_audio_dataset(data_dir, args.num_classes)
    print(f"Dataset: {dataset}")

    # Analyze dataset distribution.
    analyze_dataset(dataset)

    # Load feature extractor and model.
    print(f"Loading feature extractor and model: {args.model_name}")

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    except Exception as e:
        error_msg = f"Error loading feature extractor from {args.model_name}: {type(e).__name__}: {e}"
        print(error_msg)
        print("Please ensure the model name is correct and you have internet connectivity.")
        raise RuntimeError(error_msg) from e

    try:
        model = AutoModelForAudioClassification.from_pretrained(
            args.model_name,
            num_labels=len(LABEL2ID),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,
        )
    except Exception as e:
        error_msg = f"Error loading model from {args.model_name}: {type(e).__name__}: {e}"
        print(error_msg)
        print("Please ensure the model name is correct and you have internet connectivity.")
        raise RuntimeError(error_msg) from e

    # Preprocess dataset.
    preprocessing_workers = get_preprocessing_workers(dataset, args.preprocessing_workers)
    print(f"Preprocessing dataset with {preprocessing_workers} worker(s)...")
    map_kwargs = {
        "batched": True,
        "remove_columns": ["audio"],
    }
    if preprocessing_workers > 1:
        map_kwargs["num_proc"] = preprocessing_workers
    dataset = dataset.map(
        partial(preprocess_function, feature_extractor=feature_extractor),
        **map_kwargs,
    )

    # Training arguments.
    # Compute warmup steps equivalent to warmup_ratio=0.1 (warmup_ratio was removed in transformers 5.15.0).
    steps_per_epoch = math.ceil(len(dataset["train"]) / args.batch_size)
    warmup_steps = int(steps_per_epoch * args.epochs * 0.1)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=CHECKPOINT_SAVE_LIMIT,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=args.push_to_hub,
        hub_strategy="all_checkpoints" if args.push_to_hub else "end",
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    # Create trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    # Resume from checkpoint if provided.
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        # Train.
        print("Starting training...")
        trainer.train()

    # Evaluate.
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

    # Save model and feature extractor.
    print(f"Saving model to {output_dir}...")
    trainer.save_model(str(output_dir))
    feature_extractor.save_pretrained(str(output_dir))

    print("Training complete!")

    if args.push_to_hub:
        # The Trainer already pushed the model weights; also push the feature extractor
        # so that inference code can call AutoFeatureExtractor.from_pretrained()
        # on the Hub model ID.
        print(f"Pushing feature extractor to HuggingFace Hub: {args.hub_model_id}...")
        feature_extractor.push_to_hub(args.hub_model_id)
        print(f"Model pushed to HuggingFace Hub: {args.hub_model_id}")


if __name__ == "__main__":
    main()
