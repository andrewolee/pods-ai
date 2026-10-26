#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for train_podsai_model.py metric selection and Hub settings."""

import importlib
import sys
import types

import numpy as np
import pytest


def _import_stubbed_train_module(monkeypatch):
    """Import train_podsai_model with lightweight dependency stubs.

    Returns:
        The imported train_podsai_model module with stubbed dependencies.
    """
    fake_datasets = types.ModuleType("datasets")
    fake_datasets_config = types.ModuleType("datasets.config")
    fake_datasets_config.AUDIO_BACKENDS_USE_TORCH = False
    fake_datasets_config.AUDIOCODEC_DEFAULT_DECODER = "soundfile"
    fake_datasets.config = fake_datasets_config
    fake_datasets.Dataset = object
    fake_datasets.Audio = object
    fake_datasets.DatasetDict = object
    fake_datasets.ClassLabel = object

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoFeatureExtractor = object
    fake_transformers.AutoModelForAudioClassification = object
    fake_transformers.TrainingArguments = object
    fake_transformers.Trainer = object

    class _EvalPrediction:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids

    fake_transformers.EvalPrediction = _EvalPrediction

    fake_evaluate = types.ModuleType("evaluate")
    fake_torch = types.ModuleType("torch")

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    fake_torch.cuda = _FakeCuda()

    class _DummyMetric:
        def __init__(self, key):
            self._key = key

        def compute(self, **_kwargs):
            return {self._key: 0.0}

    fake_evaluate.load = lambda name: _DummyMetric(name)

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "datasets.config", fake_datasets_config)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "evaluate", fake_evaluate)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sys.modules.pop("train_podsai_model", None)
    return importlib.import_module("train_podsai_model")


class _FakeMetric:
    """Lightweight metric helper matching sklearn-style averaging/label behavior.

    This supports per-class output and weighted/macro averaging for accuracy,
    precision, recall, and F1 so compute_metrics can be validated without
    external metric dependencies.

    The ``compute`` method accepts predictions/references and optional averaging
    configuration, and returns ``{metric_name: value}``.

    Args:
        metric_name: Metric key name ("accuracy", "precision", "recall", or "f1").
    """

    def __init__(self, metric_name: str):
        self.metric_name = metric_name

    def compute(self, predictions, references, average=None, labels=None):
        preds = np.asarray(predictions)
        refs = np.asarray(references)
        if labels is None:
            all_labels = sorted(set(refs.tolist()) | set(preds.tolist()))
        else:
            all_labels = list(labels)

        if self.metric_name == "accuracy":
            return {"accuracy": float(np.mean(preds == refs))}

        supports = np.array([np.sum(refs == class_id) for class_id in all_labels], dtype=float)
        tps = np.array([np.sum((preds == class_id) & (refs == class_id)) for class_id in all_labels], dtype=float)
        fps = np.array([np.sum((preds == class_id) & (refs != class_id)) for class_id in all_labels], dtype=float)
        fns = np.array([np.sum((preds != class_id) & (refs == class_id)) for class_id in all_labels], dtype=float)

        precision = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps + fps) != 0)
        recall = np.divide(tps, tps + fns, out=np.zeros_like(tps), where=(tps + fns) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)

        per_class = {"precision": precision, "recall": recall, "f1": f1}[self.metric_name]
        if average is None:
            return {self.metric_name: per_class}
        if average == "weighted":
            total = np.sum(supports)
            weighted = float(np.sum(per_class * supports) / total) if total else 0.0
            return {self.metric_name: weighted}
        if average == "macro":
            macro = float(np.mean(per_class)) if len(per_class) else 0.0
            return {self.metric_name: macro}
        raise ValueError(f"Unsupported average: {average}")


def _patch_metrics(module):
    """Patch module-level metric objects with deterministic fake metrics.

    Args:
        module: train_podsai_model module instance to patch.
    """
    module.ACCURACY_METRIC = _FakeMetric("accuracy")
    module.PRECISION_METRIC = _FakeMetric("precision")
    module.RECALL_METRIC = _FakeMetric("recall")
    module.F1_METRIC = _FakeMetric("f1")


def test_default_training_model_uses_spectrogram_checkpoint(monkeypatch, tmp_path):
    """Training should default to the spectrogram-based AST checkpoint."""
    module = _import_stubbed_train_module(monkeypatch)

    captured = {}

    class _FakeFeatureExtractor:
        @classmethod
        def from_pretrained(cls, model_name):
            captured["feature_extractor_model_name"] = model_name
            return cls()

        def save_pretrained(self, _output_dir):
            return None

        def __call__(self, processed_audio, **_kwargs):
            return {"input_values": processed_audio}

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["model_kwargs"] = kwargs
            return cls()

    class _FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs

    class _FakeTrainer:
        def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.compute_metrics = compute_metrics

        def train(self, resume_from_checkpoint=None):
            return resume_from_checkpoint

        def evaluate(self):
            return {"f1": 1.0}

        def save_model(self, _output_dir):
            return None

    class _FakeDatasetDict(dict):
        def map(self, _func, batched, remove_columns):
            assert batched is True
            assert remove_columns == ["audio"]
            return self

    fake_dataset = _FakeDatasetDict(train=[{"label": 0}], test=[{"label": 0}])

    monkeypatch.setattr(module, "setup_label_mappings", lambda _num_classes: None)
    monkeypatch.setattr(module, "load_audio_dataset", lambda _data_dir, _num_classes: fake_dataset)
    monkeypatch.setattr(module, "analyze_dataset", lambda _dataset: None)
    monkeypatch.setattr(module, "AutoFeatureExtractor", _FakeFeatureExtractor)
    monkeypatch.setattr(module, "AutoModelForAudioClassification", _FakeModel)
    monkeypatch.setattr(module, "TrainingArguments", _FakeTrainingArguments)
    monkeypatch.setattr(module, "Trainer", _FakeTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_podsai_model.py",
            "--output_dir",
            str(tmp_path / "model"),
        ],
    )

    module.main()

    assert captured["feature_extractor_model_name"] == "MIT/ast-finetuned-audioset-10-10-0.4593"
    assert captured["model_name"] == "MIT/ast-finetuned-audioset-10-10-0.4593"
    assert captured["model_kwargs"]["ignore_mismatched_sizes"] is True


def test_preprocess_function_normalizes_audio_length_before_extractor(monkeypatch):
    """preprocess_function should pad/truncate clips before calling the extractor."""
    module = _import_stubbed_train_module(monkeypatch)

    captured = {}

    class _CapturingFeatureExtractor:
        def __call__(self, processed_audio, **kwargs):
            captured["processed_audio"] = processed_audio
            captured["kwargs"] = kwargs
            return {"input_values": processed_audio}

    examples = {
        "audio": [
            {"array": np.ones(16000, dtype=np.float32)},
            {"array": np.ones(64000, dtype=np.float32)},
        ],
        "label": [0, 1],
    }

    result = module.preprocess_function(examples, _CapturingFeatureExtractor(), max_duration=3.0)

    assert all(len(audio) == 48000 for audio in captured["processed_audio"])
    assert captured["kwargs"] == {"sampling_rate": 16000, "padding": True}
    assert result["labels"] == [0, 1]


def test_can_stratify_split_rejects_test_size_smaller_than_class_count(monkeypatch):
    """A 9-class dataset whose 20% test split is smaller than 9 must not attempt stratification.

    Regression test for the exact scenario flagged in PR #566 review: an 8-class
    dataset with 37 samples plus 2 new "hum" samples (39 total, 9 classes) has an
    estimated test split of round(0.2 * 39) == 8, which is smaller than the
    9-class count and would make train_test_split's stratify_by_column raise.
    """
    module = _import_stubbed_train_module(monkeypatch)
    assert module.can_stratify_split(num_unique_labels=9, min_samples_per_label=2, estimated_test_size=8) is False


def test_can_stratify_split_allows_test_size_equal_to_class_count(monkeypatch):
    """A test split exactly as large as the number of classes should allow stratification."""
    module = _import_stubbed_train_module(monkeypatch)
    assert module.can_stratify_split(num_unique_labels=9, min_samples_per_label=2, estimated_test_size=9) is True


def test_can_stratify_split_rejects_single_label(monkeypatch):
    """Stratification is impossible with only one unique label, regardless of sample counts."""
    module = _import_stubbed_train_module(monkeypatch)
    assert module.can_stratify_split(num_unique_labels=1, min_samples_per_label=100, estimated_test_size=100) is False


def test_can_stratify_split_rejects_label_with_fewer_than_two_samples(monkeypatch):
    """Stratification is impossible if any label has fewer than 2 samples."""
    module = _import_stubbed_train_module(monkeypatch)
    assert module.can_stratify_split(num_unique_labels=5, min_samples_per_label=1, estimated_test_size=50) is False


def test_preprocessing_workers_are_capped_by_smallest_split(monkeypatch, tmp_path):
    """Dataset preprocessing should not request more workers than the smallest split size."""
    module = _import_stubbed_train_module(monkeypatch)

    captured = {}

    class _FakeFeatureExtractor:
        @classmethod
        def from_pretrained(cls, _model_name):
            return cls()

        def save_pretrained(self, _output_dir):
            return None

        def __call__(self, processed_audio, **_kwargs):
            return {"input_values": processed_audio}

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class _FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs

    class _FakeTrainer:
        def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.compute_metrics = compute_metrics

        def train(self, resume_from_checkpoint=None):
            return resume_from_checkpoint

        def evaluate(self):
            return {"f1": 1.0}

        def save_model(self, _output_dir):
            return None

    class _FakeDatasetDict(dict):
        def map(self, func, **kwargs):
            assert callable(func)
            captured["map_kwargs"] = kwargs
            return self

    fake_dataset = _FakeDatasetDict(
        train=[{"label": 0}] * 8,
        test=[{"label": 0}] * 2,
    )

    monkeypatch.setattr(module, "setup_label_mappings", lambda _num_classes: None)
    monkeypatch.setattr(module, "load_audio_dataset", lambda _data_dir, _num_classes: fake_dataset)
    monkeypatch.setattr(module, "analyze_dataset", lambda _dataset: None)
    monkeypatch.setattr(module, "AutoFeatureExtractor", _FakeFeatureExtractor)
    monkeypatch.setattr(module, "AutoModelForAudioClassification", _FakeModel)
    monkeypatch.setattr(module, "TrainingArguments", _FakeTrainingArguments)
    monkeypatch.setattr(module, "Trainer", _FakeTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_podsai_model.py",
            "--output_dir",
            str(tmp_path / "model"),
            "--preprocessing_workers",
            "4",
        ],
    )

    module.main()

    assert captured["map_kwargs"] == {
        "batched": True,
        "remove_columns": ["audio"],
        "num_proc": 2,
    }
    assert captured["training_args"]["fp16"] is False


def test_training_enables_fp16_when_cuda_is_available(monkeypatch, tmp_path):
    """Training arguments should enable fp16 automatically when CUDA is available."""
    module = _import_stubbed_train_module(monkeypatch)

    captured = {}

    class _FakeFeatureExtractor:
        @classmethod
        def from_pretrained(cls, _model_name):
            return cls()

        def save_pretrained(self, _output_dir):
            return None

        def __call__(self, processed_audio, **_kwargs):
            return {"input_values": processed_audio}

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class _FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs

    class _FakeTrainer:
        def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.compute_metrics = compute_metrics

        def train(self, resume_from_checkpoint=None):
            return resume_from_checkpoint

        def evaluate(self):
            return {"f1": 1.0}

        def save_model(self, _output_dir):
            return None

    class _FakeDatasetDict(dict):
        def map(self, func, batched, remove_columns):
            assert callable(func)
            captured["map_kwargs"] = {
                "batched": batched,
                "remove_columns": remove_columns,
            }
            assert batched is True
            assert remove_columns == ["audio"]
            return self

    fake_dataset = _FakeDatasetDict(train=[{"label": 0}], test=[{"label": 0}])

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "setup_label_mappings", lambda _num_classes: None)
    monkeypatch.setattr(module, "load_audio_dataset", lambda _data_dir, _num_classes: fake_dataset)
    monkeypatch.setattr(module, "analyze_dataset", lambda _dataset: None)
    monkeypatch.setattr(module, "AutoFeatureExtractor", _FakeFeatureExtractor)
    monkeypatch.setattr(module, "AutoModelForAudioClassification", _FakeModel)
    monkeypatch.setattr(module, "TrainingArguments", _FakeTrainingArguments)
    monkeypatch.setattr(module, "Trainer", _FakeTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_podsai_model.py",
            "--output_dir",
            str(tmp_path / "model"),
        ],
    )

    module.main()

    assert captured["training_args"]["fp16"] is True
    assert "num_proc" not in captured.get("map_kwargs", {})


def test_push_to_hub_uses_last_six_checkpoints(monkeypatch, tmp_path):
    """Hub uploads should retain and publish the last 6 epoch checkpoints."""
    module = _import_stubbed_train_module(monkeypatch)

    captured_training_args = {}

    class _FakeFeatureExtractor:
        @classmethod
        def from_pretrained(cls, _model_name):
            return cls()

        def save_pretrained(self, _output_dir):
            return None

        def push_to_hub(self, _hub_model_id):
            return None

        def __call__(self, processed_audio, **_kwargs):
            return {"input_values": processed_audio}

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class _FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured_training_args.update(kwargs)

    class _FakeTrainer:
        def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.compute_metrics = compute_metrics

        def train(self, resume_from_checkpoint=None):
            return resume_from_checkpoint

        def evaluate(self):
            return {"f1": 1.0}

        def save_model(self, _output_dir):
            return None

    class _FakeDatasetDict(dict):
        def map(self, _func, batched, remove_columns):
            assert batched is True
            assert remove_columns == ["audio"]
            return self

    fake_dataset = _FakeDatasetDict(train=[{"label": 0}], test=[{"label": 0}])

    monkeypatch.setattr(module, "setup_label_mappings", lambda _num_classes: None)
    monkeypatch.setattr(module, "load_audio_dataset", lambda _data_dir, _num_classes: fake_dataset)
    monkeypatch.setattr(module, "analyze_dataset", lambda _dataset: None)
    monkeypatch.setattr(module, "AutoFeatureExtractor", _FakeFeatureExtractor)
    monkeypatch.setattr(module, "AutoModelForAudioClassification", _FakeModel)
    monkeypatch.setattr(module, "TrainingArguments", _FakeTrainingArguments)
    monkeypatch.setattr(module, "Trainer", _FakeTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_podsai_model.py",
            "--output_dir",
            str(tmp_path / "model"),
            "--push_to_hub",
            "--hub_model_id",
            "davethaler/whale-call-detector",
        ],
    )

    module.main()

    assert captured_training_args["save_strategy"] == "epoch"
    assert captured_training_args["save_total_limit"] == 6
    assert captured_training_args["hub_strategy"] == "all_checkpoints"
    assert captured_training_args["push_to_hub"] is True
    assert captured_training_args["hub_model_id"] == "davethaler/whale-call-detector"


def test_whale_f1_computed_from_whale_classes_only(monkeypatch):
    """f1 should be macro F1 over resident/transient/humpback in multiclass mode."""
    module = _import_stubbed_train_module(monkeypatch)
    _patch_metrics(module)
    module.ID2LABEL = {
        0: "water",
        1: "resident",
        2: "transient",
        3: "humpback",
        4: "vessel",
        5: "jingle",
        6: "human",
        7: "bird",
    }

    labels = np.array([1, 2, 3, 0, 4, 5, 6, 7])
    predictions = np.array([1, 2, 3, 4, 5, 6, 0, 7])
    logits = np.eye(8)[predictions]

    eval_pred = module.EvalPrediction(predictions=logits, label_ids=labels)
    metrics = module.compute_metrics(eval_pred)

    assert metrics["f1"] == 1.0
    assert metrics["f1_resident"] == 1.0
    assert metrics["f1_transient"] == 1.0
    assert metrics["f1_humpback"] == 1.0
    assert metrics["f1_bird"] == 1.0
    assert metrics["f1_water"] == 0.0
    assert metrics["f1_vessel"] == 0.0


def test_compute_metrics_prints_dataset_context(monkeypatch, capsys):
    """compute_metrics should print which evaluation dataset/split is being reported."""
    module = _import_stubbed_train_module(monkeypatch)
    _patch_metrics(module)
    module.ID2LABEL = {
        0: "water",
        1: "resident",
        2: "transient",
        3: "humpback",
        4: "vessel",
        5: "jingle",
        6: "human",
        7: "bird",
    }

    labels = np.array([1, 2, 3, 0, 4, 5, 6, 7])
    predictions = np.array([1, 2, 3, 0, 4, 5, 6, 7])
    logits = np.eye(8)[predictions]

    eval_pred = module.EvalPrediction(predictions=logits, label_ids=labels)
    module.compute_metrics(eval_pred)

    captured = capsys.readouterr().out
    assert "trainer test split from output/wav (80/20 split of training samples)" in captured


def test_whale_f1_reflects_mixed_whale_predictions(monkeypatch):
    """f1 should drop when whale-class predictions include errors."""
    module = _import_stubbed_train_module(monkeypatch)
    _patch_metrics(module)
    module.ID2LABEL = {
        0: "water",
        1: "resident",
        2: "transient",
        3: "humpback",
        4: "vessel",
        5: "jingle",
        6: "human",
        7: "bird",
    }

    labels = np.array([1, 2, 3, 0, 4, 5, 6, 7])
    predictions = np.array([1, 1, 2, 4, 5, 6, 0, 7])
    logits = np.eye(8)[predictions]

    eval_pred = module.EvalPrediction(predictions=logits, label_ids=labels)
    metrics = module.compute_metrics(eval_pred)

    # Whale-class F1s are resident=2/3, transient=0, humpback=0 => macro = 2/9.
    assert metrics["f1"] == pytest.approx(2.0 / 9.0)


def test_f1_falls_back_to_weighted_without_whale_classes(monkeypatch):
    """f1 should remain the default weighted F1 when whale class labels are not present."""
    module = _import_stubbed_train_module(monkeypatch)
    _patch_metrics(module)
    module.ID2LABEL = {0: "other", 1: "whale"}

    labels = np.array([1, 1, 0, 0])
    predictions = np.array([1, 0, 0, 0])
    logits = np.eye(2)[predictions]

    eval_pred = module.EvalPrediction(predictions=logits, label_ids=labels)
    metrics = module.compute_metrics(eval_pred)

    # Weighted F1 for this case:
    # class 0 f1=0.8 (support=2), class 1 f1=2/3 (support=2) => 0.7333...
    assert metrics["f1"] == pytest.approx(11.0 / 15.0)


def test_f1_fallback_supports_multiclass_non_whale_labels(monkeypatch):
    """Fallback weighted F1 should work with non-whale multiclass mappings."""
    module = _import_stubbed_train_module(monkeypatch)
    _patch_metrics(module)
    module.ID2LABEL = {0: "water", 1: "vessel", 2: "human", 3: "jingle", 4: "bird"}

    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    predictions = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 0])
    logits = np.eye(5)[predictions]

    eval_pred = module.EvalPrediction(predictions=logits, label_ids=labels)
    metrics = module.compute_metrics(eval_pred)

    # Symmetric confusion gives per-class F1=0.5 for all classes => weighted F1=0.5.
    assert metrics["f1"] == pytest.approx(0.5)


def test_warmup_steps_not_warmup_ratio_passed_to_training_arguments(monkeypatch, tmp_path):
    """TrainingArguments must receive warmup_steps, not warmup_ratio (removed in transformers 5.15.0).

    The computed value should equal ~10% of total training steps, where total
    steps = math.ceil(train_size / batch_size) * epochs.
    """
    import math

    module = _import_stubbed_train_module(monkeypatch)

    captured_training_args = {}

    class _FakeFeatureExtractor:
        @classmethod
        def from_pretrained(cls, _model_name):
            return cls()

        def save_pretrained(self, _output_dir):
            return None

        def push_to_hub(self, _hub_model_id):
            return None

        def __call__(self, processed_audio, **_kwargs):
            return {"input_values": processed_audio}

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class _FakeTrainingArgumentsStrict:
        """Mimics transformers>=5.15.0: raises TypeError on warmup_ratio."""

        def __init__(self, **kwargs):
            if "warmup_ratio" in kwargs:
                raise TypeError(
                    "TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'"
                )
            captured_training_args.update(kwargs)

    class _FakeTrainer:
        def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics):
            self.args = args

        def train(self, resume_from_checkpoint=None):
            return None

        def evaluate(self):
            return {"f1": 1.0}

        def save_model(self, _output_dir):
            return None

    # Use a dataset size and batch size that exercises the math.ceil path.
    train_size = 25
    batch_size = 8
    epochs = 5

    class _FakeDatasetDict(dict):
        def map(self, _func, batched, remove_columns):
            return self

    fake_dataset = _FakeDatasetDict(
        train=[{"label": 0}] * train_size,
        test=[{"label": 0}],
    )

    monkeypatch.setattr(module, "setup_label_mappings", lambda _num_classes: None)
    monkeypatch.setattr(module, "load_audio_dataset", lambda _data_dir, _num_classes: fake_dataset)
    monkeypatch.setattr(module, "analyze_dataset", lambda _dataset: None)
    monkeypatch.setattr(module, "AutoFeatureExtractor", _FakeFeatureExtractor)
    monkeypatch.setattr(module, "AutoModelForAudioClassification", _FakeModel)
    monkeypatch.setattr(module, "TrainingArguments", _FakeTrainingArgumentsStrict)
    monkeypatch.setattr(module, "Trainer", _FakeTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_podsai_model.py",
            "--output_dir",
            str(tmp_path / "model"),
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
        ],
    )

    # This call must not raise TypeError about warmup_ratio.
    module.main()

    assert "warmup_ratio" not in captured_training_args, (
        "warmup_ratio was passed to TrainingArguments; it was removed in transformers 5.15.0"
    )
    assert "warmup_steps" in captured_training_args

    expected_steps = int(math.ceil(train_size / batch_size) * epochs * 0.1)
    assert captured_training_args["warmup_steps"] == expected_steps


def test_metric_loader_falls_back_to_sklearn_when_evaluate_metric_missing(monkeypatch):
    """Metric loading should gracefully fall back when evaluate modules are unavailable."""
    module = _import_stubbed_train_module(monkeypatch)

    def _raise_not_found(_name):
        raise FileNotFoundError("missing metric module")

    monkeypatch.setattr(module.evaluate, "load", _raise_not_found)

    accuracy_metric = module._load_metric("accuracy")
    f1_metric = module._load_metric("f1")

    accuracy = accuracy_metric.compute(predictions=[1, 0, 1], references=[1, 1, 1])
    per_class_f1 = f1_metric.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 1], average=None, labels=[0, 1])

    assert accuracy["accuracy"] == pytest.approx(2.0 / 3.0)
    assert np.allclose(per_class_f1["f1"], np.array([0.5, 0.5]))
