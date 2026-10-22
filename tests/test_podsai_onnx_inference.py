# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from podsai_onnx_inference import PodsAIOnnxInference


def _inference_for_logits(monkeypatch, logits):
    audio = np.zeros(((len(logits) - 1) * 2 + 3) * 16000)
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(load=lambda *args, **kwargs: (audio, 16000)),
    )

    class FakeSession:
        def run(self, output_names, inputs):
            return [np.asarray(logits)]

    inference = PodsAIOnnxInference.__new__(PodsAIOnnxInference)
    inference.threshold = 0.5
    inference.min_num_positive_calls_threshold = 2
    inference.inference_batch_size = 32
    inference.sess = FakeSession()
    inference._input_name = "input_values"
    inference._input_shape = [None, 3, 2]
    inference.id2label = {0: "water", 1: "resident", 2: "other"}
    inference.label2id = {"water": 0, "resident": 1, "other": 2}
    inference.negative_class_ids = {0, 2}
    inference._compute_input_values = lambda *args: np.zeros(
        (len(logits), 3, 2), dtype=np.float32
    )
    return inference


def test_per_class_probabilities_average_matching_local_confidences(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(load=lambda *args, **kwargs: (np.zeros(5 * 16000), 16000)),
    )

    class FakeSession:
        def run(self, output_names, inputs):
            return [np.array([[4.0, 3.0, 0.0], [3.0, 4.0, 0.0]])]

    inference = PodsAIOnnxInference.__new__(PodsAIOnnxInference)
    inference.threshold = 0.5
    inference.min_num_positive_calls_threshold = 2
    inference.inference_batch_size = 8
    inference.sess = FakeSession()
    inference._input_name = "input_values"
    inference._input_shape = [None, 3, 2]
    inference.id2label = {0: "water", 1: "resident", 2: "other"}
    inference.label2id = {"water": 0, "resident": 1, "other": 2}
    inference.negative_class_ids = {0, 2}
    inference._compute_input_values = lambda *args: np.zeros((2, 3, 2), dtype=np.float32)

    result = inference.predict("sample.wav")

    assert result["local_predictions"] == [0, 1]
    assert result["per_class_probabilities"] == {
        "water": result["local_confidences"][0],
        "resident": result["local_confidences"][1],
        "other": 0.0,
    }


def test_rejects_input_dimensions_unsupported_by_export(monkeypatch):
    inference = _inference_for_logits(monkeypatch, np.log([[0.1, 0.8, 0.1]] * 2))
    inference._compute_input_values = lambda *args: np.zeros((2, 4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"expected \(3, 2\), got \(4, 2\)"):
        inference.predict("sample.wav")


def test_preprocessing_uses_fixed_onnx_frame_dimension():
    inference = PodsAIOnnxInference.__new__(PodsAIOnnxInference)
    inference._input_shape = [None, 1024, 128]

    assert inference._get_target_frames(seg_frames=300, max_length=1024) == 1024


def test_preprocessing_rejects_segments_larger_than_fixed_onnx_frame_dimension():
    inference = PodsAIOnnxInference.__new__(PodsAIOnnxInference)
    inference._input_shape = [None, 300, 128]

    with pytest.raises(ValueError, match="segment has 301 frames, model expects 300"):
        inference._get_target_frames(seg_frames=301, max_length=1024)


def test_preprocessing_uses_feature_extractor_length_for_dynamic_onnx_dimension():
    inference = PodsAIOnnxInference.__new__(PodsAIOnnxInference)
    inference._input_shape = [None, "frames", 128]

    assert inference._get_target_frames(seg_frames=300, max_length=1024) == 300


def test_global_prediction_labels_exclude_positive_classes_below_threshold(monkeypatch):
    logits = np.log([[0.35, 0.4, 0.25]] * 2)
    inference = _inference_for_logits(monkeypatch, logits)

    result = inference.predict("sample.wav")

    assert result["global_prediction_labels"] == []
    assert result["global_prediction_label"] == "other"


def test_global_prediction_labels_require_enough_non_adjacent_events(monkeypatch):
    logits = np.log(
        [[0.1, 0.8, 0.1]] * 2 + [[0.8, 0.1, 0.1]] * 9
    )
    inference = _inference_for_logits(monkeypatch, logits)

    result = inference.predict("sample.wav")

    assert result["global_prediction_labels"] == ["water"]
    assert result["global_prediction_label"] == "water"


def test_feature_extractor_and_config_use_pinned_authenticated_revision(
    monkeypatch,
):
    from unittest.mock import Mock

    feature_extractor_loader = Mock(return_value=object())
    config_loader = Mock(return_value=SimpleNamespace(
        id2label={0: "other"},
        label2id={"other": 0},
    ))
    fake_transformers = SimpleNamespace(
        AutoFeatureExtractor=SimpleNamespace(
            from_pretrained=feature_extractor_loader
        ),
        AutoConfig=SimpleNamespace(from_pretrained=config_loader),
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input_values", shape=[None, 3, 2])]

    fake_onnxruntime = SimpleNamespace(
        InferenceSession=lambda *_args, **_kwargs: FakeSession(),
        SessionOptions=lambda: SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    revision = "0123456789abcdef"
    PodsAIOnnxInference(
        "model.onnx",
        feature_extractor="example/model",
        model_revision=revision,
    )

    expected_kwargs = {"revision": revision, "token": "test-token"}
    feature_extractor_loader.assert_called_once_with(
        "example/model", **expected_kwargs
    )
    config_loader.assert_called_once_with("example/model", **expected_kwargs)


@pytest.mark.parametrize(
    "audio_load",
    [
        lambda: (np.array([], dtype=np.float32), 16000),
        lambda: (_ for _ in ()).throw(OSError("invalid audio")),
    ],
)
def test_empty_or_unreadable_audio_returns_negative_result(
    monkeypatch, audio_load
):
    from unittest.mock import Mock

    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(load=lambda *_args, **_kwargs: audio_load()),
    )
    inference = PodsAIOnnxInference.__new__(PodsAIOnnxInference)
    inference.threshold = 0.5
    inference.min_num_positive_calls_threshold = 2
    inference.id2label = {0: "water", 1: "resident", 2: "other"}
    inference.label2id = {"water": 0, "resident": 1, "other": 2}
    inference.negative_class_ids = {0, 2}
    inference._compute_input_values = Mock(side_effect=AssertionError)

    result = inference.predict("bad.wav")

    assert result["local_predictions"] == []
    assert result["local_confidences"] == []
    assert result["local_probs"] == []
    assert result["global_prediction"] == 2
    assert result["global_prediction_label"] == "other"
    assert result["global_prediction_labels"] == []
    assert result["global_confidence"] == 0.0
    assert result["per_class_probabilities"] == {
        "water": 0.0,
        "resident": 0.0,
        "other": 0.0,
    }


def test_empty_audio_uses_label2id_when_id2label_is_missing(monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(
            load=lambda *_args, **_kwargs: (np.array([], dtype=np.float32), 16000)
        ),
    )
    inference = PodsAIOnnxInference.__new__(PodsAIOnnxInference)
    inference.threshold = 0.5
    inference.min_num_positive_calls_threshold = 2
    inference.id2label = {}
    inference.label2id = {"other": 0}
    inference.negative_class_ids = {0}
    inference._compute_input_values = Mock(side_effect=AssertionError)

    result = inference.predict("empty.wav")

    assert result["global_prediction"] == 0
    assert result["global_prediction_label"] == "other"
