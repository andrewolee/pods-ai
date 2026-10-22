#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for run_inference.py.

Tests cover:
- run_inference() with mocked PODS-AI model
- run_inference() with mocked FastAI model
- Per-class probability output format and values
- CLI argument validation (invalid model type)
- Missing wav file error handling
- Integration tests with real models (if available)
"""

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

# Pinned PODS-AI model revision for integration-test stability.
PODSAI_TEST_MODEL_ID = "davethaler/whale-call-detector"
from model_revisions import LATEST_PODSAI_MODEL_REVISION, PODSAI_WAV2VEC2_MODEL_REVISION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(duration_s: int = 5, sr: int = 16000) -> str:
    """Write a synthetic wav file and return its path (caller must clean up)."""
    samples = np.zeros(duration_s * sr, dtype=np.float32)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, samples, sr)
    f.close()
    return f.name


def _make_fastai_model_mock(global_prediction: int = 1, global_confidence: float = 0.75,
                             num_local: int = 5) -> MagicMock:
    """Return a mock FastAI model whose predict() returns a binary result."""
    mock_model = MagicMock()
    local_confidences = [0.8] * num_local
    mock_model.predict.return_value = {
        "local_predictions": [global_prediction] * num_local,
        "local_confidences": local_confidences,
        "global_prediction": global_prediction,
        "global_confidence": global_confidence,
        "hop_duration": 1.0,
        "segment_duration": 3.0,
    }
    return mock_model


def _make_podsai_model_mock(num_local: int = 10) -> MagicMock:
    """Return a mock PODS-AI model whose predict() returns an 8-class result."""
    mock_model = MagicMock()

    # 8-class label mapping matching the standard schema.
    mock_model.id2label = {
        0: "water",
        1: "resident",
        2: "transient",
        3: "humpback",
        4: "vessel",
        5: "jingle",
        6: "human",
        7: "bird",
    }
    mock_model.label2id = {v: k for k, v in mock_model.id2label.items()}
    mock_model.threshold = 0.5

    # Simulate most windows predicting "resident" (1) with confidence above threshold,
    # and a few predicting "water" (0) with confidence below threshold.
    local_predictions = [1] * (num_local - 2) + [0] * 2
    local_confidences = [0.7] * (num_local - 2) + [0.1] * 2

    # Per-class probabilities: resident should have the highest average
    per_class_probabilities = {
        "water": 0.1,
        "resident": 0.7,
        "transient": 0.0,
        "humpback": 0.0,
        "vessel": 0.0,
        "jingle": 0.0,
        "human": 0.0,
        "bird": 0.0,
    }

    mock_model.predict.return_value = {
        "local_predictions": local_predictions,
        "local_confidences": local_confidences,
        "global_prediction": 1,
        "global_prediction_label": "resident",
        "global_confidence": 0.7,
        "per_class_probabilities": per_class_probabilities,
        "hop_duration": 2.0,
        "segment_duration": 3.0,
    }
    return mock_model


def _make_orcahello_model_mock(global_prediction: int = 1, global_confidence: float = 0.75,
                                num_local: int = 5) -> MagicMock:
    """Return a mock OrcaHello model whose predict() returns a binary result."""
    mock_model = MagicMock()
    local_confidences = [0.8] * num_local
    mock_model.predict.return_value = {
        "local_predictions": [global_prediction] * num_local,
        "local_confidences": local_confidences,
        "global_prediction": global_prediction,
        "global_prediction_label": "resident" if global_prediction else "other",
        "global_confidence": global_confidence,
        "hop_duration": 1.0,
        "segment_duration": 2.0,
    }
    return mock_model


def _resolve_podsai_test_model_path() -> str:
    """Return local PODS-AI model path or a pinned Hub snapshot for integration tests."""
    local_path = Path("model/multiclass")
    if local_path.exists():
        return str(local_path)

    try:
        from huggingface_hub import file_exists as hf_file_exists
        from huggingface_hub import snapshot_download as hf_snapshot_download
        if not hf_file_exists(
            PODSAI_TEST_MODEL_ID,
            "preprocessor_config.json",
            revision=LATEST_PODSAI_MODEL_REVISION,
        ):
            pytest.skip(
                f"Hub model '{PODSAI_TEST_MODEL_ID}' revision "
                f"'{LATEST_PODSAI_MODEL_REVISION}' is missing preprocessor_config.json."
            )
        return hf_snapshot_download(
            repo_id=PODSAI_TEST_MODEL_ID,
            revision=LATEST_PODSAI_MODEL_REVISION,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "training_args.bin",
            ],
        )
    except Exception:
        pytest.skip(
            f"HuggingFace Hub is not reachable; cannot load model "
            f"'{PODSAI_TEST_MODEL_ID}' revision '{LATEST_PODSAI_MODEL_REVISION}'"
        )


def model_inference_helper(label: str, wav_file: str, model_type: str, model_path: str) -> dict:
    """Download a specific testing WAV and run run_inference() against it.

    This helper is used by the integration tests and must be module-level so
    pytest test methods can call it directly.
    """
    from download_wavs import download_testing_file
    from run_inference import run_inference

    output_root = Path("output/testing-wav")
    category_dir = output_root / label
    wav_path = category_dir / wav_file
    print(f"\nProcessing {wav_path}...")
    download_testing_file(wav_path, output_root)
    result = run_inference(str(wav_path), model_type=model_type, model_path=model_path)
    return result



def _verify_fastai_result_structure(result: dict) -> None:
    """Verify FastAI result has expected structure and valid values."""
    assert "probabilities" in result
    assert "global_prediction_label" in result
    assert "global_confidence" in result

    # Verify binary classes.
    assert set(result["probabilities"].keys()) == {"other", "resident"}

    # Verify probabilities sum to 1.0.
    total_prob = sum(result["probabilities"].values())
    assert abs(total_prob - 1.0) < 1e-3

    # Verify all values in valid range.
    for prob in result["probabilities"].values():
        assert 0.0 <= prob <= 1.0

    assert 0.0 <= result["global_confidence"] <= 1.0
    assert result["global_prediction_label"] in {"other", "resident"}


def _verify_podsai_result_structure(result: dict) -> None:
    """Verify PODS-AI result has expected structure and valid values."""

    # Currently the published PODS-AI model only has 7 classes, so accept it too for now.
    # TODO: clean this up once the 8-class model is published.
    old_expected_classes = {"water", "resident", "transient", "humpback", "vessel", "jingle", "human"}
    expected_classes = {"water", "resident", "transient", "humpback", "vessel", "jingle", "human", "bird"}

    assert "probabilities" in result
    assert "global_prediction_label" in result
    assert "global_confidence" in result

    # Verify all 8 classes are present.
    assert set(result["probabilities"].keys()) == expected_classes or set(result["probabilities"].keys()) == old_expected_classes

    # Verify all values in valid range.
    for prob in result["probabilities"].values():
        assert 0.0 <= prob <= 1.0

    assert 0.0 <= result["global_confidence"] <= 1.0
    assert result["global_prediction_label"] in expected_classes


def _print_fastai_result(result: dict, label: str = "") -> None:
    """Print FastAI inference results for debugging."""
    prefix = f"FastAI inference results{f' on {label}' if label else ''}:"
    print(f"\n{prefix}")
    print(f"  Global prediction: {result['global_prediction_label']}")
    print(f"  Global confidence: {result['global_confidence']:.4f}")
    print(f"  Probabilities: {result['probabilities']}")


def _print_podsai_result(result: dict, label: str = "") -> None:
    """Print PODS-AI inference results for debugging."""
    prefix = f"PODS-AI inference results{f' on {label}' if label else ''}:"
    print(f"\n{prefix}")
    print(f"  Global prediction: {result['global_prediction_label']}")
    print(f"  Global confidence: {result['global_confidence']:.4f}")
    print("  Probabilities:")
    for class_label, prob in sorted(result["probabilities"].items()):
        print(f"    {class_label}: {prob:.4f}")


def _verify_fastai_prediction(result: dict, audio_type: str) -> None:
    """
    Verify FastAI model predicted the correct class for the audio type.

    For non-resident audio, expect "other".
    """
    expected = "resident" if audio_type == "resident" else "other"

    actual = result["global_prediction_label"]
    assert actual == expected, (
        f"FastAI model predicted '{actual}' for {audio_type} audio, "
        f"but expected '{expected}'"
    )


def _verify_podsai_prediction(result: dict, audio_type: str, allow_category_match: bool = False) -> None:
    """
    Verify PODS-AI model predicted the correct class for the audio type.

    Args:
        result: Inference result dictionary.
        audio_type: Expected audio type (resident, transient, humpback, water, vessel, human, jingle, bird).
        allow_category_match: If True, accept category match (whale vs non-whale) instead of exact match.
                              Defaults to False, requiring exact match for all classes.

    For all classes:
        - By default (allow_category_match=False), requires exact match.
        - If allow_category_match=True, accepts category match:
          - For whale classes: accepts any whale class.
          - For non-whale classes: accepts any non-whale class.
    """
    actual = result["global_prediction_label"]

    whale_classes = {"resident", "transient", "humpback"}
    non_whale_classes = {"water", "vessel", "human", "jingle", "bird"}

    if allow_category_match:
        # Category match mode.
        if audio_type in whale_classes:
            # Accept any whale class.
            assert actual in whale_classes, (
                f"PODS-AI model predicted '{actual}' for {audio_type} audio, "
                f"but expected one of {whale_classes}"
            )
        elif audio_type in non_whale_classes:
            # Accept any non-whale class.
            assert actual in non_whale_classes, (
                f"PODS-AI model predicted '{actual}' for {audio_type} audio, "
                f"but expected one of {non_whale_classes}"
            )
    else:
        # Exact match required.
        assert actual == audio_type, (
            f"PODS-AI model predicted '{actual}' for {audio_type} audio, "
            f"but expected exact match '{audio_type}'"
        )


# ---------------------------------------------------------------------------
# Tests for run_inference()
# ---------------------------------------------------------------------------

class TestRunInferencePodsAI:
    """Tests for run_inference() with a mocked PODS-AI model."""

    def test_onnx_model_uses_explicit_feature_extractor(self):
        from run_inference import run_inference

        model = _make_podsai_model_mock()
        with patch("run_inference.get_model_inference", return_value=model) as load_model:
            run_inference(
                "sample.wav",
                model_type="podsai",
                model_path="model.onnx",
                feature_extractor="example/extractor",
                model_revision="0123456789abcdef",
            )

        load_model.assert_called_once_with(
            model_type="podsai",
            model_path="model.onnx",
            feature_extractor="example/extractor",
            model_revision="0123456789abcdef",
        )

    def test_onnx_model_defaults_to_pinned_feature_extractor_revision(self):
        from run_inference import run_inference

        model = _make_podsai_model_mock()
        with patch("run_inference.get_model_inference", return_value=model) as load_model:
            run_inference(
                "sample.wav",
                model_type="podsai",
                model_path="model.onnx",
            )

        load_model.assert_called_once_with(
            model_type="podsai",
            model_path="model.onnx",
            feature_extractor="davethaler/whale-call-detector",
            model_revision=LATEST_PODSAI_MODEL_REVISION,
        )

    def test_returns_expected_keys(self):
        """run_inference returns probabilities, labels, confidence, but NOT proposed_description."""
        wav_path = _make_wav()
        try:
            mock_model = _make_podsai_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="podsai", model_path="fake-path")

            assert "probabilities" in result
            assert "global_prediction_label" in result
            assert "global_confidence" in result
            assert "proposed_description" not in result
            assert "positive_segments_count" in result
            assert "positive_segments" in result
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_positive_segments_include_pacific_timestamps(self):
        """PODS-AI positives include Pacific timestamps when start_time_utc is provided."""
        wav_path = _make_wav()
        try:
            mock_model = _make_podsai_model_mock(num_local=4)
            mock_model.predict.return_value = {
                "local_predictions": [0, 1, 2, 4],
                "local_confidences": [0.6, 0.7, 0.8, 0.9],
                "global_prediction": 1,
                "global_prediction_label": "resident",
                "global_confidence": 0.7,
                "per_class_probabilities": {
                    "water": 0.0,
                    "resident": 0.7,
                    "transient": 0.8,
                    "humpback": 0.0,
                    "vessel": 0.9,
                    "jingle": 0.0,
                    "human": 0.0,
                    "bird": 0.0,
                },
                "hop_duration": 2.0,
                "segment_duration": 3.0,
            }
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference

                result = run_inference(
                    wav_path,
                    model_type="podsai",
                    model_path="fake-path",
                    start_time_utc=datetime(2025, 1, 15, 20, 29, 0, tzinfo=timezone.utc),
                )

            assert result["positive_segments_count"] == 2
            assert len(result["positive_segments"]) == 2
            assert result["positive_segments"][0]["label"] == "resident"
            assert result["positive_segments"][0]["start_time_pacific"] == "2025-01-15 12:29:02 PST"
            assert result["positive_segments"][1]["label"] == "transient"
            assert result["positive_segments"][1]["start_time_pacific"] == "2025-01-15 12:29:04 PST"
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_probabilities_cover_all_seven_classes(self):
        """All 8 PODS-AI class labels are present in the probabilities output."""
        expected_labels = {"water", "resident", "transient", "humpback", "vessel", "jingle", "human", "bird"}
        wav_path = _make_wav()
        try:
            mock_model = _make_podsai_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="podsai", model_path="fake-path")

            assert set(result["probabilities"].keys()) == expected_labels
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_predicted_class_probability_equals_global_confidence(self):
        """The globally predicted class should have a probability equal to global_confidence."""
        wav_path = _make_wav()
        try:
            mock_model = _make_podsai_model_mock(num_local=10)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="podsai", model_path="fake-path")

            # "resident" windows (local_predictions==1) all have confidence 0.7 > threshold 0.5.
            # So resident probability = mean([0.7]*8) = 0.7 = global_confidence.
            assert abs(result["probabilities"]["resident"] - 0.7) < 1e-4
            assert abs(result["probabilities"]["resident"] - result["global_confidence"]) < 1e-4
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_classes_below_threshold_have_low_probability(self):
        """Classes whose windows are all below the confidence threshold should have low probability."""
        wav_path = _make_wav()
        try:
            mock_model = _make_podsai_model_mock(num_local=10)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="podsai", model_path="fake-path")

            # "water" has low average probability (0.1) since most windows don't predict it.
            assert result["probabilities"]["water"] == 0.1
            # Classes never predicted should still be 0.0.
            for label in ["transient", "humpback", "vessel", "jingle", "human", "bird"]:
                assert result["probabilities"][label] == 0.0
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_empty_predictions_returns_all_zero_probabilities(self):
        """When local_predictions is empty, all class probabilities should be 0.0."""
        wav_path = _make_wav()
        try:
            mock_model = _make_podsai_model_mock()
            mock_model.predict.return_value = {
                "local_predictions": [],
                "local_confidences": [],
                "global_prediction": 0,
                "global_prediction_label": "water",
                "global_confidence": 0.0,
                "per_class_probabilities": {
                    "water": 0.0,
                    "resident": 0.0,
                    "transient": 0.0,
                    "humpback": 0.0,
                    "vessel": 0.0,
                    "jingle": 0.0,
                    "human": 0.0,
                    "bird": 0.0,
                },
            }
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="podsai", model_path="fake-path")

            for prob in result["probabilities"].values():
                assert prob == 0.0
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_negative_legacy_prediction_has_no_global_positive_labels(self):
        """Legacy negative results should not be promoted to positive labels."""
        wav_path = _make_wav()
        try:
            mock_model = _make_podsai_model_mock()
            mock_model.predict.return_value = {
                "local_predictions": [],
                "local_confidences": [],
                "global_prediction_label": "water",
                "global_confidence": 0.0,
                "per_class_probabilities": {"water": 1.0},
            }
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference

                result = run_inference(wav_path, model_type="podsai", model_path="fake-path")

            assert result["global_prediction_label"] == "water"
            assert result["global_prediction_labels"] == []
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_defaults_model_path_to_davethaler_hub(self):
        """When model_path is None for podsai, get_model_inference uses davethaler/whale-call-detector."""
        wav_path = _make_wav()
        try:
            mock_model = _make_podsai_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model) as mock_factory:
                from run_inference import run_inference, PODSAI_MODEL_ID
                run_inference(wav_path, model_type="podsai", model_path=None)

            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args
            model_path_arg = call_kwargs.kwargs.get("model_path") or (
                call_kwargs.args[0] if call_kwargs.args else None
            )
            assert model_path_arg == PODSAI_MODEL_ID
            assert call_kwargs.kwargs.get("model_revision") == LATEST_PODSAI_MODEL_REVISION
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_wav2vec2_variant_uses_wav2vec2_revision(self):
        """When --type wav2vec2 is selected, the Wav2Vec2 pinned revision is used."""
        wav_path = _make_wav()
        try:
            mock_model = _make_podsai_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model) as mock_factory:
                from run_inference import (
                    PODSAI_MODEL_ID,
                    PODSAI_WAV2VEC2_MODEL_REVISION,
                    run_inference,
                )
                run_inference(wav_path, model_type="podsai", model_path=None, model_variant="wav2vec2")

            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args
            model_path_arg = call_kwargs.kwargs.get("model_path") or (
                call_kwargs.args[0] if call_kwargs.args else None
            )
            assert model_path_arg == PODSAI_MODEL_ID
            assert call_kwargs.kwargs.get("model_revision") == PODSAI_WAV2VEC2_MODEL_REVISION
        finally:
            Path(wav_path).unlink(missing_ok=True)


class TestRunInferenceFastAI:
    """Tests for run_inference() with a mocked FastAI model."""

    def test_returns_expected_keys(self):
        """run_inference with fastai returns a dict with the required keys."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="fastai", model_path="./model")

            assert "probabilities" in result
            assert "global_prediction_label" in result
            assert "global_confidence" in result
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_probabilities_contain_two_classes(self):
        """FastAI output should contain exactly 'other' and 'resident' classes."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="fastai", model_path="./model")

            assert set(result["probabilities"].keys()) == {"other", "resident"}
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_fastai_probabilities_sum_to_one(self):
        """'other' and 'resident' probabilities should sum to 1.0."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock(global_confidence=0.75, num_local=5)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="fastai", model_path="./model")

            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 1e-3
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_global_prediction_resident_when_positive(self):
        """When global_prediction=1, global_prediction_label should be 'resident'."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock(global_prediction=1, global_confidence=0.8)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="fastai", model_path="./model")

            assert result["global_prediction_label"] == "resident"
            assert abs(result["global_confidence"] - 0.8) < 1e-6
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_global_prediction_other_when_negative(self):
        """When global_prediction=0, global_prediction_label should be 'other'."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock(global_prediction=0, global_confidence=0.0)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="fastai", model_path="./model")

            assert result["global_prediction_label"] == "other"
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_defaults_model_path_to_dot_model(self):
        """When model_path is None for fastai, get_model_inference is called with './model'."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model) as mock_factory:
                from run_inference import run_inference
                run_inference(wav_path, model_type="fastai", model_path=None)

            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args
            assert call_kwargs.kwargs.get("model_path") == "./model" or \
                   (call_kwargs.args and "./model" in call_kwargs.args)
        finally:
            Path(wav_path).unlink(missing_ok=True)


class TestRunInferenceOrcaHello:
    """Tests for run_inference() with a mocked OrcaHello SRKW model."""

    def test_returns_expected_keys(self):
        """run_inference with orcahello returns a dict with the required keys."""
        wav_path = _make_wav()
        try:
            mock_model = _make_orcahello_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="orcahello",
                                       model_path="orcasound/orcahello-srkw-detector-v1")

            assert "probabilities" in result
            assert "global_prediction_label" in result
            assert "global_confidence" in result
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_probabilities_contain_two_classes(self):
        """OrcaHello output should contain exactly 'other' and 'resident' classes."""
        wav_path = _make_wav()
        try:
            mock_model = _make_orcahello_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="orcahello",
                                       model_path="orcasound/orcahello-srkw-detector-v1")

            assert set(result["probabilities"].keys()) == {"other", "resident"}
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_orcahello_probabilities_sum_to_one(self):
        """'other' and 'resident' probabilities should sum to 1.0."""
        wav_path = _make_wav()
        try:
            mock_model = _make_orcahello_model_mock(global_confidence=0.75, num_local=5)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="orcahello",
                                       model_path="orcasound/orcahello-srkw-detector-v1")

            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 1e-3
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_global_prediction_resident_when_positive(self):
        """When global_prediction=1, global_prediction_label should be 'resident'."""
        wav_path = _make_wav()
        try:
            mock_model = _make_orcahello_model_mock(global_prediction=1, global_confidence=0.8)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="orcahello",
                                       model_path="orcasound/orcahello-srkw-detector-v1")

            assert result["global_prediction_label"] == "resident"
            assert abs(result["global_confidence"] - 0.8) < 1e-6
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_global_prediction_other_when_negative(self):
        """When global_prediction=0, global_prediction_label should be 'other'."""
        wav_path = _make_wav()
        try:
            mock_model = _make_orcahello_model_mock(global_prediction=0, global_confidence=0.0)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="orcahello",
                                       model_path="orcasound/orcahello-srkw-detector-v1")

            assert result["global_prediction_label"] == "other"
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_orcahello_inference_error_is_not_silently_accepted(self):
        """Audio decoder failures must fail the runner instead of becoming negatives."""
        wav_path = _make_wav()
        try:
            mock_model = _make_orcahello_model_mock()
            mock_model.predict.return_value["error"] = "LibsndfileError: System error"
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference

                with pytest.raises(RuntimeError, match="OrcaHello inference failed"):
                    run_inference(wav_path, model_type="orcahello", model_path="model")
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_defaults_model_path_to_orcahello_hub(self):
        """When model_path is None for orcahello, get_model_inference uses orcahello-srkw-detector-v1."""
        wav_path = _make_wav()
        try:
            mock_model = _make_orcahello_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model) as mock_factory:
                from run_inference import run_inference
                run_inference(wav_path, model_type="orcahello", model_path=None)

            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args
            model_path_arg = call_kwargs.kwargs.get("model_path") or (
                call_kwargs.args[0] if call_kwargs.args else None
            )
            assert model_path_arg == "orcasound/orcahello-srkw-detector-v1"
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_forwards_model_revision_to_orcahello_loader(self):
        """run_inference forwards an OrcaHello revision when loading the model."""
        wav_path = _make_wav()
        revision = "0123456789abcdef0123456789abcdef01234567"
        try:
            mock_model = _make_orcahello_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model) as factory:
                from run_inference import run_inference

                run_inference(
                    wav_path,
                    model_type="orcahello",
                    model_path="orcasound/orcahello-srkw-detector-v1",
                    model_revision=revision,
                )

            assert factory.call_args.kwargs["model_revision"] == revision
        finally:
            Path(wav_path).unlink(missing_ok=True)



class TestRunInferenceErrors:
    """Tests for error handling in run_inference()."""

    def test_raises_on_unknown_model_type(self):
        """run_inference raises ValueError for an unknown model type."""
        wav_path = _make_wav()
        try:
            from run_inference import run_inference
            with pytest.raises(ValueError, match="Unknown model type"):
                run_inference(wav_path, model_type="unknown_model", model_path="./model")
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_raises_on_unknown_podsai_type(self):
        """run_inference raises ValueError for an unknown PODS-AI model type."""
        wav_path = _make_wav()
        try:
            from run_inference import run_inference
            with pytest.raises(ValueError, match="Unknown PODS-AI model variant"):
                run_inference(wav_path, model_type="podsai", model_path=None, model_variant="unknown")
        finally:
            Path(wav_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests for build_tags_list()
# ---------------------------------------------------------------------------

class TestBuildTagsList:
    """Tests for the build_tags_list() function."""

    def test_tags_include_global_prediction(self):
        """Tags should include positive global prediction label."""
        from run_inference import build_tags_list
        
        result = {
            "global_prediction_label": "resident",
            "local_predictions": [1, 1, 1],
            "local_confidences": [0.7, 0.8, 0.75],
        }
        id2label = {
            1: "resident",
        }
        
        tags = build_tags_list(result, id2label)
        assert set(tags) == {"srkw"}

    def test_tags_append_context_class_when_most_common_segment(self):
        """Tags should include vessel when vessel is most common and differs from global label."""
        from run_inference import build_tags_list
        
        result = {
            "global_prediction_label": "resident",
            "local_predictions": [4, 1, 4, 1, 4, 1, 4],
            "local_confidences": [0.7, 0.7, 0.7, 0.8, 0.7, 0.75, 0.7],
        }
        id2label = {
            1: "resident",
            4: "vessel",
        }
        
        tags = build_tags_list(result, id2label)
        assert set(tags) == {"srkw", "vessel"}

    def test_tags_include_all_global_positive_labels_without_duplicates(self):
        """Multi-label global predictions are preserved before local context."""
        from run_inference import build_tags_list

        result = {
            "global_prediction_label": "resident",
            "global_prediction_labels": ["resident", "transient"],
            "local_predictions": [1, 2, 1, 2],
        }
        id2label = {1: "resident", 2: "transient"}

        assert build_tags_list(result, id2label) == ["srkw", "transient"]

    def test_tags_do_not_add_secondary_local_whale_context(self):
        """Local whale labels below the global threshold remain out of context tags."""
        from run_inference import build_tags_list

        result = {
            "global_prediction_label": "resident",
            "local_predictions": ["resident", "humpback", "resident"],
        }

        assert build_tags_list(result) == ["srkw"]

    def test_abiotic_is_not_treated_as_a_global_whale_label(self):
        """Legacy single-label output does not promote abiotic to a whale tag."""
        from run_inference import build_tags_list

        result = {
            "global_prediction_label": "abiotic",
        }

        assert build_tags_list(result) == []

    def test_tags_deduplicate_global_and_local_positive_labels(self):
        """Global positives are emitted once even when repeated locally."""
        from run_inference import build_tags_list

        result = {
            "global_prediction_label": "resident",
            "global_prediction_labels": ["resident", "transient", "resident"],
            "local_predictions": ["resident", "resident", "water"],
        }

        assert build_tags_list(result) == ["srkw", "transient", "water"]

# ---------------------------------------------------------------------------
# Tests for main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    """Tests for the main() entry point."""

    def test_main_returns_one_when_wav_not_found(self, tmp_path):
        """main() returns exit code 1 when the wav file does not exist."""
        nonexistent = str(tmp_path / "nonexistent.wav")
        with patch("sys.argv", ["run_inference.py", nonexistent, "--model", "fastai", "--model-path", "./model"]):
            from run_inference import main
            assert main() == 1

    def test_main_returns_one_when_no_wav_or_download_args(self):
        """main() returns exit code 1 when neither wav_file nor download args are provided."""
        with patch("sys.argv", ["run_inference.py", "--model", "fastai", "--model-path", "./model"]):
            from run_inference import main
            assert main() == 1

    def test_main_returns_one_when_only_one_download_arg_provided(self):
        """main() returns exit code 1 when only one of download args is provided."""
        with patch(
            "sys.argv",
            ["run_inference.py", "--node-name", "rpi_sunset_bay", "--model", "fastai", "--model-path", "./model"],
        ):
            from run_inference import main
            assert main() == 1

    def test_main_returns_one_when_multiple_timestamp_args_provided(self):
        """main() returns exit code 1 when both start and end timestamp args are provided."""
        with patch(
            "sys.argv",
            [
                "run_inference.py",
                "--node-name",
                "rpi_sunset_bay",
                "--end-timestamp-str",
                "2025_01_15_12_30_00_PST",
                "--start-timestamp-utc",
                "2025-01-15T20:29:00Z",
                "--model",
                "fastai",
                "--model-path",
                "./model",
            ],
        ):
            from run_inference import main
            assert main() == 1

    def test_main_returns_one_when_wav_and_download_args_provided(self):
        """main() returns exit code 1 when both wav_file and download args are provided."""
        wav_path = _make_wav()
        try:
            with patch(
                "sys.argv",
                [
                    "run_inference.py",
                    wav_path,
                    "--node-name",
                    "rpi_sunset_bay",
                    "--end-timestamp-str",
                    "2025_01_15_12_30_00_PST",
                    "--model",
                    "fastai",
                    "--model-path",
                    "./model",
                ],
            ):
                from run_inference import main
                assert main() == 1
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_main_returns_zero_on_success(self):
        """main() returns exit code 0 on successful inference."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch("sys.argv", ["run_inference.py", wav_path, "--model", "fastai", "--model-path", "./model"]), \
                 patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import main
                assert main() == 0
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_main_forwards_feature_extractor_option(self):
        wav_path = _make_wav()
        try:
            with patch(
                "sys.argv",
                [
                    "run_inference.py",
                    wav_path,
                    "--model",
                    "podsai",
                    "--model-path",
                    "model.onnx",
                    "--feature-extractor",
                    "example/extractor",
                ],
            ), patch(
                "run_inference.run_inference", return_value={}
            ) as invoke, patch("run_inference.print_results"):
                from run_inference import main

                assert main() == 0

            assert invoke.call_args.kwargs["feature_extractor"] == "example/extractor"
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_main_downloads_wav_from_node_name_and_timestamp(self):
        """main() can download wav when --node-name and --end-timestamp-str are provided."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch(
                "sys.argv",
                [
                    "run_inference.py",
                    "--node-name",
                    "rpi_sunset_bay",
                    "--end-timestamp-str",
                    "2025_01_15_12_30_00_PST",
                    "--model",
                    "fastai",
                    "--model-path",
                    "./model",
                ],
            ), patch(
                "run_inference.download_audio_from_start_utc",
                return_value=wav_path,
            ) as mock_download, patch(
                "run_inference.get_model_inference", return_value=mock_model
            ):
                from run_inference import main
                assert main() == 0
                mock_download.assert_called_once()
                _, start_time_utc, _ = mock_download.call_args.args
                assert start_time_utc == datetime(2025, 1, 15, 20, 29, tzinfo=timezone.utc)
                assert mock_download.call_args.kwargs["duration_seconds"] == 60.0
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_main_training_set_downloads_from_end_timestamp(self):
        """main() requests a 3-second training clip ending at the given timestamp."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch(
                "sys.argv",
                [
                    "run_inference.py",
                    "--node-name",
                    "rpi_sunset_bay",
                    "--end-timestamp-str",
                    "2025_01_15_12_30_00_PST",
                    "--set",
                    "training",
                    "--model",
                    "fastai",
                    "--model-path",
                    "./model",
                ],
            ), patch(
                "run_inference.download_audio_from_start_utc",
                return_value=wav_path,
            ) as mock_download, patch(
                "run_inference.get_model_inference", return_value=mock_model
            ):
                from run_inference import main
                assert main() == 0
                mock_download.assert_called_once()
                _, start_time_utc, _ = mock_download.call_args.args
                assert start_time_utc == datetime(2025, 1, 15, 20, 29, 57, tzinfo=timezone.utc)
                assert mock_download.call_args.kwargs["duration_seconds"] == 3.0
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_main_downloads_wav_from_node_name_and_start_timestamp_utc(self):
        """main() can download wav when --node-name and --start-timestamp-utc are provided."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch(
                "sys.argv",
                [
                    "run_inference.py",
                    "--node-name",
                    "rpi_sunset_bay",
                    "--start-timestamp-utc",
                    "2025-01-15T20:29:00Z",
                    "--model",
                    "fastai",
                    "--model-path",
                    "./model",
                ],
            ), patch(
                "run_inference.download_audio_from_start_utc",
                return_value=wav_path,
            ) as mock_download, patch(
                "run_inference.get_model_inference", return_value=mock_model
            ):
                from run_inference import main
                assert main() == 0
                mock_download.assert_called_once()
                _, start_time_utc, _ = mock_download.call_args[0]
                assert isinstance(start_time_utc, datetime)
                assert start_time_utc.tzinfo == timezone.utc
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_main_training_set_downloads_3s_clip(self):
        """main() requests a 3-second clip when --set training is given."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch(
                "sys.argv",
                [
                    "run_inference.py",
                    "--node-name",
                    "rpi_sunset_bay",
                    "--start-timestamp-utc",
                    "2025-01-15T20:29:00Z",
                    "--set",
                    "training",
                    "--model",
                    "fastai",
                    "--model-path",
                    "./model",
                ],
            ), patch(
                "run_inference.download_audio_from_start_utc",
                return_value=wav_path,
            ) as mock_download, patch(
                "run_inference.get_model_inference", return_value=mock_model
            ):
                from run_inference import main
                assert main() == 0
                assert mock_download.call_args.kwargs["duration_seconds"] == 3.0
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_main_returns_one_when_download_fails(self):
        """main() returns exit code 1 when audio download returns None."""
        with patch(
            "sys.argv",
            [
                "run_inference.py",
                "--node-name",
                "rpi_sunset_bay",
                "--end-timestamp-str",
                "2025_01_15_12_30_00_PST",
                "--model",
                "fastai",
                "--model-path",
                "./model",
            ],
        ), patch("run_inference.download_audio_from_start_utc", return_value=None):
            from run_inference import main
            assert main() == 1

    def test_main_returns_one_on_value_error(self):
        """main() returns exit code 1 when run_inference raises ValueError."""
        wav_path = _make_wav()
        try:
            # Calling with an unknown model type should raise ValueError.
            with patch("sys.argv", ["run_inference.py", wav_path, "--model", "unknown_model"]):
                from run_inference import main
                assert main() == 1
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_print_results_output_contains_class_names(self, capsys):
        """print_results() writes class names to stdout."""
        from run_inference import print_results
        results = {
            "probabilities": {"other": 0.3, "resident": 0.7},
            "global_prediction_label": "resident",
            "global_confidence": 0.7,
        }
        print_results(results, "fastai")
        captured = capsys.readouterr()
        assert "other" in captured.out
        assert "resident" in captured.out
        assert "fastai" in captured.out
        assert "0.7000" in captured.out

    def test_print_results_outputs_positive_segment_summary_for_podsai(self, capsys):
        """print_results() writes positive segment count and timestamps for PODS-AI."""
        from run_inference import print_results
        results = {
            "probabilities": {"resident": 0.7, "water": 0.3},
            "global_prediction_label": "resident",
            "global_confidence": 0.7,
            "local_predictions": [0, 1, 2, 4],
            "local_prediction_labels": ["water", "resident", "transient", "vessel"],
            "local_confidences": [0.9, 0.7, 0.8, 0.9],
            "positive_segments_count": 2,
            "positive_segments": [
                {
                    "label": "resident",
                    "confidence": 0.7,
                    "start_time_pacific": "2025-01-15 12:29:02 PST",
                },
                {
                    "label": "transient",
                    "confidence": 0.8,
                    "start_time_pacific": "2025-01-15 12:29:04 PST",
                },
            ],
        }
        print_results(results, "podsai")
        captured = capsys.readouterr()
        assert "Positive segments: 2/4" in captured.out
        assert "  +2.0s: resident (whale confidence: 0.700)" in captured.out
        assert "  +4.0s: transient (whale confidence: 0.800)" in captured.out


class TestPinnedPodsAIModelPath:
    """Tests for pinned PODS-AI model resolution used by integration tests."""

    def test_prefers_local_multiclass_directory(self, tmp_path, monkeypatch):
        """When model/multiclass exists locally, no Hub download is needed."""
        monkeypatch.chdir(tmp_path)
        local_dir = Path("model/multiclass")
        local_dir.mkdir(parents=True)

        assert _resolve_podsai_test_model_path() == str(local_dir)

    def test_downloads_pinned_hub_revision(self, tmp_path, monkeypatch):
        """When local model is absent, a pinned Hub revision is downloaded."""
        monkeypatch.chdir(tmp_path)
        with (
            patch("huggingface_hub.file_exists", return_value=True) as mock_file_exists,
            patch("huggingface_hub.snapshot_download", return_value="/tmp/pinned-model") as mock_snapshot,
        ):
            assert _resolve_podsai_test_model_path() == "/tmp/pinned-model"

        mock_file_exists.assert_called_once_with(
            PODSAI_TEST_MODEL_ID,
            "preprocessor_config.json",
            revision=LATEST_PODSAI_MODEL_REVISION,
        )
        mock_snapshot.assert_called_once_with(
            repo_id=PODSAI_TEST_MODEL_ID,
            revision=LATEST_PODSAI_MODEL_REVISION,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "training_args.bin",
            ],
        )


# ---------------------------------------------------------------------------
# Integration tests with real models (if available)
# ---------------------------------------------------------------------------

class TestIntegrationWithRealModels:
    """
    Integration tests that run inference on real wav files with real models.

    These tests are skipped if the required models or wav files are not present.
    They verify end-to-end functionality with actual model weights.
    """

    # Shared fixtures for model paths.
    @pytest.fixture
    def fastai_model_path(self) -> str:
        """Path to the FastAI model directory."""
        path = Path("model")
        if not path.exists():
            # Attempt to download the model automatically before skipping.
            from model_inference import download_model_if_needed
            try:
                download_model_if_needed(str(path))
            except Exception as e:
                pytest.skip(f"FastAI model download failed: {e}")
            if not path.exists():
                pytest.skip(f"FastAI model directory not found: {path}")
        return str(path)

    @pytest.fixture
    def podsai_model_path(self) -> str:
        """Path to local PODS-AI model or a pinned Hub snapshot fallback."""
        return _resolve_podsai_test_model_path()

    @pytest.fixture
    def podsai_onnx_model_path(self) -> str:
        """Path to an exported PODS-AI ONNX model, when available."""
        path = Path("podsai.onnx")
        if not path.is_file():
            pytest.skip(f"ONNX model not found at {path}")
        pytest.importorskip(
            "onnxruntime", reason="ONNX Runtime is required for ONNX inference"
        )
        return str(path)

    # Fixtures for test wav files (one per audio type).
    def _get_testing_wav_path(self, category: str) -> str:
        """Download (if needed) and return one testing wav path for the given category.

        If a WAV file already exists in output/testing-wav/{category}/, it is returned
        immediately.  Otherwise the first matching row in testing_samples.csv is
        downloaded via download_testing_sample().  The test is skipped when neither
        the file nor the CSV is available, or when the download fails.
        """
        from download_wavs import download_testing_sample, parse_csv

        output_root = Path("output/testing-wav")
        category_dir = output_root / category

        # Return early if a file was already downloaded.
        candidates = sorted(category_dir.glob("*.wav"))
        if candidates:
            return str(candidates[0])

        # Attempt to download one sample from the testing CSV.
        testing_csv_path = Path("output/csv/testing_60s_samples.csv")
        if not testing_csv_path.exists():
            pytest.skip(f"No testing samples CSV found at {testing_csv_path}")

        rows = parse_csv(testing_csv_path)
        category_rows = [row for row in rows if row.category == category]
        if not category_rows:
            pytest.skip(f"No testing samples for category '{category}' in {testing_csv_path}")

        # Try each row until one downloads successfully.
        for row in category_rows:
            download_testing_sample(row, output_root)
            candidates = sorted(category_dir.glob("*.wav"))
            if candidates:
                return str(candidates[0])

        pytest.skip(f"Failed to download a testing wav for category '{category}'")

    @pytest.fixture
    def resident_wav_path(self) -> str:
        """Path to a real 60-second resident orca wav file for testing."""
        return self._get_testing_wav_path("resident")

    @pytest.fixture
    def transient_wav_path(self) -> str:
        """Path to a real 60-second transient orca wav file for testing."""
        return self._get_testing_wav_path("transient")

    @pytest.fixture
    def humpback_wav_path(self) -> str:
        """Path to a real 60-second humpback whale wav file for testing."""
        return self._get_testing_wav_path("humpback")

    @pytest.fixture
    def vessel_wav_path(self) -> str:
        """Path to a real 60-second vessel noise wav file for testing."""
        return self._get_testing_wav_path("vessel")

    @pytest.fixture
    def water_wav_path(self) -> str:
        """Path to a real 60-second water/ambient noise wav file for testing."""
        return self._get_testing_wav_path("water")

    @pytest.fixture
    def human_wav_path(self) -> str:
        """Path to a real 60-second human voice wav file for testing."""
        return self._get_testing_wav_path("human")

    @pytest.fixture
    def jingle_wav_path(self) -> str:
        """Path to a real 60-second jingle/signal wav file for testing."""
        return self._get_testing_wav_path("jingle")

    @pytest.fixture
    def bird_wav_path(self) -> str:
        """Path to a real 60-second bird/signal wav file for testing."""
        return self._get_testing_wav_path("bird")

    # Parametrized tests for FastAI model on different audio types.
    @pytest.mark.parametrize("label,wav_file,xfail_reason", [
        ("resident", "rpi-sunset-bay_2026_01_08_01_52_39_PST.wav", None),
        ("transient", "rpi-orcasound-lab_2024_12_29_13_37_23_PST.wav",
         "FastAI binary model may predict resident on transient clips"),
        ("humpback", "rpi-orcasound-lab_2023_10_28_08_08_03_PST.wav", None),
        ("vessel", "rpi-orcasound-lab_2025_11_30_12_53_54_PST.wav", "FastAI binary model may predict vessel as resident"),
        ("water", "rpi-andrews-bay_2026_08_30_02_22_52_PST.wav", None),
        ("human", "rpi-sunset-bay_2024_07_19_12_50_08_PST.wav", None),
        ("jingle", "rpi-bush-point_2026_07_19_09_54_48_PST.wav", "FastAI binary model may predict jingle as resident"),
        ("bird", "rpi-bush-point_2026_08_09_17_36_48_PST.wav", None),
    ])
    def test_fastai_model_inference(
        self,
        label: str,
        wav_file: str,
        xfail_reason: Optional[str],
        fastai_model_path: str,
        request: pytest.FixtureRequest
    ) -> None:
        """Test FastAI model inference on various audio types."""

        # Apply xfail marker if this test case is expected to fail.
        if xfail_reason:
            request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=False))

        result = model_inference_helper(label, wav_file, "fastai", fastai_model_path)

        _verify_fastai_result_structure(result)
        _verify_fastai_prediction(result, label)
        _print_fastai_result(result, label)

    # Parametrized tests for PODS-AI model on different audio types.
    @pytest.mark.parametrize("label,wav_file,xfail_reason", [
        ("resident", "rpi-sunset-bay_2026_01_08_01_52_39_PST.wav", None),
        ("transient", "rpi-orcasound-lab_2025_01_02_10_12_24_PST.wav", None),
        ("humpback", "rpi-orcasound-lab_2025_01_05_21_18_05_PST.wav", None),
        ("vessel", "rpi-orcasound-lab_2025_11_30_12_53_54_PST.wav", None),
        ("water", "rpi-bush-point_2025_06_29_18_23_17_PST.wav", "PODS-AI model may misclassify water as vessel"),
        ("human", "rpi-sunset-bay_2024_07_19_12_50_08_PST.wav", None),
        ("jingle", "rpi-bush-point_2026_07_19_09_54_48_PST.wav", None),
        ("bird", "rpi-bush-point_2026_08_09_17_36_48_PST.wav", "PODS-AI model may misclassify bird as jingle"),
    ])
    def test_podsai_model_inference(
        self,
        label: str,
        wav_file: str,
        xfail_reason: Optional[str],
        podsai_model_path: str,
        request: pytest.FixtureRequest
    ) -> None:
        """Test PODS-AI model inference on various audio types."""

        # Apply xfail marker if this test case is expected to fail.
        if xfail_reason:
            request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=False))

        result = model_inference_helper(label, wav_file, "podsai", podsai_model_path)

        _verify_podsai_result_structure(result)
        # Always require exact match - no category matching allowed.
        _verify_podsai_prediction(result, label, allow_category_match=False)
        _print_podsai_result(result, label)

    # Parametrized tests for the exported PODS-AI ONNX model on different audio types.
    @pytest.mark.parametrize("label,wav_file,xfail_reason", [
        ("resident", "rpi-sunset-bay_2026_01_08_01_52_39_PST.wav", None),
        ("transient", "rpi-orcasound-lab_2025_01_02_10_12_24_PST.wav", None),
        ("humpback", "rpi-orcasound-lab_2025_01_05_21_18_05_PST.wav", None),
        ("vessel", "rpi-orcasound-lab_2025_11_30_12_53_54_PST.wav", None),
        ("water", "rpi-bush-point_2025_06_29_18_23_17_PST.wav", "PODS-AI model may misclassify water as vessel"),
        ("human", "rpi-sunset-bay_2024_07_19_12_50_08_PST.wav", None),
        ("jingle", "rpi-bush-point_2026_07_19_09_54_48_PST.wav", None),
        ("bird", "rpi-bush-point_2026_08_09_17_36_48_PST.wav", "PODS-AI model may misclassify bird as jingle"),
    ])
    def test_podsai_onnx_model_inference(
        self,
        label: str,
        wav_file: str,
        xfail_reason: Optional[str],
        podsai_onnx_model_path: str,
        request: pytest.FixtureRequest
    ) -> None:
        """Test exported PODS-AI ONNX inference on various audio types."""

        if xfail_reason:
            request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=False))

        result = model_inference_helper(
            label, wav_file, "podsai", podsai_onnx_model_path
        )

        _verify_podsai_result_structure(result)
        _verify_podsai_prediction(result, label, allow_category_match=False)
        _print_podsai_result(result, label)

    # Parametrized CLI integration tests.
    @pytest.mark.parametrize("label,wav_file,model_type,model_path_fixture", [
        ("resident", "rpi-sunset-bay_2026_01_08_01_52_39_PST.wav", "fastai", "fastai_model_path"),
        ("resident", "rpi-sunset-bay_2026_01_08_01_52_39_PST.wav", "podsai", "podsai_model_path"),
        ("transient", "rpi-orcasound-lab_2024_12_29_13_37_23_PST.wav", "fastai", "fastai_model_path"),
        ("transient", "rpi-orcasound-lab_2024_12_29_13_37_23_PST.wav", "podsai", "podsai_model_path"),
        ("humpback", "rpi-orcasound-lab_2023_10_28_08_08_03_PST.wav", "fastai", "fastai_model_path"),
        ("humpback", "rpi-orcasound-lab_2023_10_28_08_08_03_PST.wav", "podsai", "podsai_model_path"),
        ("vessel", "rpi-orcasound-lab_2025_11_30_12_53_54_PST.wav", "fastai", "fastai_model_path"),
        ("vessel", "rpi-orcasound-lab_2025_11_30_12_53_54_PST.wav", "podsai", "podsai_model_path"),
        ("water", "rpi-andrews-bay_2026_08_30_02_22_52_PST.wav", "fastai", "fastai_model_path"),
        ("water", "rpi-andrews-bay_2026_08_30_02_22_52_PST.wav", "podsai", "podsai_model_path"),
        ("human", "rpi-sunset-bay_2024_07_19_12_50_08_PST.wav", "fastai", "fastai_model_path"),
        ("human", "rpi-sunset-bay_2024_07_19_12_50_08_PST.wav", "podsai", "podsai_model_path"),
        ("jingle", "rpi-bush-point_2026_07_19_09_54_48_PST.wav", "fastai", "fastai_model_path"),
        ("jingle", "rpi-bush-point_2026_07_19_09_54_48_PST.wav", "podsai", "podsai_model_path"),
        ("bird", "rpi-bush-point_2026_08_09_17_36_48_PST.wav", "fastai", "fastai_model_path"),
        ("bird", "rpi-bush-point_2026_08_09_17_36_48_PST.wav", "podsai", "podsai_model_path"),
    ])
    def test_cli_integration(
        self,
        label: str,
        wav_file: str,
        model_type: str,
        model_path_fixture: str,
        request: pytest.FixtureRequest
    ) -> None:
        """Test CLI integration with various audio types and models."""
        from run_inference import main

        output_root = Path("output/testing-wav")
        category_dir = output_root / label
        wav_path = category_dir / wav_file
        model_path = request.getfixturevalue(model_path_fixture)

        with patch("sys.argv", [
            "run_inference.py",
            str(wav_path),
            "--model", model_type,
            "--model-path", model_path
        ]):
            exit_code = main()

        assert exit_code == 0

    @pytest.fixture
    def orcahello_model_path(self) -> str:
        """HuggingFace Hub model ID for the OrcaHello SRKW detector."""
        hub_id = "orcasound/orcahello-srkw-detector-v1"
        try:
            from huggingface_hub import file_exists as hf_file_exists
            if not hf_file_exists(hub_id, "config.json"):
                pytest.skip(
                    f"Hub model '{hub_id}' is not accessible. "
                    f"Check your internet connection and HuggingFace Hub availability."
                )
        except Exception:
            pytest.skip(f"HuggingFace Hub is not reachable; cannot load model '{hub_id}'")
        return hub_id

    @pytest.mark.parametrize("label,wav_file,xfail_reason", [
        ("resident", "rpi-sunset-bay_2026_01_08_01_52_39_PST.wav", None),
        ("transient", "rpi-orcasound-lab_2024_12_29_13_37_23_PST.wav",
         "OrcaHello SRKW detector may predict resident on transient clips"),
        ("humpback", "rpi-orcasound-lab_2023_10_28_08_08_03_PST.wav", None),
        ("vessel", "rpi-orcasound-lab_2025_11_30_12_53_54_PST.wav",
         "OrcaHello SRKW detector may predict resident on vessel noise clips"),
        ("water", "rpi-andrews-bay_2026_08_30_02_22_52_PST.wav", None),
        ("human", "rpi-sunset-bay_2024_07_19_12_50_08_PST.wav",
         "OrcaHello SRKW detector may predict resident on human voice clips"),
        ("jingle", "rpi-bush-point_2026_07_19_09_54_48_PST.wav",
         "OrcaHello SRKW detector may predict resident on jingle clips"),
        ("bird", "rpi-bush-point_2026_08_09_17_36_48_PST.wav", None),
    ])
    def test_orcahello_model_inference(
        self,
        label: str,
        wav_file: str,
        xfail_reason: Optional[str],
        orcahello_model_path: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test OrcaHello SRKW detector inference on various audio types."""

        if xfail_reason:
            request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=False))

        result = model_inference_helper(label, wav_file, "orcahello", orcahello_model_path)

        # OrcaHello is a binary model: "resident" or "other".
        _verify_fastai_result_structure(result)
        _verify_fastai_prediction(result, label)
        _print_fastai_result(result, f"orcahello-{label}")
