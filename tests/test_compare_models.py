#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for compare_models.py.

Tests cover:
- load_test_samples() WAV discovery
- find_wav_file() path construction
- is_correct_prediction() label mapping
- evaluate_model() with mocked run_inference
- print_summary() output
- ModelResult property calculations, including whale-class F1 and per-whale-class FP/FN rates
- main() CLI error handling
"""

import sys
from pathlib import Path
from unittest.mock import patch
from model_revisions import LATEST_PODSAI_MODEL_REVISION, PREVIOUS_PODSAI_MODEL_REVISION, PODSAI_WAV2VEC2_MODEL_REVISION

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_testing_rows():
    """Return a list of testing sample rows."""
    return [
        {
            "Category": "resident",
            "NodeName": "rpi_orcasound_lab",
            "StartTimestamp": "2023_08_18_00_59_53_PST",
            "URI": "https://example.com/1",
            "Description": "J pod calls",
            "Notes": "tp_human_only",
            "Confidence": "",
        },
        {
            "Category": "human",
            "NodeName": "rpi_sunset_bay",
            "StartTimestamp": "2024_08_07_11_23_23_PST",
            "URI": "https://example.com/2",
            "Description": "Human voices",
            "Notes": "fp_machine_only",
            "Confidence": "62.3839",
        },
        {
            "Category": "humpback",
            "NodeName": "rpi_orcasound_lab",
            "StartTimestamp": "2023_10_28_07_33_52_PST",
            "URI": "https://example.com/3",
            "Description": "Humpback",
            "Notes": "tp_human_only",
            "Confidence": "",
        },
    ]

def _make_wav(wav_dir, category="resident", node_name="rpi_orcasound_lab", start_timestamp="2023_08_18_00_59_53_PST"):
    """Create a dummy WAV file for the sample and return a list containing the filename."""
    node_name_in_filename = node_name.replace("_", "-")
    wav_file = wav_dir / category / f"{node_name_in_filename}_{start_timestamp}.wav"
    wav_file.parent.mkdir(parents=True, exist_ok=True)
    wav_file.touch()
    return [wav_file]

def test_multilabel_prediction_counts_matching_species_as_correct():
    """Exact-match models accept a supported class in a multi-label result."""
    from compare_models import is_correct_prediction

    assert is_correct_prediction("transient", "resident", "podsai", ["resident", "transient"])
    assert not is_correct_prediction("humpback", "resident", "podsai", ["resident", "transient"])


def test_validate_custom_model_path_expands_tilde(tmp_path, monkeypatch):
    """Existing tilde paths are normalized before they reach the loader."""
    from compare_models import resolve_custom_model_path

    local_model = tmp_path / "custom-model"
    local_model.mkdir()
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_custom_model_path("~/custom-model") == (str(local_model), True)


def test_validate_custom_model_path_rejects_missing_local_path(tmp_path):
    """Missing local-looking paths fail before model loading."""
    from compare_models import validate_custom_model_path

    missing = tmp_path / "missing-model"
    with pytest.raises(ValueError, match="does not exist"):
        validate_custom_model_path(str(missing))
    with pytest.raises(ValueError, match="does not exist"):
        validate_custom_model_path("/missing-model")


def test_validate_inference_model_rejects_non_podsai_backend():
    """Custom models must expose the PODS-AI predict method."""
    from compare_models import validate_inference_model

    with pytest.raises(ValueError, match=r"predict\(\)"):
        validate_inference_model(object(), "custom")


def test_main_loads_custom_model_from_cli(tmp_path):
    """main() loads and evaluates the custom model selected by the CLI."""
    from compare_models import ModelResult, main

    wav_dir = tmp_path / "testing-wav"
    _make_wav(wav_dir, "resident")
    custom_model = tmp_path / "custom-model"
    custom_model.mkdir()
    args = [
        "compare_models.py", "--wav-dir", str(wav_dir), "--models", "custom",
        "--custom-model-path", str(custom_model),
        "--custom-model-revision", "deadbeef",
        "--max-samples", "1",
    ]
    with patch.object(sys, "argv", args), \
         patch("compare_models.get_model_inference", return_value=type("Model", (), {"predict": lambda self, _: {}})()) as load_model, \
         patch("compare_models.evaluate_model", return_value=ModelResult("custom")):
        assert main() == 0
    load_model.assert_called_once_with(
        model_type="podsai", model_path=str(custom_model), model_revision=None
    )


def test_main_expands_tilde_custom_model_path(tmp_path, monkeypatch):
    """main() passes the expanded local path to the model loader."""
    from compare_models import ModelResult, main

    wav_dir = tmp_path / "testing-wav"
    _make_wav(wav_dir, "resident")
    custom_model = tmp_path / "custom-model"
    custom_model.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    args = [
        "compare_models.py", "--wav-dir", str(wav_dir), "--models", "custom",
        "--custom-model-path", "~/custom-model", "--max-samples", "1",
    ]
    model = type("Model", (), {"predict": lambda self, _: {}})()
    with patch.object(sys, "argv", args), \
         patch("compare_models.get_model_inference", return_value=model) as load_model, \
         patch("compare_models.evaluate_model", return_value=ModelResult("custom")):
        assert main() == 0
    load_model.assert_called_once_with(
        model_type="podsai", model_path=str(custom_model), model_revision=None
    )


def test_main_forwards_custom_model_revision(tmp_path):
    """The custom Hub revision is passed to the PODS-AI loader."""
    from compare_models import ModelResult, main

    wav_dir = tmp_path / "testing-wav"
    _make_wav(wav_dir, "resident")
    args = [
        "compare_models.py", "--wav-dir", str(wav_dir), "--models", "custom",
        "--custom-model-path", "org/model", "--custom-model-revision", "deadbeef",
        "--max-samples", "1",
    ]
    model = type("Model", (), {"predict": lambda self, _: {}})()
    with patch.object(sys, "argv", args), \
         patch("compare_models.get_model_inference", return_value=model) as load_model, \
         patch("compare_models.evaluate_model", return_value=ModelResult("custom")):
        assert main() == 0
    load_model.assert_called_once_with(
        model_type="podsai", model_path="org/model", model_revision="deadbeef"
    )


def test_main_forwards_orcahello_model_revision(tmp_path):
    """The OrcaHello revision option reaches model loading and evaluation."""
    from compare_models import ModelResult, main

    wav_dir = tmp_path / "testing-wav"
    _make_wav(wav_dir, "resident")
    revision = "0123456789abcdef0123456789abcdef01234567"
    args = [
        "compare_models.py", "--wav-dir", str(wav_dir), "--models", "orcahello",
        "--orcahello-model-revision", revision, "--max-samples", "1",
    ]
    with patch.object(sys, "argv", args), \
         patch("compare_models.get_model_inference", return_value=object()) as load_model, \
         patch("compare_models.evaluate_model", return_value=ModelResult("orcahello")) as evaluate:
        assert main() == 0

    load_model.assert_called_once_with(
        model_type="orcahello",
        model_path="orcasound/orcahello-srkw-detector-v1",
        model_revision=revision,
    )
    assert evaluate.call_args.kwargs["model_revision"] == revision


def test_custom_model_uses_category_membership_for_abiotic():
    """Custom models use multi-class membership for background categories."""
    from compare_models import is_correct_prediction

    assert is_correct_prediction("abiotic", "other", "custom", ["abiotic"])
    assert is_correct_prediction("abiotic", "other", "fastai", ["other"])


# ---------------------------------------------------------------------------
# Tests for load_test_samples()
# ---------------------------------------------------------------------------

class TestLoadTestSamples:
    """Tests for load_test_samples()."""

    def _write_testing_wavs(self, wav_dir: Path, rows):
        """Create WAV files from testing rows under category directories."""
        for row in rows:
            node = row["NodeName"].replace("_", "-")
            wav = wav_dir / row["Category"] / f"{node}_{row['StartTimestamp']}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.touch()

    def test_loads_all_samples(self, tmp_path):
        """load_test_samples loads all matching WAV files under wav_dir."""
        from compare_models import load_test_samples, get_category_from_path
        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, _make_testing_rows())

        wav_paths = load_test_samples(wav_dir)
        assert len(wav_paths) == 3
        assert {get_category_from_path(s, wav_dir) for s in wav_paths} == {"resident", "human", "humpback"}


    def test_returns_empty_list_for_missing_dir(self):
        """load_test_samples returns [] when wav_dir does not exist."""
        from compare_models import load_test_samples

        wav_paths = load_test_samples(Path("/nonexistent/testing-wav"))
        assert wav_paths == []


    def test_respects_max_samples_limit(self, tmp_path):
        """load_test_samples respects max_samples parameter."""
        from compare_models import load_test_samples
        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, _make_testing_rows())

        wav_paths = load_test_samples(wav_dir, max_samples=2)
        assert len(wav_paths) == 2

    def test_category_filter_returns_only_matching_samples(self, tmp_path):
        """load_test_samples returns only samples matching the category filter."""
        from compare_models import load_test_samples, get_category_from_path
        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, _make_testing_rows())

        wav_paths = load_test_samples(wav_dir, category_filter="resident")
        assert len(wav_paths) == 1
        assert all(get_category_from_path(s, wav_dir) == "resident" for s in wav_paths)

    def test_category_filter_returns_empty_for_no_match(self, tmp_path):
        """load_test_samples returns [] when category filter matches no rows."""
        from compare_models import load_test_samples
        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, _make_testing_rows())

        wav_paths = load_test_samples(wav_dir, category_filter="transient")
        assert wav_paths == []

    def test_category_filter_combined_with_max_samples(self, tmp_path):
        """load_test_samples applies both category_filter and max_samples."""
        from compare_models import load_test_samples, get_category_from_path

        wav_dir = tmp_path / "testing-wav"
        rows = [
            {"Category": "humpback", "NodeName": "rpi_a", "StartTimestamp": "2024_01_01_00_00_00_PST"},
            {"Category": "humpback", "NodeName": "rpi_b", "StartTimestamp": "2024_01_01_00_01_00_PST"},
            {"Category": "humpback", "NodeName": "rpi_c", "StartTimestamp": "2024_01_01_00_02_00_PST"},
        ]
        self._write_testing_wavs(wav_dir, rows)

        wav_paths = load_test_samples(wav_dir, max_samples=2, category_filter="humpback")
        assert len(wav_paths) == 2
        assert all(get_category_from_path(s, wav_dir) == "humpback" for s in wav_paths)


    def test_loads_uppercase_wav_extension(self, tmp_path):
        """load_test_samples accepts uppercase WAV filename extensions."""
        from compare_models import load_test_samples, get_category_from_path

        wav_dir = tmp_path / "testing-wav"
        wav = wav_dir / "resident" / "rpi-lab_2023_01_01_00_00_00_PST.WAV"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.touch()

        wav_paths = load_test_samples(wav_dir)
        assert len(wav_paths) == 1
        assert get_category_from_path(wav_paths[0], wav_dir) == "resident"


# ---------------------------------------------------------------------------
# Tests for ModelResult properties
# ---------------------------------------------------------------------------

class TestModelResultProperties:
    """Tests for ModelResult computed properties."""

    def test_evaluated_excludes_skipped(self):
        """evaluated = total - skipped."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, skipped=3)
        assert r.evaluated == 7

    def test_accuracy_none_when_no_evaluated(self):
        """accuracy is None when evaluated == 0."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=5, skipped=5)
        assert r.accuracy is None

    def test_accuracy_correct_fraction(self):
        """accuracy is correct/evaluated."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, correct=8, skipped=0)
        assert abs(r.accuracy - 0.8) < 1e-9

    def test_false_positive_rate_none_when_no_evaluated(self):
        """false_positive_rate is None when evaluated == 0."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=3, skipped=3)
        assert r.false_positive_rate is None

    def test_false_positive_rate_correct_fraction(self):
        """false_positive_rate is false_positives/evaluated."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, false_positives=2, skipped=0)
        assert abs(r.false_positive_rate - 0.2) < 1e-9

    def test_false_negative_rate_correct_fraction(self):
        """false_negative_rate is false_negatives/evaluated."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, false_negatives=1, skipped=0)
        assert abs(r.false_negative_rate - 0.1) < 1e-9

    def test_avg_predict_time_none_when_no_times(self):
        """avg_predict_time is None when predict_times is empty."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=5, skipped=5)
        assert r.avg_predict_time is None

    def test_avg_predict_time_calculates_mean(self):
        """avg_predict_time is the mean of predict_times."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=3, predict_times=[1.0, 2.0, 3.0])
        assert abs(r.avg_predict_time - 2.0) < 1e-9

    def test_whale_f1_none_when_no_whale_labels_present(self):
        """whale_f1 is None when the confusion matrix includes no whale classes."""
        from compare_models import ModelResult
        r = ModelResult(
            model_type="fastai",
            confusion_matrix={"human": {"other": 2}, "water": {"other": 1}},
        )
        assert r.whale_f1 is None

    def test_whale_f1_macro_average_over_present_whale_classes(self):
        """whale_f1 averages per-class F1 over present humpback/resident/transient labels."""
        from compare_models import ModelResult
        r = ModelResult(
            model_type="podsai",
            confusion_matrix={
                "resident": {"resident": 2, "transient": 1},
                "humpback": {"humpback": 1, "resident": 1},
                "transient": {"transient": 1},
                "human": {"resident": 1},
            },
        )
        assert r.whale_f1 == pytest.approx((4 / 7 + 2 / 3 + 2 / 3) / 3)

    def test_false_negative_rate_for_label_uses_actual_label_denominator(self):
        """Per-label FN% is normalized by the number of actual samples of that label."""
        from compare_models import ModelResult
        r = ModelResult(
            model_type="fastai",
            total=4,
            confusion_matrix={
                "transient": {"other": 2},
                "resident": {"resident": 1},
                "human": {"other": 1},
            },
        )
        assert r.false_negative_count_for_label("transient") == 2
        assert r.false_negative_rate_for_label("transient") == pytest.approx(1.0)

    def test_false_positive_rate_for_label_uses_non_label_denominator(self):
        """Per-label FP% is normalized by samples whose true label is not that label."""
        from compare_models import ModelResult
        r = ModelResult(
            model_type="podsai",
            total=5,
            confusion_matrix={
                "resident": {"resident": 1},
                "human": {"resident": 1, "human": 1},
                "transient": {"resident": 1, "transient": 1},
            },
        )
        assert r.false_positive_count_for_label("resident") == 2
        assert r.false_positive_rate_for_label("resident") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests for evaluate_model()
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    """Tests for evaluate_model() with mocked run_inference."""

    def _make_wav_files(self, tmp_path, samples):
        """Create dummy WAV files for the given samples under tmp_path/testing-wav."""
        wav_dir = tmp_path / "testing-wav"
        for sample in samples:
            node_name_in_filename = sample.node_name.replace("_", "-")
            wav_filename = f"{node_name_in_filename}_{sample.start_timestamp}.wav"
            wav_file = wav_dir / sample.category / wav_filename
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            wav_file.touch()
        return wav_dir

    def test_correct_resident_prediction_counted(self, tmp_path):
        """A resident sample predicted as "resident" (fastai) counts as correct."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "resident", "rpi_orcasound_lab", "2023_08_18_00_59_53_PST")

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.5}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", wav_files, wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.skipped == 0

    def test_false_positive_counted(self, tmp_path):
        """A non-resident sample predicted as "resident" (fastai) counts as false positive."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "human", "rpi_sunset_bay", "2024_08_07_11_23_23_PST")

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.7, "predict_time": 1.2}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", wav_files, wav_dir)

        assert result.correct == 0
        assert result.false_positives == 1
        assert result.false_negatives == 0


    def test_fastai_other_prediction_correct_for_non_resident(self, tmp_path):
        """Binary models still count non-resident predicted as non-resident as correct."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "human", "rpi_sunset_bay", "2024_08_07_11_23_23_PST")

        mock_result = {"global_prediction_label": "other", "global_confidence": 0.7, "predict_time": 1.2}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", wav_files, wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_false_negative_counted(self, tmp_path):
        """A resident sample predicted as "other" (fastai) counts as false negative."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "resident", "rpi_orcasound_lab", "2023_08_18_00_59_53_PST")

        mock_result = {"global_prediction_label": "other", "global_confidence": 0.1, "predict_time": 1.0}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", wav_files, wav_dir)

        assert result.correct == 0
        assert result.false_positives == 0
        assert result.false_negatives == 1

    
    def test_skips_sample_on_inference_error(self, tmp_path):
        """Samples that raise an exception during inference are counted as skipped."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "resident", "rpi_orcasound_lab", "2023_08_18_00_59_53_PST")

        with patch("compare_models.run_inference", side_effect=RuntimeError("model error")):
            result = evaluate_model("fastai", "./model", wav_files, wav_dir)

        assert result.skipped == 1
        assert result.correct == 0

    def test_podsai_resident_prediction_correct(self, tmp_path):
        """PODS-AI "resident" prediction for a resident sample counts as correct."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "resident", "rpi_orcasound_lab", "2023_08_18_00_59_53_PST")

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.9, "predict_time": 2.0}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/path/to/model", wav_files, wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_podsai_non_matching_non_resident_prediction_not_correct(self, tmp_path):
        """PODS-AI uses exact category matches for Correct, even within non-resident classes."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "human", "rpi_sunset_bay", "2024_08_07_11_23_23_PST")

        mock_result = {"global_prediction_label": "water", "global_confidence": 0.8, "predict_time": 1.8}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/path/to/model", wav_files, wav_dir)

        assert result.correct == 0
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_podsai_exact_matching_category_counts_as_correct(self, tmp_path):
        """PODS-AI counts exact multiclass matches as correct."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "humpback", "rpi_sunset_bay", "2024_08_07_11_23_23_PST")

        mock_result = {"global_prediction_label": "humpback", "global_confidence": 0.8, "predict_time": 1.8}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/path/to/model", wav_files, wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_total_equals_sample_count(self, tmp_path):
        """ModelResult.total always equals the number of samples passed."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "resident", "rpi_orcasound_lab", "2023_08_18_00_59_53_PST")
        wav_files += _make_wav(wav_dir, "human", "rpi_sunset_bay", "2024_08_07_11_23_23_PST")

        mock_result = {"global_prediction_label": "other", "global_confidence": 0.1, "predict_time": 1.0}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", wav_files, wav_dir)

        assert result.total == 2

    def test_records_predict_times(self, tmp_path):
        """evaluate_model records predict_time from inference results."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "resident", "rpi_orcasound_lab", "2023_08_18_00_59_53_PST")

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 2.5}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", wav_files, wav_dir)

        assert len(result.predict_times) == 1
        assert abs(result.predict_times[0] - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# Tests for print_summary()
# ---------------------------------------------------------------------------

class TestPrintSummary:
    """Tests for print_summary() output formatting."""

    def test_prints_header_and_separator(self, capsys):
        """print_summary prints the "Model Comparison Summary" header."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, correct=4, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "Model Comparison Summary" in captured

    def test_prints_model_name(self, capsys):
        """print_summary includes the model type in the output."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="orcahello", total=5, correct=3, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "orcahello" in captured

    def test_prints_accuracy_percentage(self, capsys):
        """print_summary shows accuracy as a percentage."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=10, correct=8, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "80.0%" in captured

    def test_prints_na_when_no_evaluated_samples(self, capsys):
        """print_summary shows N/A for accuracy when all samples are skipped."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, skipped=5)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "N/A" in captured

    def test_prints_skipped_count_when_nonzero(self, capsys):
        """print_summary shows the skipped count when some samples were skipped."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=10, correct=5, skipped=3)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "3 skipped" in captured

    def test_prints_definitions(self, capsys):
        """print_summary includes the definitions block."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, correct=4, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "Definitions:" in captured
        assert "Accuracy     = Correct / Evaluated" in captured
        assert "[R|T|H]FP%   = among non-[R|T|H] samples, fraction predicted as that class" in captured
        assert "compares end-to-end 60-second inference on all WAV files in --wav-dir" in captured

    def test_prints_avg_time(self, capsys):
        """print_summary includes average time column."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, correct=4, skipped=0, predict_times=[1.5, 2.0, 1.8, 2.2, 1.7])]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "Avg Time" in captured

    def test_prints_whale_f1_column(self, capsys):
        """print_summary includes the whale-class F1 column."""
        from compare_models import ModelResult, print_summary
        results = [
            ModelResult(
                model_type="podsai",
                total=3,
                correct=2,
                confusion_matrix={
                    "resident": {"resident": 1},
                    "humpback": {"resident": 1},
                    "transient": {"transient": 1},
                },
            )
        ]
        print_summary(results)
        captured = capsys.readouterr().out
        assert " F1 " in captured
        assert "0.556" in captured

    def test_prints_per_whale_fp_fn_rate_columns(self, capsys):
        """print_summary includes resident, transient, and humpback FP%/FN% columns without counts."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=1, correct=1, confusion_matrix={"resident": {"resident": 1}})]
        print_summary(results)
        captured = capsys.readouterr().out
        for header in ("RFP%", "RFN%", "TFP%", "TFN%", "HFP%", "HFN%"):
            assert header in captured
        for header in (" RFP ", " RFN ", " TFP ", " TFN ", " HFP ", " HFN "):
            assert header not in captured

    def test_binary_model_non_resident_whale_rates_match_expected_summary(self, capsys):
        """fastai/orcahello show 0% FP and 100% FN rates for transient/humpback when present."""
        from compare_models import ModelResult, print_summary
        result = ModelResult(
            model_type="fastai",
            total=4,
            correct=1,
            confusion_matrix={
                "resident": {"resident": 1},
                "transient": {"other": 2},
                "humpback": {"other": 1},
            },
            predict_times=[1.0, 1.1, 1.2, 1.3],
        )
        print_summary([result])
        captured = capsys.readouterr().out
        fastai_line = next(line for line in captured.splitlines() if line.strip().startswith("fastai"))
        columns = fastai_line.split()
        assert columns[7:9] == ["0.0%", "100.0%"]
        assert columns[9:11] == ["0.0%", "100.0%"]


# ---------------------------------------------------------------------------
# Tests for main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    """Tests for the main() entry point."""

    def _write_testing_wavs(self, wav_dir: Path, rows):
        """Create WAV files from testing rows under category directories."""
        for row in rows:
            node = row["NodeName"].replace("_", "-")
            wav = wav_dir / row["Category"] / f"{node}_{row['StartTimestamp']}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.touch()

    def test_returns_1_for_missing_wav_dir(self, tmp_path):
        """main() returns 1 when the WAV directory does not exist."""
        from compare_models import main

        test_args = [
            "compare_models.py",
            "--wav-dir", str(tmp_path / "nonexistent-wav-dir"),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_rejects_removed_testing_csv_flag(self, tmp_path):
        """main() rejects the removed --testing-csv option."""
        from compare_models import main

        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--testing-csv", "output/csv/testing_60s_samples.csv",
            "--wav-dir", str(wav_dir),
        ]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2

    def test_returns_1_for_unknown_model(self, tmp_path):
        """main() returns 1 when an unrecognised model type is specified."""
        from compare_models import main

        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "unknown_model",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    @pytest.mark.parametrize(
        "model", ["fastai", "orcahello", "wav2vec2", "oldpodsai", "podsai", "custom"]
    )
    def test_returns_1_for_duplicate_model(self, tmp_path, capsys, model):
        """main() rejects duplicate model names with a model-specific error."""
        from compare_models import main

        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", f"{model},{model}",
        ]
        with patch.object(sys, "argv", test_args), \
             patch("compare_models.get_model_inference") as load_model:
            assert main() == 1

        assert f"{model} may be listed only once in --models" in capsys.readouterr().err
        load_model.assert_not_called()

    def test_returns_1_for_missing_custom_model_path(self, tmp_path):
        """main() rejects a missing path before attempting model loading."""
        from compare_models import main

        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, [_make_testing_rows()[0]])
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "custom",
            "--custom-model-path", str(tmp_path / "missing-model"),
        ]
        with patch.object(sys, "argv", test_args), \
             patch("compare_models.get_model_inference") as load_model:
            assert main() == 1
        load_model.assert_not_called()

    def test_returns_1_when_custom_name_has_no_path(self, tmp_path, capsys):
        """main() rejects custom without its required path option."""
        from compare_models import main

        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, [_make_testing_rows()[0]])
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "custom",
        ]
        with patch.object(sys, "argv", test_args), \
             patch("compare_models.get_model_inference") as load_model:
            assert main() == 1
        load_model.assert_not_called()
        assert "--custom-model-path" in capsys.readouterr().err

    def test_custom_path_auto_adds_custom_model(self, tmp_path):
        """main() adds custom when a path is supplied without the model name."""
        from compare_models import ModelResult, main

        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, [_make_testing_rows()[0]])
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--custom-model-path", "org/model",
        ]
        custom_model = type("Model", (), {"predict": lambda self, _: {}})()
        with patch.object(sys, "argv", test_args), \
             patch("compare_models.get_model_inference", return_value=custom_model), \
             patch("compare_models.evaluate_model", return_value=ModelResult("custom")) as evaluate:
            assert main() == 0
        assert evaluate.call_args.kwargs["result_model_type"] == "custom"

    def test_reports_custom_loader_failure_without_traceback(self, tmp_path, capsys):
        """main() converts a Hub loader failure into a concise CLI error."""
        from compare_models import main

        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, [_make_testing_rows()[0]])
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "custom",
            "--custom-model-path", "missing-org/missing-model",
        ]
        with patch.object(sys, "argv", test_args), \
             patch(
                 "compare_models.get_model_inference",
                 side_effect=RuntimeError("repository not found"),
             ):
            assert main() == 1

        captured = capsys.readouterr()
        assert "Error loading custom model 'missing-org/missing-model'" in captured.err
        assert "RuntimeError: repository not found" in captured.err
        assert "Traceback" not in captured.err

    def test_accepts_oldpodsai_model(self, tmp_path):
        """main() accepts oldpodsai and evaluates it using the podsai inference path."""
        from compare_models import main

        rows = _make_testing_rows()

        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, [rows[0]])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.5}
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "oldpodsai",
            "--max-samples", "1",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.get_model_inference", return_value=object()):
                with patch("compare_models.run_inference", return_value=mock_result) as mock_infer:
                    result = main()

        assert result == 0
        mock_infer.assert_called_once()
        assert mock_infer.call_args.kwargs["model_type"] == "podsai"
        assert mock_infer.call_args.kwargs["model_revision"] == PREVIOUS_PODSAI_MODEL_REVISION

    def test_oldpodsai_per_file_output_uses_oldpodsai_label(self, tmp_path, capsys):
        """Per-file output for oldpodsai should be labeled [oldpodsai], not [podsai]."""
        from compare_models import main

        rows = _make_testing_rows()
        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, [rows[0]])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.5}
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "oldpodsai",
            "--max-samples", "1",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.get_model_inference", return_value=object()):
                with patch("compare_models.run_inference", return_value=mock_result):
                    result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "[oldpodsai]" in captured.out
        assert "[podsai]" not in captured.out

    def test_default_models_include_oldpodsai(self, tmp_path):
        """main() defaults to evaluating fastai, orcahello, wav2vec2, oldpodsai, and podsai."""
        from compare_models import ModelResult, main

        rows = _make_testing_rows()
        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, [rows[0]])

        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--max-samples", "1",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.get_model_inference", return_value=object()):
                with patch("compare_models.evaluate_model", side_effect=lambda **kwargs: ModelResult(
                    model_type=kwargs["model_type"],
                    total=1,
                    skipped=1,
                )) as mock_evaluate:
                    result = main()

        assert result == 0
        assert mock_evaluate.call_count == 5
        # Order matches default models: fastai, orcahello, wav2vec2, oldpodsai, podsai.
        called_revisions = [call.kwargs["model_revision"] for call in mock_evaluate.call_args_list]
        assert called_revisions == [None, None, PODSAI_WAV2VEC2_MODEL_REVISION, PREVIOUS_PODSAI_MODEL_REVISION, LATEST_PODSAI_MODEL_REVISION]

    def test_reuses_preloaded_model_for_each_sample(self, tmp_path):
        """main() preloads one model per type and reuses it for all samples."""
        from compare_models import main

        rows = _make_testing_rows()
        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, rows)

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.5}
        preloaded_model = object()
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--max-samples", "2",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.get_model_inference", return_value=preloaded_model) as mock_get_model:
                with patch("compare_models.run_inference", return_value=mock_result) as mock_infer:
                    result = main()

        assert result == 0
        mock_get_model.assert_called_once()
        assert mock_infer.call_count == 2
        assert all(
            call.kwargs["inference_model"] is preloaded_model
            for call in mock_infer.call_args_list
        )

    def test_returns_0_on_success_with_fastai(self, tmp_path):
        """main() returns 0 when it successfully evaluates fastai on test samples."""
        from compare_models import main

        rows = _make_testing_rows()
        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, rows)

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.5}
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--fastai-model-path", "./model",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.get_model_inference", return_value=object()):
                with patch("compare_models.run_inference", return_value=mock_result):
                    result = main()
        assert result == 0

    def test_returns_1_when_no_test_samples(self, tmp_path):
        """main() returns 1 when wav_dir contains no valid test samples."""
        from compare_models import main

        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_respects_max_samples_argument(self, tmp_path):
        """main() respects --max-samples argument."""
        from compare_models import main

        rows = _make_testing_rows()
        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, rows[:2])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.5}
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--max-samples", "2",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.get_model_inference", return_value=object()):
                with patch("compare_models.run_inference", return_value=mock_result) as mock_infer:
                    result = main()
                    # Should only call inference twice (max 2 samples)
                    assert mock_infer.call_count == 2
        assert result == 0

    def test_returns_1_for_invalid_max_samples(self, tmp_path):
        """main() returns 1 when --max-samples is zero or negative."""
        from compare_models import main

        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--max-samples", "0",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_category_filter_returns_only_matching_samples(self, tmp_path):
        """main() --category filters samples to the specified category."""
        from compare_models import main

        rows = _make_testing_rows()
        wav_dir = tmp_path / "testing-wav"
        # Only create WAV for the resident sample.
        self._write_testing_wavs(wav_dir, [rows[0]])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.9, "predict_time": 1.0}
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--fastai-model-path", "./model",
            "--category", "resident",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.get_model_inference", return_value=object()):
                with patch("compare_models.run_inference", return_value=mock_result) as mock_infer:
                    result = main()
                    # Only the one resident sample should be evaluated.
                    assert mock_infer.call_count == 1
        assert result == 0

    def test_returns_1_for_category_with_no_samples(self, tmp_path):
        """main() returns 1 when --category matches no WAV-discovered samples."""
        from compare_models import main

        wav_dir = tmp_path / "testing-wav"
        self._write_testing_wavs(wav_dir, _make_testing_rows())
        test_args = [
            "compare_models.py",
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--category", "transient",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1


# ---------------------------------------------------------------------------
# Tests for ModelResult confusion_matrix tracking
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    """Tests for per-class confusion matrix tracking in ModelResult and evaluate_model()."""

    def test_confusion_matrix_populated_on_correct_prediction(self, tmp_path):
        """evaluate_model records actual→predicted in confusion_matrix for correct predictions."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "resident")

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.9, "predict_time": 1.0}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/model", wav_files, wav_dir)

        assert result.confusion_matrix == {"resident": {"resident": 1}}

    def test_multilabel_metrics_count_secondary_species_as_true_positive(self, tmp_path):
        """F1 and false-negative rates use all labels, not only the primary view."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "transient")
        mock_result = {
            "global_prediction_label": "resident",
            "global_prediction_labels": ["resident", "transient"],
            "global_confidence": 0.8,
            "predict_time": 1.0,
        }
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/model", wav_files, wav_dir)

        assert result.correct == 1
        assert result.false_negative_count_for_label("transient") == 0
        assert result.false_positive_count_for_label("resident") == 1
        assert result.whale_f1 == 0.5

    def test_confusion_matrix_populated_on_false_positive(self, tmp_path):
        """evaluate_model records actual→predicted in confusion_matrix for false positives."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "humpback")

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8, "predict_time": 1.0}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/model", wav_files, wav_dir)

        assert result.confusion_matrix == {"humpback": {"resident": 1}}

    def test_confusion_matrix_not_updated_for_skipped_samples(self, tmp_path):
        """Skipped samples (missing WAV) do not appear in the confusion matrix."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        wav_file = wav_dir / "resident/nonexistent.wav"

        with patch("compare_models.run_inference") as mock_infer:
            mock_infer.side_effect = FileNotFoundError("missing wav")
            result = evaluate_model("fastai", "./model", [wav_file], wav_dir)

        assert result.skipped == 1
        assert result.confusion_matrix == {}

    def test_confusion_matrix_accumulates_multiple_samples(self, tmp_path):
        """evaluate_model accumulates counts across multiple samples."""
        from compare_models import evaluate_model

        wav_dir = tmp_path / "testing-wav"
        wav_files = _make_wav(wav_dir, "resident", "rpi_orcasound_lab", "2023_08_18_00_59_53_PST")
        wav_files += _make_wav(wav_dir, "resident", "rpi_orcasound_lab", "2023_08_19_00_00_00_PST")
        wav_files += _make_wav(wav_dir, "humpback", "rpi_sunset_bay", "2023_08_20_00_00_00_PST")

        def fake_infer(wav_path, model_type, model_path, model_revision=None, inference_model=None):
            if "humpback" in str(wav_path):
                return {"global_prediction_label": "water", "global_confidence": 0.7, "predict_time": 1.0}
            return {"global_prediction_label": "resident", "global_confidence": 0.9, "predict_time": 1.0}

        with patch("compare_models.run_inference", side_effect=fake_infer):
            result = evaluate_model("podsai", "/model", wav_files, wav_dir)

        assert result.confusion_matrix["resident"]["resident"] == 2
        assert result.confusion_matrix["humpback"]["water"] == 1

    def test_confusion_matrix_empty_when_no_matrix(self, capsys):
        """print_confusion_matrix does nothing when confusion_matrix is empty."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(model_type="fastai")
        print_confusion_matrix(result)
        captured = capsys.readouterr().out
        assert captured == ""

    def test_print_confusion_matrix_contains_labels(self, capsys):
        """print_confusion_matrix includes all seen labels in header and rows."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="podsai",
            confusion_matrix={
                "resident": {"resident": 5, "water": 1},
                "humpback": {"water": 3, "humpback": 2},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        assert "resident" in captured
        assert "humpback" in captured
        assert "water" in captured

    def test_print_confusion_matrix_shows_correct_counts(self, capsys):
        """print_confusion_matrix displays the right numeric values."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="fastai",
            confusion_matrix={
                "resident": {"resident": 7, "other": 2},
                "other": {"resident": 1, "other": 9},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        assert "7" in captured
        assert "2" in captured
        assert "1" in captured
        assert "9" in captured

    def test_print_confusion_matrix_includes_total_column(self, capsys):
        """print_confusion_matrix appends a total column with per-row totals."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="fastai",
            confusion_matrix={
                "resident": {"resident": 7, "other": 2},
                "other": {"resident": 1, "other": 9},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        lines = captured.splitlines()
        header_line = next(line for line in lines if "other" in line and "resident" in line)
        resident_line = next(line for line in lines if line.startswith("  resident"))
        other_line = next(line for line in lines if line.startswith("     other"))
        assert "total" in header_line
        assert resident_line.split()[-1] == "9"
        assert other_line.split()[-1] == "10"

    def test_print_confusion_matrix_zero_for_unseen_pairs(self, capsys):
        """print_confusion_matrix shows 0 for class pairs that never occurred."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="podsai",
            confusion_matrix={
                "resident": {"resident": 3},
                "humpback": {"humpback": 4},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        # "resident" predicted as "humpback" should be 0, shown somewhere.
        lines = [line for line in captured.splitlines() if "resident" in line and not line.strip().startswith("Confusion")]
        # The resident row should contain a zero for the humpback column.
        assert any("0" in line for line in lines)

    def test_print_confusion_matrix_omits_all_zero_rows(self, capsys):
        """print_confusion_matrix omits rows where every predicted count is zero."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="fastai",
            confusion_matrix={
                "resident": {"other": 9, "resident": 1},
                "other": {},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        # The "other" actual row is all-zero and must not appear as a row label.
        lines = captured.splitlines()
        row_lines = [line for line in lines if not line.strip().startswith("Confusion") and line.strip()]
        row_labels = [line.split()[0] for line in row_lines[1:]]  # skip header line
        assert "other" not in row_labels

    def test_print_confusion_matrix_omits_all_zero_columns(self, capsys):
        """print_confusion_matrix omits columns where every count across all rows is zero."""
        from compare_models import ModelResult, print_confusion_matrix

        result = ModelResult(
            model_type="fastai",
            confusion_matrix={
                "human": {"other": 1, "resident": 1},
                "resident": {"other": 9, "resident": 1},
            },
        )
        print_confusion_matrix(result)
        captured = capsys.readouterr().out

        # Only "other" and "resident" columns were ever predicted; "human" must not appear.
        header_line = [line for line in captured.splitlines() if "other" in line and "resident" in line][0]
        assert "human" not in header_line

    def test_print_summary_includes_confusion_matrices(self, capsys):
        """print_summary prints confusion matrices for each model after the table."""
        from compare_models import ModelResult, print_summary

        results = [
            ModelResult(
                model_type="fastai",
                total=2,
                correct=2,
                confusion_matrix={"resident": {"resident": 2}},
            ),
        ]
        print_summary(results)
        captured = capsys.readouterr().out

        assert "Confusion Matrix for fastai" in captured
        assert "resident" in captured
