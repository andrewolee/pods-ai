#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import wave

import numpy as np
import soundfile as sf


def test_orcahello_factory_forwards_model_revision():
    """The generic model factory forwards revisions to the OrcaHello loader."""
    from model_inference import get_model_inference

    revision = "0123456789abcdef0123456789abcdef01234567"
    with patch("orcahello_inference.get_orcahello_srkw_inference") as factory:
        get_model_inference(
            model_path="orcasound/orcahello-srkw-detector-v1",
            model_type="orcahello",
            model_revision=revision,
        )

    factory.assert_called_once_with(
        "orcasound/orcahello-srkw-detector-v1", model_revision=revision
    )


def test_onnx_factory_forwards_extractor_and_revision():
    """The ONNX backend receives the extractor and HF revision options."""
    from model_inference import get_model_inference

    factory = MagicMock()
    with patch.dict(
        "sys.modules",
        {"podsai_onnx_inference": SimpleNamespace(
            get_podsai_onnx_inference=factory
        )},
    ):
        get_model_inference(
            model_path="model.onnx",
            model_type="podsai",
            feature_extractor="example/model",
            model_revision="0123456789abcdef",
        )

    factory.assert_called_once_with(
        "model.onnx",
        feature_extractor="example/model",
        model_revision="0123456789abcdef",
    )


def test_orcahello_loader_passes_revision_only_for_hub_models(tmp_path):
    """Only Hub model loads receive a requested revision."""
    from orcahello_inference import OrcaHelloSRKWInference

    calls = []

    class FakeDetector:
        """Minimal detector stub for testing model loading arguments."""

        _device = "cpu"
        _dtype = "float32"

        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls.append((model_path, kwargs))
            return cls()

        def eval(self):
            return self

    class FakeConfig:
        """Minimal inference configuration stub."""

        @classmethod
        def from_dict(cls, values):
            return SimpleNamespace(as_dict=lambda: values)

    revision = "0123456789abcdef0123456789abcdef01234567"
    with patch(
        "orcahello_inference._get_orcahello_classes",
        return_value=(FakeDetector, FakeConfig),
    ):
        OrcaHelloSRKWInference("orcasound/model", model_revision=revision)

        local_model = tmp_path / "local-model"
        local_model.mkdir()
        OrcaHelloSRKWInference(str(local_model), model_revision=revision)
        OrcaHelloSRKWInference("orcasound/model")

    assert calls[0] == ("orcasound/model", {"config": {}, "revision": revision})
    assert calls[1] == (str(local_model), {"config": {}})
    assert calls[2] == ("orcasound/model", {"config": {}})


def test_export_wave_file_closes_output_before_decoding(tmp_path):
    """An exported segment is fully readable before the next pipeline stage."""
    from model_inference import export_wave_file
    from pydub import AudioSegment

    source = tmp_path / "source.wav"
    destination = tmp_path / "segment.wav"
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000)

    export_wave_file(AudioSegment.from_wav(source), 0, 1, destination)
    samples, sample_rate = sf.read(destination)

    assert sample_rate == 16000
    assert len(samples) == 16000


def test_fastai_predict_with_empty_valid_loader(tmp_path):
    """FastAI model prediction works when split_none() is used."""
    from model_inference import FastAIModel

    positive_confidences = [0.25, 0.75]

    wav_path = tmp_path / "example.wav"
    # Create a 4-second audio file at 16kHz (64000 samples).
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 64000)

    def fake_extract_segments(_audio_path, sample_dict, dest_dir, _suffix):
        for wav_name, segments in sample_dict.items():
            stem = Path(wav_name).stem.lower()
            for begin_time, end_time in segments:
                segment_path = Path(dest_dir) / f"{stem}_{begin_time}_{end_time}.wav"
                # Create 3-second segments at 16kHz (48000 samples).
                with wave.open(str(segment_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(b"\x00\x00" * 48000)

    mock_loaded_model = MagicMock()
    mock_loaded_model.model = MagicMock()
    mock_loaded_model.predict.side_effect = [
        (None, None, [0.0, positive_confidences[0]]),
        (None, None, [0.0, positive_confidences[1]]),
    ]

    score_paths = [
        Path(tmp_path / "example_0_3.wav"),
        Path(tmp_path / "example_1_4.wav"),
    ]
    score_items = MagicMock()
    score_items.items = score_paths
    score_items.__iter__.return_value = iter(score_paths)

    # When split_none() is used, items are in testdb.x (not in a separate validation set).
    fake_testdb = SimpleNamespace(
        x=score_items,  # Items to score are here.
        train_ds=SimpleNamespace(x=score_items),
    )

    fake_test = MagicMock()
    fake_test.transform.return_value.databunch.return_value = fake_testdb

    with patch("model_inference.load_model", return_value=mock_loaded_model), \
         patch("model_inference.get_duration", return_value=4.0), \
         patch("model_inference.extract_segments", side_effect=fake_extract_segments) as mock_extract_segments, \
         patch("model_inference.gc.collect"), \
         patch("model_inference.torch.cuda.is_available", return_value=False), \
         patch("audio.data.AudioList") as mock_audio_list:
        # Mock AudioList constructor to return a mock that chains to fake_test.
        mock_audio_list.return_value.split_none.return_value.label_empty.return_value = fake_test

        model = FastAIModel(
            model_path="model",
            smooth_predictions=False,
            min_num_positive_calls_threshold=1,
        )
        result = model.predict(str(wav_path))

    assert result["local_confidences"] == positive_confidences
    assert result["local_predictions"] == [0, 1]
    assert result["global_prediction"] == 1
    assert result["submission"]["start_time_s"].tolist() == [0, 1]
    assert mock_loaded_model.predict.call_count == 2
    mock_loaded_model.predict.assert_any_call(score_paths[0])
    mock_loaded_model.predict.assert_any_call(score_paths[1])
    mock_extract_segments.assert_called_once()


def test_fastai_predict_prefers_populated_testdb_x(tmp_path):
    """Use testdb.x when it already exposes generated clips."""
    from model_inference import FastAIModel

    positive_confidences = [0.25, 0.75]

    wav_path = tmp_path / "example.wav"
    # Create a 4-second audio file at 16kHz (64000 samples).
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 64000)

    def fake_extract_segments(_audio_path, sample_dict, dest_dir, _suffix):
        for wav_name, segments in sample_dict.items():
            stem = Path(wav_name).stem.lower()
            for begin_time, end_time in segments:
                segment_path = Path(dest_dir) / f"{stem}_{begin_time}_{end_time}.wav"
                # Create 3-second segments at 16kHz (48000 samples).
                with wave.open(str(segment_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(b"\x00\x00" * 48000)

    mock_loaded_model = MagicMock()
    mock_loaded_model.model = MagicMock()
    mock_loaded_model.predict.side_effect = [
        (None, None, [0.0, positive_confidences[0]]),
        (None, None, [0.0, positive_confidences[1]]),
    ]

    score_paths = [
        Path(tmp_path / "example_0_3.wav"),
        Path(tmp_path / "example_1_4.wav"),
    ]
    score_items = MagicMock()
    score_items.items = score_paths
    score_items.__iter__.return_value = iter(score_paths)
    score_items.__len__.return_value = len(score_paths)

    empty_score_items = MagicMock()
    empty_score_items.items = []
    empty_score_items.__iter__.return_value = iter([])
    empty_score_items.__len__.return_value = 0

    fake_testdb = SimpleNamespace(
        x=score_items,
        train_ds=SimpleNamespace(x=empty_score_items),
    )

    fake_test = MagicMock()
    fake_test.transform.return_value.databunch.return_value = fake_testdb

    with patch("model_inference.load_model", return_value=mock_loaded_model), \
         patch("model_inference.get_duration", return_value=4.0), \
         patch("model_inference.extract_segments", side_effect=fake_extract_segments), \
         patch("model_inference.gc.collect"), \
         patch("model_inference.torch.cuda.is_available", return_value=False), \
         patch("audio.data.AudioList") as mock_audio_list:
        # Mock AudioList constructor to return a mock that chains to fake_test.
        mock_audio_list.return_value.split_none.return_value.label_empty.return_value = fake_test

        model = FastAIModel(
            model_path="model",
            smooth_predictions=False,
            min_num_positive_calls_threshold=1,
        )
        result = model.predict(str(wav_path))

    assert result["local_confidences"] == positive_confidences
    assert result["local_predictions"] == [0, 1]
    assert result["global_prediction"] == 1
    assert result["submission"]["start_time_s"].tolist() == [0, 1]
    mock_loaded_model.predict.assert_any_call(score_paths[0])
    mock_loaded_model.predict.assert_any_call(score_paths[1])
