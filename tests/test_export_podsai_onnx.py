# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import export_podsai_onnx


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def float(self):
        return self

    def to(self, _device):
        return self


def test_export_uses_cli_model_revision_and_batch_size(monkeypatch):
    model = Mock()
    model.audio_spectrogram_transformer = None
    feature_extractor = SimpleNamespace(
        max_length=1024,
        num_mel_bins=7,
        hop_length=150,
    )
    inference = SimpleNamespace(
        model=model,
        feature_extractor=feature_extractor,
        _ensure_ast_position_embeddings=Mock(return_value=True),
        _use_spectrogram_input=True,
    )
    inference_factory = Mock(return_value=inference)
    monkeypatch.setitem(
        sys.modules,
        "podsai_inference",
        SimpleNamespace(PodsAIInference=inference_factory),
    )

    class FakeModule:
        def __init__(self):
            pass

        def to(self, _device):
            return self

    export = Mock()
    fake_torch = SimpleNamespace(
        device=lambda value: value,
        float32=np.float32,
        nn=SimpleNamespace(Module=FakeModule),
        randn=lambda *shape, **_kwargs: FakeTensor(np.zeros(shape, dtype=np.float32)),
        onnx=SimpleNamespace(export=export),
    )
    monkeypatch.setattr(export_podsai_onnx, "torch", fake_torch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_podsai_onnx.py",
            "--model",
            "example/model",
            "--revision",
            "0123456789abcdef",
            "--out",
            "model.onnx",
            "--batch-size",
            "3",
            "--frames",
            "20",
            "--mel-bins",
            "7",
        ],
    )

    export_podsai_onnx.main()

    inference_factory.assert_called_once_with(
        "example/model", device="cpu", model_revision="0123456789abcdef"
    )
    inference._ensure_ast_position_embeddings.assert_called_once_with(20, 7)
    assert export.call_args.args[1][0].array.shape == (3, 20, 7)
    assert export.call_args.args[2] == "model.onnx"
    assert export.call_args.kwargs["dynamic_axes"] == {
        "input_values": {0: "batch"},
        "logits": {0: "batch"},
    }


def test_export_rejects_raw_audio_model(monkeypatch):
    inference = SimpleNamespace(
        model=Mock(),
        _use_spectrogram_input=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "podsai_inference",
        SimpleNamespace(PodsAIInference=Mock(return_value=inference)),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["export_podsai_onnx.py", "--model", "example/wav2vec2"],
    )
    monkeypatch.setattr(
        export_podsai_onnx,
        "torch",
        SimpleNamespace(device=lambda value: value),
    )

    with pytest.raises(ValueError, match="only spectrogram-based models"):
        export_podsai_onnx.main()


def test_export_rejects_dimensions_incompatible_with_runtime(monkeypatch):
    inference = SimpleNamespace(
        model=Mock(),
        feature_extractor=SimpleNamespace(
            max_length=1024,
            num_mel_bins=128,
            hop_length=10,
        ),
        _ensure_ast_position_embeddings=Mock(return_value=True),
        _use_spectrogram_input=True,
    )
    monkeypatch.setitem(
        sys.modules,
        "podsai_inference",
        SimpleNamespace(PodsAIInference=Mock(return_value=inference)),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["export_podsai_onnx.py", "--frames", "301"],
    )
    monkeypatch.setattr(
        export_podsai_onnx,
        "torch",
        SimpleNamespace(device=lambda value: value),
    )

    with pytest.raises(ValueError, match="expected --frames 300"):
        export_podsai_onnx.main()


def test_export_uses_max_length_when_position_embeddings_cannot_resize(monkeypatch):
    inference = SimpleNamespace(
        model=Mock(),
        feature_extractor=SimpleNamespace(
            max_length=1024,
            num_mel_bins=128,
            hop_length=10,
        ),
        _ensure_ast_position_embeddings=Mock(return_value=False),
        _use_spectrogram_input=True,
    )
    monkeypatch.setitem(
        sys.modules,
        "podsai_inference",
        SimpleNamespace(PodsAIInference=Mock(return_value=inference)),
    )

    class FakeModule:
        def to(self, _device):
            return self

    export = Mock()
    monkeypatch.setattr(
        export_podsai_onnx,
        "torch",
        SimpleNamespace(
            device=lambda value: value,
            float32=np.float32,
            nn=SimpleNamespace(Module=FakeModule),
            randn=lambda *shape, **_kwargs: FakeTensor(
                np.zeros(shape, dtype=np.float32)
            ),
            onnx=SimpleNamespace(export=export),
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["export_podsai_onnx.py", "--frames", "300", "--mel-bins", "128"],
    )

    with pytest.raises(ValueError, match="expected --frames 1024"):
        export_podsai_onnx.main()

    inference._ensure_ast_position_embeddings.assert_called_once_with(300, 128)
    monkeypatch.setattr(
        "sys.argv",
        ["export_podsai_onnx.py", "--frames", "1024", "--mel-bins", "128"],
    )

    export_podsai_onnx.main()

    assert inference._ensure_ast_position_embeddings.call_count == 2
    assert export.call_args.args[1][0].shape == (1, 1024, 128)
