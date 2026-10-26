# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from validate_podsai_onnx import validate_outputs


def test_validate_outputs_accepts_matching_logits_and_top_k():
    logits = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])

    assert validate_outputs(logits, logits + 1e-6) == pytest.approx(1e-6)


def test_validate_outputs_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        validate_outputs(np.zeros((1, 3)), np.zeros((1, 2)))


def test_validate_outputs_rejects_logits_outside_tolerance():
    with pytest.raises(ValueError, match="exceed tolerance"):
        validate_outputs(np.array([[0.0, 1.0]]), np.array([[0.0, 1.01]]))


def test_validate_outputs_rejects_top_k_mismatch():
    torch_out = np.array([[1.0, 0.5, 0.0]])
    onnx_out = np.array([[0.5, 1.0, 0.0]])

    with pytest.raises(ValueError, match="Top-2 predictions differ"):
        validate_outputs(torch_out, onnx_out, top_k=2, rtol=1.0, atol=1.0)
