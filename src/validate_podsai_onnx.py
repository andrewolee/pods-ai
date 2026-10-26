#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Validate ONNX export against PyTorch model and benchmark runtimes.

Usage:
  python src/validate_podsai_onnx.py --onnx podsai.onnx --model davethaler/whale-call-detector --revision <revision> --wav path/to/sample.wav

This will:
 - Load a sample WAV and compute input_values via PodsAIInference._compute_input_values
 - Run the HF model forward and the ONNX session forward and compare logits
 - Require logits to match within rtol=1e-4 and atol=1e-5, with matching top-5 classes
 - Report timings for PyTorch and ONNX runs
"""
import argparse
import time
import numpy as np
import torch


def compute_input_values_from_wav(wav_path: str, model_path: str, model_revision: str = None):
    # Reuse PodsAIInference helper by loading the class
    from podsai_inference import PodsAIInference
    inf = PodsAIInference(model_path, device='cpu', model_revision=model_revision)
    # Load wav and compute input_values for a single sample using default segment/hop
    import librosa
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    # compute number of positions for default segment/hop
    segment_samples = 3 * sr
    hop_samples = 2 * sr
    audio_duration = len(audio) / sr
    num_positions = int(np.floor((audio_duration - 3) / 2)) + 1 if audio_duration >= 3 else 1
    input_values = inf._compute_input_values(audio, sr, num_positions, hop_samples, segment_samples)
    # Input shape produced: (num_positions, frames, mel_bins)
    # For export/onnx we will use a single batch of all positions flattened as batch dimension
    return input_values, inf.model


def run_onnx(onnx_path: str, input_np: np.ndarray, runs: int = 3):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    # input_np shape should be (batch, frames, mel)
    times = []
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        res = sess.run(None, {input_name: input_np.astype(np.float32)})
        t1 = time.perf_counter()
        times.append(t1 - t0)
        out = res[0]
    return out, np.mean(times), np.std(times)


def run_torch(model, input_tensor: torch.Tensor, runs: int = 3):
    times = []
    out = None
    with torch.inference_mode():
        for _ in range(runs):
            t0 = time.perf_counter()
            res = model(input_values=input_tensor)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            out = res.logits.detach().cpu().numpy()
    return out, np.mean(times), np.std(times)


def validate_outputs(
    torch_out: np.ndarray,
    onnx_out: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    top_k: int = 5,
) -> float:
    """Validate ONNX logits against PyTorch using numeric and top-k checks."""
    if torch_out.shape != onnx_out.shape:
        raise ValueError(
            f"Shape mismatch: torch {torch_out.shape} vs onnx {onnx_out.shape}"
        )
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    if not np.allclose(torch_out, onnx_out, rtol=rtol, atol=atol):
        raise ValueError(
            f"Logits exceed tolerance (rtol={rtol}, atol={atol}); "
            f"max absolute difference: {max_abs_diff:.6f}"
        )

    compared_k = min(top_k, torch_out.shape[-1])
    torch_top_k = np.argsort(torch_out, axis=-1)[:, -compared_k:][:, ::-1]
    onnx_top_k = np.argsort(onnx_out, axis=-1)[:, -compared_k:][:, ::-1]
    if not np.array_equal(torch_top_k, onnx_top_k):
        raise ValueError(
            f"Top-{compared_k} predictions differ: "
            f"torch {torch_top_k.tolist()} vs onnx {onnx_top_k.tolist()}"
        )
    return max_abs_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--model", default="davethaler/whale-call-detector")
    parser.add_argument("--revision", default=None, help="HuggingFace model revision")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--rtol", type=float, default=1e-4, help="Logit relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-5, help="Logit absolute tolerance")
    parser.add_argument("--top-k", type=int, default=5, help="Number of leading classes to compare")
    args = parser.parse_args()

    print("Computing input values from WAV using PodsAIInference helper (may load HF config)...")
    input_values, torch_model = compute_input_values_from_wav(
        args.wav, args.model, model_revision=args.revision
    )

    # Collapse positions into batch dimension
    inp = input_values.detach().cpu().numpy()
    batch_np = inp  # shape (num_positions, frames, mel)
    print(f"Prepared input batch shape for ONNX: {batch_np.shape}")

    # Run a single forward on torch
    import torch
    input_tensor = torch.from_numpy(batch_np).float()
    print("Running PyTorch forward... (3 runs)")
    torch_out, torch_mean, torch_sd = run_torch(torch_model, input_tensor, runs=3)
    print(f"PyTorch logits shape: {torch_out.shape}, mean time: {torch_mean:.3f}s (sd {torch_sd:.3f})")

    print("Running ONNX forward... (3 runs)")
    onnx_out, onnx_mean, onnx_sd = run_onnx(args.onnx, batch_np, runs=3)
    print(f"ONNX logits shape: {onnx_out.shape}, mean time: {onnx_mean:.3f}s (sd {onnx_sd:.3f})")

    diff = validate_outputs(
        torch_out, onnx_out, rtol=args.rtol, atol=args.atol, top_k=args.top_k
    )
    print(f"ONNX validation passed; max abs difference: {diff:.6f}")


if __name__ == '__main__':
    main()
