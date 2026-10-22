#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""ONNX Runtime-backed PODS-AI inference wrapper.

This provides a lightweight replacement for PodsAIInference that runs the
core model forward using an ONNX InferenceSession. It reuses the same
preprocessing (mel fbank windows) as PodsAIInference by loading the
feature-extractor configuration from a HuggingFace model id or local folder.
The exported model must have fixed spectrogram dimensions; prediction rejects
segment durations whose preprocessed inputs do not match those dimensions.

Usage:
  inf = get_podsai_onnx_inference(onnx_path='podsai.onnx', feature_extractor='davethaler/whale-call-detector')
  result = inf.predict('path/to/sample.wav')
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from model_inference import ModelInference


def _count_non_adjacent_positive_events(positive_mask: list[bool]) -> int:
    events = 0
    run_length = 0
    for value in positive_mask:
        if value:
            run_length += 1
        elif run_length:
            events += (run_length + 1) // 2
            run_length = 0
    if run_length:
        events += (run_length + 1) // 2
    return events


def _positive_event_ids(positive_mask: list[bool]) -> list[Optional[int]]:
    event_ids = []
    event_id = -1
    run_length = 0
    for value in positive_mask:
        if not value:
            run_length = 0
            event_ids.append(None)
            continue
        if run_length % 2 == 0:
            event_id += 1
        event_ids.append(event_id)
        run_length += 1
    return event_ids


class PodsAIOnnxInference(ModelInference):
    def __init__(
        self,
        onnx_path: str,
        feature_extractor: Optional[str] = None,
        threshold: float = 0.5,
        min_num_positive_calls_threshold: int = 2,
        inference_batch_size: int = 8,
        ort_intra_threads: Optional[int] = None,
        ort_inter_threads: Optional[int] = None,
        model_revision: Optional[str] = None,
    ) -> None:
        super().__init__(onnx_path)
        try:
            import onnxruntime as ort
            from transformers import AutoFeatureExtractor, AutoConfig
        except Exception as e:
            raise RuntimeError(f"Required packages missing for ONNX inference: {e}")

        self.onnx_path = str(onnx_path)
        self.threshold = float(threshold)
        self.min_num_positive_calls_threshold = int(min_num_positive_calls_threshold)
        self.inference_batch_size = int(inference_batch_size)

        # Load feature extractor/config to obtain spectrogram params and labels.
        if not feature_extractor:
            raise ValueError(
                "feature_extractor must be provided (HF model id or local path) when using ONNX model"
            )

        pretrained_kwargs = {}
        if not Path(feature_extractor).exists():
            if model_revision:
                pretrained_kwargs["revision"] = model_revision
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                pretrained_kwargs["token"] = hf_token

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor, **pretrained_kwargs
        )
        cfg = AutoConfig.from_pretrained(feature_extractor, **pretrained_kwargs)
        # Build id2label/label2id mapping from config if present.
        raw_id2label = getattr(cfg, "id2label", None) or {}
        raw_label2id = getattr(cfg, "label2id", None) or {}
        # id2label may be either {"0": "water"} or {0: "water"}; normalize to int keys
        if raw_id2label:
            try:
                self.id2label = {int(k): v for k, v in raw_id2label.items()}
            except Exception:
                # Already int-keyed or unexpected format
                self.id2label = {k: v for k, v in raw_id2label.items()}
        else:
            self.id2label = {}

        # label2id may be either {"0": "water"} reversed or {"water": 0}; normalize to name->id mapping
        self.label2id = {}
        if raw_label2id:
            # If keys look like numeric strings mapping to labels, invert
            try:
                # Try to detect numeric-string keys
                if all(str(k).isdigit() for k in raw_label2id.keys()):
                    # raw_label2id: {"0": "water"} -> want {"water": 0}
                    for k, v in raw_label2id.items():
                        self.label2id[v] = int(k)
                else:
                    # raw_label2id: {"water": 0}
                    for k, v in raw_label2id.items():
                        self.label2id[k] = int(v)
            except Exception:
                # Fallback: leave empty
                self.label2id = {}

        self.negative_label_names = {
            "abiotic", "bird", "human", "jingle", "other", "vessel", "water"
        }
        self.negative_class_ids = {
            self.label2id[name]
            for name in self.negative_label_names
            if name in self.label2id
        }
        if not self.negative_class_ids:
            raise ValueError(
                "Model must include at least one negative/background class"
            )

        # Create ONNX Runtime session with CPU provider.
        sess_opts = ort.SessionOptions()
        if ort_intra_threads:
            sess_opts.intra_op_num_threads = int(ort_intra_threads)
        if ort_inter_threads:
            sess_opts.inter_op_num_threads = int(ort_inter_threads)

        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.onnx_path, sess_options=sess_opts, providers=providers)
        # Input name expected by exported model.
        model_input = self.sess.get_inputs()[0]
        self._input_name = model_input.name
        self._input_shape = model_input.shape

    def _compute_input_values(self, audio: np.ndarray, sr: int, num_positions: int, hop_samples: int, segment_samples: int):
        """Compute mel fbank windows replicating PodsAIInference behavior."""
        import torch
        import torchaudio

        num_mel_bins: int = getattr(self.feature_extractor, 'num_mel_bins', 128)
        max_length: int = getattr(self.feature_extractor, 'max_length', 1024)
        fe_mean: float = getattr(self.feature_extractor, 'mean', -4.2677393)
        fe_std: float = getattr(self.feature_extractor, 'std', 4.5689974)
        do_normalize: bool = getattr(self.feature_extractor, 'do_normalize', True)
        frame_shift_ms: float = getattr(self.feature_extractor, 'hop_length', 10)

        frames_per_second = 1000.0 / frame_shift_ms
        hop_frames = round(hop_samples / sr * frames_per_second)
        seg_frames = round(segment_samples / sr * frames_per_second)
        target_frames = self._get_target_frames(seg_frames, max_length)

        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        full_fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_shift=frame_shift_ms,
        )

        windows = []
        for pos_idx in range(num_positions):
            start_frame = pos_idx * hop_frames
            end_frame = start_frame + seg_frames
            window = full_fbank[start_frame:end_frame, :]
            window = window - window.mean()
            if window.shape[0] < target_frames:
                pad_len = target_frames - window.shape[0]
                window = torch.nn.functional.pad(window, (0, 0, 0, pad_len))
            else:
                window = window[:target_frames, :]
            windows.append(window)

        input_values = torch.stack(windows, dim=0)
        if do_normalize:
            input_values = (input_values - fe_mean) / (fe_std * 2.0)

        return input_values.detach().cpu().numpy().astype(np.float32)

    def _get_target_frames(self, seg_frames: int, max_length: int) -> int:
        input_shape = getattr(self, "_input_shape", None)
        if (
            input_shape
            and len(input_shape) > 1
            and isinstance(input_shape[1], (int, np.integer))
        ):
            target_frames = int(input_shape[1])
            if seg_frames > target_frames:
                raise ValueError(
                    "Segment spectrogram exceeds the fixed ONNX frame dimension: "
                    f"segment has {seg_frames} frames, model expects {target_frames}"
                )
            return target_frames
        return max(1, min(max_length, seg_frames))

    def predict(self, wav_path: str, segment_duration: int = 3, hop_duration: int = 2, threshold: Optional[float] = None, min_num_positive_calls_threshold: Optional[int] = None) -> dict:
        """Predict using the exported fixed spectrogram shape.

        Segment durations are supported only when preprocessing produces the
        same time and mel dimensions as the exported model.
        """
        threshold = threshold if threshold is not None else self.threshold
        min_num_positive_calls_threshold = min_num_positive_calls_threshold if min_num_positive_calls_threshold is not None else self.min_num_positive_calls_threshold

        if "other" in self.label2id:
            primary_negative_id = self.label2id["other"]
        else:
            primary_negative_id = min(self.negative_class_ids)
        empty_per_class_probabilities = {
            label: 0.0 for label in self.id2label.values()
        }
        primary_negative_label = self.id2label.get(primary_negative_id)
        if primary_negative_label is None:
            primary_negative_label = next(
                (
                    label
                    for label, class_id in self.label2id.items()
                    if class_id == primary_negative_id
                ),
                "other",
            )

        def empty_result():
            return {
                "local_predictions": [],
                "local_confidences": [],
                "local_probs": [],
                "global_prediction": primary_negative_id,
                "global_prediction_label": primary_negative_label,
                "global_prediction_labels": [],
                "global_confidence": 0.0,
                "per_class_probabilities": empty_per_class_probabilities,
                "hop_duration": float(hop_duration),
                "segment_duration": float(segment_duration),
            }

        # Load audio
        import librosa
        try:
            audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        except Exception as e:
            print(f"Error loading audio file {wav_path}: {type(e).__name__}: {e}")
            return empty_result()
        if len(audio) == 0:
            print(f"Warning: Audio file {wav_path} is empty")
            return empty_result()
        audio_duration = len(audio) / sr
        segment_samples = int(segment_duration * sr)
        hop_samples = int(hop_duration * sr)
        num_positions = int(math.floor((audio_duration - segment_duration) / hop_duration)) + 1 if audio_duration >= segment_duration else 1

        if num_positions <= 0:
            num_positions = 1

        # Build model input batch
        batch_np = self._compute_input_values(audio, sr, num_positions, hop_samples, segment_samples)
        # batch_np shape: (num_positions, frames, mel_bins)
        expected_shape = getattr(self, "_input_shape", None)
        if expected_shape is not None:
            actual_shape = batch_np.shape
            if len(expected_shape) != batch_np.ndim or any(
                isinstance(expected_shape[dimension], (int, np.integer))
                and expected_shape[dimension] != actual_shape[dimension]
                for dimension in range(1, min(len(expected_shape), batch_np.ndim))
            ):
                raise ValueError(
                    "Input spectrogram dimensions do not match the ONNX model "
                    f"for segment_duration={segment_duration}: "
                    f"expected {tuple(expected_shape[1:])}, got {tuple(actual_shape[1:])}"
                )

        # Run ORT in batches
        all_logits = []
        t0_total = time.perf_counter()
        for start in range(0, batch_np.shape[0], self.inference_batch_size):
            end = min(start + self.inference_batch_size, batch_np.shape[0])
            batch = batch_np[start:end]
            res = self.sess.run(None, {self._input_name: batch})
            logits = res[0]
            all_logits.append(logits)
        t1_total = time.perf_counter()
        logits_np = np.vstack(all_logits) if all_logits else np.empty((0, 0), dtype=np.float32)

        # Convert logits to probabilities (softmax)
        def softmax(x):
            e = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)

        if logits_np.size == 0:
            return {
                "local_predictions": [],
                "local_confidences": [],
                "local_probs": [],
                "global_prediction": None,
                "global_prediction_label": "",
                "global_prediction_labels": [],
                "global_confidence": 0.0,
                "per_class_probabilities": {},
                "hop_duration": float(hop_duration),
                "segment_duration": float(segment_duration),
            }

        probs = softmax(logits_np)
        segment_probs = [row.tolist() for row in probs]
        local_pred_ids = [int(np.argmax(row)) for row in probs]
        local_confidences = [
            float(1.0 - sum(row[class_id] for class_id in self.negative_class_ids))
            for row in probs
        ]

        # Per-class average local confidence, conditioned on the predicted class.
        num_labels = probs.shape[1]
        per_class_probs = {}
        for class_id in range(num_labels):
            matching_confidences = [
                confidence
                for confidence, predicted_id in zip(local_confidences, local_pred_ids)
                if predicted_id == class_id
            ]
            per_class_probs[self.id2label.get(class_id, str(class_id))] = (
                float(np.mean(matching_confidences)) if matching_confidences else 0.0
            )

        positive_mask = [
            class_id not in self.negative_class_ids and confidence >= threshold
            for class_id, confidence in zip(local_pred_ids, local_confidences)
        ]
        total_segments = len(local_pred_ids)
        scaled_threshold = max(1, (total_segments + 9) // 10)
        effective_threshold = min(scaled_threshold, min_num_positive_calls_threshold)

        event_ids = _positive_event_ids(positive_mask)
        class_event_confidences = {}
        for class_id, confidence, event_id in zip(
            local_pred_ids, local_confidences, event_ids
        ):
            if (
                event_id is None
                or class_id in self.negative_class_ids
                or confidence < threshold
            ):
                continue
            class_event_confidences.setdefault(class_id, {}).setdefault(
                event_id, []
            ).append(confidence)

        class_votes = {
            class_id: [float(np.mean(values)) for values in events.values()]
            for class_id, events in class_event_confidences.items()
        }
        qualifying_classes = {
            class_id: confidences
            for class_id, confidences in class_votes.items()
            if len(confidences) >= effective_threshold
        }
        if qualifying_classes:
            ordered_classes = sorted(
                qualifying_classes,
                key=lambda class_id: (
                    -len(qualifying_classes[class_id]),
                    -float(np.mean(qualifying_classes[class_id])),
                    self.id2label.get(class_id, str(class_id)),
                ),
            )
            global_id = ordered_classes[0]
            global_confidence = float(np.mean(qualifying_classes[global_id]))
        elif (
            _count_non_adjacent_positive_events(positive_mask) >= effective_threshold
            and class_votes
        ):
            global_id = sorted(
                class_votes,
                key=lambda class_id: (
                    -len(class_votes[class_id]),
                    -float(np.mean(class_votes[class_id])),
                    self.id2label.get(class_id, str(class_id)),
                ),
            )[0]
            global_confidence = float(np.mean(class_votes[global_id]))
        else:
            background_predictions = [
                class_id
                for class_id in local_pred_ids
                if class_id in self.negative_class_ids
            ]
            if background_predictions:
                global_id = max(
                    dict.fromkeys(background_predictions),
                    key=background_predictions.count,
                )
                global_confidence = float(np.mean([
                    row[global_id] for row in segment_probs
                ]))
            elif "other" in self.label2id:
                global_id = self.label2id["other"]
                global_confidence = 0.0
            elif self.negative_class_ids:
                global_id = min(self.negative_class_ids)
                global_confidence = 0.0
            else:
                global_id = local_pred_ids[0]
                global_confidence = 0.0

        class_means = {
            class_id: float(np.mean([
                confidence
                for confidence, predicted_id in zip(local_confidences, local_pred_ids)
                if predicted_id == class_id
            ]))
            for class_id in range(num_labels)
            if class_id in local_pred_ids
        }
        seen_local_ids = list(dict.fromkeys(local_pred_ids))
        negative_ids = {
            self.label2id[name]
            for name in {"jingle", "bird"}
            if name in self.label2id
        }
        background_ids = {
            self.label2id[name]
            for name in {"abiotic", "other", "water", "vessel", "human"}
            if name in self.label2id
        }
        positive_ids = set(seen_local_ids) - negative_ids - background_ids
        qualifying_ids = list(qualifying_classes)
        class_event_counts = {}
        for class_id in seen_local_ids:
            event_count = _count_non_adjacent_positive_events([
                prediction == class_id for prediction in local_pred_ids
            ])
            class_event_counts[class_id] = event_count
            if (
                class_id not in qualifying_classes
                and (class_id in negative_ids or class_id in background_ids)
                and event_count >= effective_threshold
            ):
                qualifying_ids.append(class_id)

        def sort_group(class_ids):
            return sorted(
                class_ids,
                key=lambda class_id: (
                    -class_event_counts.get(class_id, 0),
                    -class_means.get(class_id, 0.0),
                    self.id2label.get(class_id, str(class_id)),
                ),
            )

        ordered_ids = (
            sort_group(class_id for class_id in qualifying_ids if class_id in positive_ids)
            + sort_group(class_id for class_id in qualifying_ids if class_id in negative_ids)
            + sort_group(
                class_id
                for class_id in qualifying_ids
                if class_id in background_ids
                or (class_id not in positive_ids and class_id not in negative_ids)
            )
        )
        if ordered_ids:
            global_id = ordered_ids[0]
            global_confidence = (
                float(np.mean([row[global_id] for row in segment_probs]))
                if global_id in self.negative_class_ids
                else class_means.get(global_id, 0.0)
            )
        global_label = self.id2label.get(global_id, str(global_id))
        global_labels = [
            self.id2label.get(class_id, str(class_id)) for class_id in ordered_ids
        ]

        return {
            "local_predictions": local_pred_ids,
            "local_confidences": local_confidences,
            "local_probs": segment_probs,
            "global_prediction": int(global_id),
            "global_prediction_label": global_label,
            "global_prediction_labels": global_labels,
            "global_confidence": float(global_confidence),
            "per_class_probabilities": per_class_probs,
            "hop_duration": float(hop_duration),
            "segment_duration": float(segment_duration),
            "predict_time": (t1_total - t0_total),
        }


def get_podsai_onnx_inference(onnx_path: str, feature_extractor: Optional[str] = None, **kwargs) -> PodsAIOnnxInference:
    return PodsAIOnnxInference(onnx_path, feature_extractor=feature_extractor, **kwargs)
