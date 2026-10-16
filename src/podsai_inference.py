#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
PODS-AI model inference wrapper for orca call detection.

This module provides a wrapper around HuggingFace audio classification models
that implements the interface expected by PODS-AI's model_inference system.
The default training path now targets spectrogram-based architectures such as
AST, while inference remains compatible with existing PODS-AI checkpoints.
"""

import os
import torch
import librosa
import numpy as np
from collections import Counter
from typing import Optional, Sequence
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from pathlib import Path
from datetime import datetime

# Import base class to establish inheritance
from model_inference import ModelInference

# Segment grouping size for scaling the positive calls threshold.
# For every SEGMENT_GROUP_SIZE segments, require at least 1 positive prediction.
SEGMENT_GROUP_SIZE = 10


def count_non_adjacent_positive_events(positive_mask: Sequence[bool | int]) -> int:
    """Count distinct acoustic events by collapsing adjacent positive segments.

    At most two adjacent segments are collapsed into one event. Therefore a
    contiguous positive run of length ``L`` contributes ``(L + 1) // 2`` events.

    Sliding windows hop by 1-2 seconds, so one call often lights up two
    neighboring segments. Those neighbors are one event. A gap of one or more
    non-positive segments starts a new event.

    Examples:
        [1, 1, 1, 0] → 2 events
        [1, 0, 1, 0] → 2 events
        [1, 1, 0, 1, 1] → 2 events
        [1, 1, 1, 1] → 2 events
        [1, 1, 1, 1, 1] → 3 events
    """
    events = 0
    run_length = 0
    for value in positive_mask:
        if bool(value):
            run_length += 1
        elif run_length:
            events += (run_length + 1) // 2
            run_length = 0
    if run_length:
        events += (run_length + 1) // 2
    return events


def meets_min_positive_event_threshold(
    positive_mask: Sequence[bool | int],
    min_num_positive_calls_threshold: int,
) -> bool:
    """Return True when collapsed event count meets the configured threshold."""
    return (
        count_non_adjacent_positive_events(positive_mask)
        >= min_num_positive_calls_threshold
    )


def _positive_event_ids(positive_mask: Sequence[bool | int]) -> list[Optional[int]]:
    """Assign segments to global positive events, capped at two segments each."""
    event_ids: list[Optional[int]] = []
    event_id = -1
    run_length = 0
    for value in positive_mask:
        if not bool(value):
            run_length = 0
            event_ids.append(None)
            continue
        if run_length % 2 == 0:
            event_id += 1
        event_ids.append(event_id)
        run_length += 1
    return event_ids

# HuggingFace model_type value for the Audio Spectrogram Transformer architecture.
# AST models expect a pre-computed mel spectrogram; raw-audio models (e.g. Wav2Vec2)
# use the feature extractor directly.
MODEL_TYPE_AST = "audio-spectrogram-transformer"
# AST adds two learned special tokens: [CLS] and distillation token.
NUM_SPECIAL_TOKENS = 2
# Default AST patch geometry when config omits explicit values.
DEFAULT_AST_PATCH_SIZE = 16
DEFAULT_AST_FREQUENCY_STRIDE = 10
DEFAULT_AST_TIME_STRIDE = 10


class PodsAIInference(ModelInference):  # Inherit from ModelInference
    """
    Inference wrapper for PODS-AI audio classification models.

    This class implements the interface expected by model_inference.py,
    providing predictions in the format required by extract_training_samples.py.
    """

    def __init__(self, model_path: str, device: Optional[str] = None,
                 threshold: float = 0.5, min_num_positive_calls_threshold: int = 2,
                 model_revision: Optional[str] = None,
                 inference_batch_size: int = 8) -> None:
        """
        Initialize the inference model.

        Args:
            model_path: Path to model directory or HuggingFace Hub model ID
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            threshold: Confidence threshold for positive predictions (default: 0.5)
            min_num_positive_calls_threshold: Minimum distinct (non-adjacent) positive
                                             events for a global positive classification.
                                             Adjacent overlapping positives collapse to one
                                             event. The effective threshold scales with audio
                                             length: at least 1 event per SEGMENT_GROUP_SIZE
                                             (10) segments, capped at this value. Formula:
                                             min(ceil(segments/10), min_num_positive_calls_threshold).
                                             Default: use instance value (typically 2).
            model_revision: Git commit hash or tag to pin the HuggingFace Hub model revision.
                           Ignored when model_path is a local directory. (default: None)
            inference_batch_size: Number of spectrogram windows processed per model forward pass
                                  (default: 8). Smaller values reduce peak attention-matrix memory
                                  at the cost of more forward passes; tune for your hardware.
        """
        super().__init__(model_path)  # Call parent constructor
        self.threshold = threshold
        self.min_num_positive_calls_threshold = min_num_positive_calls_threshold
        self.inference_batch_size = inference_batch_size

        # Auto-detect device. Default to GPU if available, otherwise CPU. Allow override via argument.
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        revision_info = f" @ {model_revision}" if model_revision else ""
        print(f"Loading PODS-AI model from {model_path}{revision_info}...")
        print(f"Using device: {self.device}")

        # Build kwargs for from_pretrained; only pass revision/token for Hub IDs (not local paths).
        pretrained_kwargs: dict = {}
        if not Path(model_path).exists():
            if model_revision:
                pretrained_kwargs["revision"] = model_revision
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                pretrained_kwargs["token"] = hf_token

        # Load feature extractor and model.
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_path, **pretrained_kwargs
            )
        except Exception as e:
            error_msg = f"Error loading feature extractor from {model_path}: {type(e).__name__}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

        try:
            # Try SDPA (Scaled Dot-Product Attention) first for better memory efficiency.
            # SDPA fuses the Q×K^T softmax Attn×V computation in one kernel, which
            # reduces intermediate tensor allocations and peak attention-matrix memory.
            # Fall back to the default implementation when SDPA is not supported.
            try:
                self.model = AutoModelForAudioClassification.from_pretrained(
                    model_path, attn_implementation="sdpa", **pretrained_kwargs
                )
            except (ValueError, NotImplementedError):
                self.model = AutoModelForAudioClassification.from_pretrained(
                    model_path, **pretrained_kwargs
                )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            error_msg = f"Error loading model from {model_path}: {type(e).__name__}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

        # Detect model architecture to select the appropriate inference input path.
        # AST models (audio-spectrogram-transformer) consume a pre-computed mel spectrogram
        # of shape (batch, max_length, num_mel_bins).  All other models (e.g. Wav2Vec2)
        # expect raw audio samples of shape (batch, sequence_length).
        self._use_spectrogram_input = (
            getattr(self.model.config, 'model_type', '') == MODEL_TYPE_AST
        )
        # Cache resized AST positional embeddings by token count to avoid re-interpolating.
        self._ast_pos_embed_cache: dict[int, torch.Tensor] = {}

        # Get label mapping. This assumes the model was trained with a config that includes id2label and label2id.
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        print(f"Model loaded successfully. Label mapping: {self.label2id}")

        # Print model version/date information.
        self._print_model_metadata(model_path)

        # Validate that model has at least one negative/background class.
        # Accept either explicit "other" or treat specific classes as negative.
        negative_classes = {
            "abiotic", "bird", "human", "jingle", "other", "vessel", "water"
        }
        positive_classes = {"resident", "transient", "humpback"}

        found_negative = negative_classes & set(self.label2id.keys())
        found_positive = positive_classes & set(self.label2id.keys())

        if not found_negative:
            raise ValueError(
                "Model must include at least one negative/background class "
                "(other, abiotic, water, vessel, jingle, human, or bird). "
                f"Found labels: {list(self.label2id.keys())}. "
                f"Please train the model with at least one negative class to distinguish from whale calls."
            )

        if not found_positive:
            print(
                f"Warning: Model has negative classes but no expected positive classes (resident, transient, humpback). "
                f"This may work but confidence scores may not be meaningful. "
                f"Found labels: {list(self.label2id.keys())}"
            )
        elif found_positive != positive_classes:
            missing = positive_classes - found_positive
            print(
                f"Warning: Model is missing some expected positive classes: {missing}. "
                f"Predictions will be computed from available classes: {found_positive}"
            )

        # Store which classes are considered negative (non-whale)
        self.negative_class_ids = {self.label2id[label] for label in found_negative}
        print(f"Treating classes as negative/background: {found_negative}")

    def _ensure_ast_position_embeddings(
        self, target_frames: int, num_mel_bins: int
    ) -> bool:
        """
        Resize AST position embeddings to match the requested spectrogram shape.
        This updates model embeddings in-place for the requested token shape.
        Callers should not share this inference instance across threads.

        Returns:
            True if embeddings already match target shape or were resized successfully.
            False if AST embedding resize is unavailable for the current model.
        """
        ast_module = getattr(self.model, "audio_spectrogram_transformer", None)
        embeddings = getattr(ast_module, "embeddings", None)
        position_embeddings = getattr(embeddings, "position_embeddings", None)
        if position_embeddings is None or position_embeddings.ndim != 3:
            return False

        config = self.model.config
        try:
            patch_size = int(getattr(config, "patch_size", DEFAULT_AST_PATCH_SIZE))
            freq_stride = int(getattr(config, "frequency_stride", DEFAULT_AST_FREQUENCY_STRIDE))
            time_stride = int(getattr(config, "time_stride", DEFAULT_AST_TIME_STRIDE))
            cfg_max_length = int(getattr(config, "max_length", target_frames))
            cfg_num_mel_bins = int(getattr(config, "num_mel_bins", num_mel_bins))
        except (TypeError, ValueError):
            return False

        target_freq = (num_mel_bins - patch_size) // freq_stride + 1
        target_time = (target_frames - patch_size) // time_stride + 1
        if target_freq < 1 or target_time < 1:
            return False
        target_tokens = target_freq * target_time + NUM_SPECIAL_TOKENS
        target_cache_key = (target_freq, target_time)

        if position_embeddings.shape[1] == target_tokens:
            return True

        if target_cache_key in self._ast_pos_embed_cache:
            resized = self._ast_pos_embed_cache[target_cache_key]
            embeddings.position_embeddings = torch.nn.Parameter(resized, requires_grad=False)
            return True

        source_freq = (cfg_num_mel_bins - patch_size) // freq_stride + 1
        source_time = (cfg_max_length - patch_size) // time_stride + 1
        if source_freq < 1 or source_time < 1:
            return False
        source_cache_key = (source_freq, source_time)

        source_embeddings = self._ast_pos_embed_cache.get(source_cache_key)
        if source_embeddings is None:
            # Keep an immutable copy of the current embeddings because we mutate
            # model embeddings in-place for resized token grids.
            source_embeddings = position_embeddings.detach().clone()
            self._ast_pos_embed_cache[source_cache_key] = source_embeddings

        source_patch = source_embeddings[:, NUM_SPECIAL_TOKENS:, :]
        source_tokens = source_patch.shape[1]
        if source_freq * source_time != source_tokens:
            return False

        # Convert token sequence [batch, tokens, hidden] into [batch, hidden, freq, time]
        # so bilinear interpolation can resize the 2D patch grid.
        hidden_dim = source_embeddings.shape[-1]
        source_patch = source_patch.reshape(1, source_freq, source_time, hidden_dim).permute(0, 3, 1, 2)
        resized_patch = torch.nn.functional.interpolate(
            source_patch, size=(target_freq, target_time), mode="bilinear", align_corners=False
        )
        # Convert back from [batch, hidden, freq, time] to token sequence format.
        resized_patch = resized_patch.permute(0, 2, 3, 1).reshape(1, target_freq * target_time, hidden_dim)
        resized = torch.cat([source_embeddings[:, :NUM_SPECIAL_TOKENS, :], resized_patch], dim=1)

        self._ast_pos_embed_cache[target_tokens] = resized.detach()
        embeddings.position_embeddings = torch.nn.Parameter(resized, requires_grad=False)
        return True

    def _print_model_metadata(self, model_path: str) -> None:
        """
        Print model metadata including version, date, and architecture information.

        Args:
            model_path: Path to model directory or HuggingFace Hub model ID
        """
        config = self.model.config
        metadata_found = False

        # Print model name/path from config.
        if hasattr(config, '_name_or_path') and config._name_or_path:
            print(f"Model name/path: {config._name_or_path}")
            metadata_found = True

        # Print architecture information.
        if hasattr(config, 'architectures') and config.architectures:
            print(f"Model architecture: {', '.join(config.architectures)}")
            metadata_found = True

        # Print HuggingFace Hub commit hash if available.
        if hasattr(config, '_commit_hash') and config._commit_hash:
            print(f"Model commit hash: {config._commit_hash}")
            metadata_found = True

        # Check for local model directory and extract file timestamps.
        model_path_obj = Path(model_path)
        if model_path_obj.exists() and model_path_obj.is_dir():
            # Check config.json modification time.
            config_json_path = model_path_obj / "config.json"
            if config_json_path.exists():
                mtime = datetime.fromtimestamp(config_json_path.stat().st_mtime)
                print(f"Model config last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                metadata_found = True

            # Check pytorch_model.bin modification time.
            model_bin_path = model_path_obj / "pytorch_model.bin"
            if model_bin_path.exists():
                mtime = datetime.fromtimestamp(model_bin_path.stat().st_mtime)
                print(f"Model weights last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                metadata_found = True

            # Check for training_args.bin with training metadata.
            training_args_path = model_path_obj / "training_args.bin"
            if training_args_path.exists():
                try:
                    training_args = torch.load(training_args_path, map_location='cpu', weights_only=False)
                    if hasattr(training_args, 'output_dir'):
                        print(f"Training output directory: {training_args.output_dir}")
                    if hasattr(training_args, 'num_train_epochs'):
                        print(f"Training epochs: {training_args.num_train_epochs}")
                    metadata_found = True
                except Exception as e:
                    # Silently skip if training_args.bin can't be loaded.
                    pass

        if not metadata_found:
            print("Model metadata: No version or date information available.")

    def _compute_input_values(
        self,
        audio: np.ndarray,
        sr: int,
        num_positions: int,
        hop_samples: int,
        segment_samples: int,
    ) -> torch.Tensor:
        """
        Compute mel spectrogram for the full audio once, then extract sliding windows.

        Computing the spectrogram once avoids the overhead of 29 separate feature-extractor
        calls and eliminates redundant computation for overlapping windows (a 2-second hop
        on 3-second segments has 1 second of overlap per adjacent pair).

        Args:
            audio: Full audio samples at sample rate sr.
            sr: Sample rate of the audio.
            num_positions: Number of sliding-window positions.
            hop_samples: Hop size in samples.
            segment_samples: Segment size in samples.

        Returns:
            Tensor of shape (num_positions, target_frames, num_mel_bins) ready for the model.
        """
        import torchaudio

        # Read ASTFeatureExtractor configuration with sensible defaults.
        num_mel_bins: int = getattr(self.feature_extractor, 'num_mel_bins', 128)
        max_length: int = getattr(self.feature_extractor, 'max_length', 1024)
        fe_mean: float = getattr(self.feature_extractor, 'mean', -4.2677393)
        fe_std: float = getattr(self.feature_extractor, 'std', 4.5689974)
        do_normalize: bool = getattr(self.feature_extractor, 'do_normalize', True)
        frame_shift_ms: float = getattr(self.feature_extractor, 'hop_length', 10)

        # Compute mel spectrogram for the entire audio in a single torchaudio call.
        # Using the same parameters as ASTFeatureExtractor._extract_fbank_features().
        waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, n_samples)
        full_fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_shift=frame_shift_ms,
        )  # Shape: (total_frames, num_mel_bins)

        # Convert sample counts to frame counts.
        frames_per_second = 1000.0 / frame_shift_ms  # e.g., 100 frames/s at 10 ms/frame
        hop_frames = round(hop_samples / sr * frames_per_second)
        seg_frames = round(segment_samples / sr * frames_per_second)

        # Use segment-length windows for AST to avoid unnecessary compute, but ensure
        # model positional embeddings are resized to the resulting patch grid.
        target_frames = max(1, min(max_length, seg_frames))
        if self._use_spectrogram_input:
            if not self._ensure_ast_position_embeddings(target_frames, num_mel_bins):
                print("Warning: AST positional embedding resize unavailable; using full max_length input.")
                target_frames = max_length

        # Slice each window, apply per-utterance mean normalisation, and pad to target_frames.
        # This replicates ASTFeatureExtractor._extract_fbank_features() for each window.
        windows: list[torch.Tensor] = []
        for pos_idx in range(num_positions):
            start_frame = pos_idx * hop_frames
            end_frame = start_frame + seg_frames
            window = full_fbank[start_frame:end_frame, :]

            # Per-utterance mean subtraction (matches ASTFeatureExtractor._extract_fbank_features).
            window = window - window.mean()

            # Pad or truncate to target_frames.
            if window.shape[0] < target_frames:
                pad_len = target_frames - window.shape[0]
                window = torch.nn.functional.pad(window, (0, 0, 0, pad_len))
            else:
                window = window[:target_frames, :]

            windows.append(window)

        input_values = torch.stack(windows, dim=0)  # (num_positions, max_length, num_mel_bins)

        # Apply global normalisation (matches ASTFeatureExtractor.normalize).
        if do_normalize:
            input_values = (input_values - fe_mean) / (fe_std * 2.0)

        return input_values

    def predict(self, wav_path: str, segment_duration: int = 3, hop_duration: int = 2,
                threshold: Optional[float] = None, min_num_positive_calls_threshold: Optional[int] = None) -> dict[str, object]:
        """
        Run inference on a wav file using sliding window.

        Uses a sliding window approach with configurable segment and hop duration to match
        the orcahello LiveInferenceOrchestrator behavior. Unlike FastAIModel which uses a
        1-second hop and produces per-second outputs, this implementation uses a configurable
        hop_duration (default 2 seconds) for efficiency.

        The timestamp correction logic in extract_training_samples.py automatically adapts
        by inferring hop_duration = audio_duration / len(local_confidences), so the different
        output length is handled transparently.

        Args:
            wav_path: Path to wav file (typically 60 seconds long)
            segment_duration: Duration of each segment in seconds (default: 3)
            hop_duration: Hop size in seconds between segments (default: 2)
                         With hop_duration=2, a 60s audio produces ~29 confidence values.
                         With hop_duration=1, a 60s audio produces ~58 confidence values (matching FastAI).
            threshold: Confidence threshold for positive (non-other) predictions (default: use instance value)
            min_num_positive_calls_threshold: Minimum distinct (non-adjacent) positive
                                             events for a global positive classification.
                                             Adjacent overlapping positives collapse to one
                                             event. The effective threshold scales with audio
                                             length: at least 1 event per SEGMENT_GROUP_SIZE
                                             (10) segments, capped at this value. Formula:
                                             min(ceil(segments/10), min_num_positive_calls_threshold).
                                             Default: use instance value (typically 2).

        Returns:
            Dictionary with keys:
                - local_predictions: List of class IDs for each hop_duration interval.
                                    Length = floor((audio_duration - segment_duration) / hop_duration) + 1
                - local_confidences: List of whale-call likelihood scores (0.0-1.0) at each interval.
                                    Computed as 1 - P(negative classes). Used by timestamp correction
                                    to identify the most likely position of whale calls.
                                    local_confidences[i] represents the score at time offset
                                    i * hop_duration seconds from the start.
                - global_prediction: Overall class ID for the entire audio
                - global_prediction_label: Legacy primary label for the global prediction
                - global_prediction_labels: Ordered labels, or [] when none qualify
                - global_confidence: Overall confidence score (0.0-1.0) for the global prediction
            Returns dict with empty lists and error values if audio loading fails.
        """
        # Use instance values if not overridden.
        threshold = threshold if threshold is not None else self.threshold
        min_num_positive_calls_threshold = (
            min_num_positive_calls_threshold
            if min_num_positive_calls_threshold is not None
            else self.min_num_positive_calls_threshold
        )

        # Get the primary negative class ID for error returns.
        # This is used when we cannot process the audio and need to return a default negative prediction.
        if "other" in self.label2id:
            primary_negative_id = self.label2id["other"]
        else:
            primary_negative_id = min(self.negative_class_ids)
        empty_per_class_probabilities = {
            label: 0.0 for label in self.id2label.values()
        }

        # Load audio. Resample to 16kHz and convert to mono. Handle exceptions gracefully.
        try:
            audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        except Exception as e:
            error_msg = f"Error loading audio file {wav_path}: {type(e).__name__}: {e}"
            print(error_msg)
            return {
                "local_predictions": [],
                "local_confidences": [],
                "local_probs": [],
                "global_prediction": primary_negative_id,
                "global_prediction_label": self.id2label[primary_negative_id],
                "global_prediction_labels": [],
                "per_class_probabilities": empty_per_class_probabilities,
                "global_confidence": 0.0,
                "hop_duration": float(hop_duration),
                "segment_duration": float(segment_duration),
            }

        # Handle empty audio. This can happen if the file is corrupted or has no valid audio data.
        if len(audio) == 0:
            print(f"Warning: Audio file {wav_path} is empty")
            return {
                "local_predictions": [],
                "local_confidences": [],
                "local_probs": [],
                "global_prediction": primary_negative_id,
                "global_prediction_label": self.id2label[primary_negative_id],
                "global_prediction_labels": [],
                "per_class_probabilities": empty_per_class_probabilities,
                "global_confidence": 0.0,
                "hop_duration": float(hop_duration),
                "segment_duration": float(segment_duration),
            }

        # Calculate segment and hop sizes in samples.
        segment_samples = segment_duration * sr
        hop_samples = hop_duration * sr

        # Calculate total audio duration and number of segments with sliding window.
        audio_duration = len(audio) / sr

        # Generate segment predictions using sliding window with hop_duration-second hop.
        # Each position represents a segment_duration-second window starting at position_index * hop_duration.
        # Store both class predictions and their probabilities.
        segment_class_ids: list[int] = []
        segment_probs: list[np.ndarray] = []

        # Calculate number of positions based on hop_duration.
        # num_positions = how many segment_duration windows fit with hop_duration spacing
        # Last position must start early enough that segment_duration window fits within audio.
        num_positions = int(np.floor((audio_duration - segment_duration) / hop_duration)) + 1

        # Handle very short audio (shorter than segment_duration).
        # In this case, we will just process one segment starting at 0 seconds.
        if num_positions < 1:
            num_positions = 1

        if self._use_spectrogram_input:
            # AST path: compute the full-audio mel spectrogram once, then slice windows.
            # This avoids 29 separate fbank calls (one per segment) and eliminates redundant
            # computation for overlapping windows, dramatically reducing feature-extraction time.
            # Windows are fed to the model in mini-batches to limit peak attention-matrix memory.
            input_values = self._compute_input_values(
                audio, sr, num_positions, hop_samples, segment_samples
            )

            all_probs_list: list[torch.Tensor] = []
            with torch.inference_mode():
                for batch_start in range(0, num_positions, self.inference_batch_size):
                    batch = input_values[batch_start:batch_start + self.inference_batch_size]
                    outputs = self.model(input_values=batch.to(self.device))
                    batch_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    all_probs_list.append(batch_probs.cpu())
            all_probs = torch.cat(all_probs_list, dim=0)
        else:
            # Raw-audio path (e.g., Wav2Vec2): collect segments, run feature extractor,
            # then do a single batched forward pass.
            segments = []
            for pos_idx in range(num_positions):
                start = pos_idx * hop_samples
                end = min(start + segment_samples, len(audio))
                segment = audio[start:end]
                if len(segment) < segment_samples:
                    segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
                segments.append(segment)

            with torch.inference_mode():
                inputs = self.feature_extractor(
                    segments, sampling_rate=sr, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                all_probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu()

        predicted_classes = torch.argmax(all_probs, dim=-1).tolist()
        segment_class_ids = [int(c) for c in predicted_classes]
        segment_probs = [all_probs[i].numpy() for i in range(num_positions)]

        # Add any missing classes, with probability 0.
        num_model_classes = len(segment_probs[0])
        num_label_classes = max(self.id2label.keys()) + 1
        if num_label_classes > num_model_classes:
            pad_width = num_label_classes - num_model_classes
            segment_probs = [
                np.pad(probs, (0, pad_width), mode="constant", constant_values=0.0)
                for probs in segment_probs
            ]

        # Guard against empty predictions list.
        # This can happen if the audio is too short or if there was an error during processing.
        if not segment_class_ids:
            print(f"Warning: No segments processed for {wav_path}")
            return {
                "local_predictions": [],
                "local_confidences": [],
                "local_probs": [],
                "global_prediction": primary_negative_id,
                "global_prediction_label": self.id2label[primary_negative_id],
                "global_prediction_labels": [],
                "per_class_probabilities": empty_per_class_probabilities,
                "global_confidence": 0.0,
                "hop_duration": float(hop_duration),
                "segment_duration": float(segment_duration),
            }

        # Get local predictions (class IDs) and call-likelihood confidences.
        # For timestamp correction, we need the likelihood of a whale call being present,
        # not the confidence in the predicted class (which could be a background class).
        local_predictions: list[int] = []
        local_confidences: list[float] = []

        for probs in segment_probs:
            predicted_class = int(np.argmax(probs))

            # Compute call-likelihood as 1 - P(negative classes).
            # This represents "how likely is there a whale call" regardless of which class wins.
            negative_prob = sum(probs[class_id] for class_id in self.negative_class_ids)
            call_likelihood = 1.0 - negative_prob

            local_predictions.append(predicted_class)
            local_confidences.append(float(call_likelihood))

        # Determine global prediction based on voting among high-confidence predictions.
        # For positive (whale) classes, we require multiple high-confidence predictions.
        # For negative (background) classes, we use the most common prediction.

        # Filter for positive (whale) predictions with confidence above threshold.
        positive_mask = [
            class_id not in self.negative_class_ids and conf >= threshold
            for class_id, conf in zip(local_predictions, local_confidences)
        ]

        # Scale the positive-event threshold based on the number of segments.
        # For every SEGMENT_GROUP_SIZE segments, require at least 1 distinct event.
        # Cap at min_num_positive_calls_threshold to avoid requiring too many for very long clips.
        total_segments = len(local_predictions)
        scaled_threshold = max(1, (total_segments + SEGMENT_GROUP_SIZE - 1) // SEGMENT_GROUP_SIZE)
        effective_threshold = min(scaled_threshold, min_num_positive_calls_threshold)

        # Build class support from the global event boundaries. This prevents
        # class jitter (for example resident/transient alternation) from
        # turning one acoustic event into multiple per-class events.
        event_ids = _positive_event_ids(positive_mask)
        class_event_confidences: dict[int, dict[int, list[float]]] = {}
        for class_id, conf, event_id in zip(
            local_predictions, local_confidences, event_ids
        ):
            if event_id is None or class_id in self.negative_class_ids or conf < threshold:
                continue
            class_event_confidences.setdefault(class_id, {}).setdefault(
                event_id, []
            ).append(conf)

        # Each class contributes at most one vote per globally collapsed event.
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
            # Deterministic order: vote count, mean confidence, then label name
            # lexicographically (descending metrics, ascending label name).
            ordered_classes = sorted(
                qualifying_classes,
                key=lambda cid: (
                    -len(qualifying_classes[cid]),
                    -float(np.mean(qualifying_classes[cid])),
                    self.id2label[cid],
                ),
            )
            global_prediction_id = ordered_classes[0]
            global_confidence = float(np.mean(qualifying_classes[global_prediction_id]))
        else:
            # Preserve the legacy majority-vote result when aggregate evidence
            # reaches the threshold but no individual class does.
            if (
                meets_min_positive_event_threshold(positive_mask, effective_threshold)
                and class_votes
            ):
                # Tie-breaker order is count -> mean confidence -> label name
                # lexicographically, matching the multi-label ordering above.
                global_prediction_id = sorted(
                    class_votes,
                    key=lambda cid: (
                        -len(class_votes[cid]),
                        -float(np.mean(class_votes[cid])),
                        self.id2label[cid],
                    ),
                )[0]
                global_confidence = float(np.mean(class_votes[global_prediction_id]))
                ordered_classes = [global_prediction_id]
            else:
                ordered_classes = []
            if not ordered_classes:
                # Filter to background classes when no whale class has enough evidence.
                from collections import Counter
                background_predictions = [
                    c for c in local_predictions if c in self.negative_class_ids
                ]

                if background_predictions:
                    class_counts = Counter(background_predictions)
                    global_prediction_id = class_counts.most_common(1)[0][0]
                    global_confidence = float(np.mean([
                        probs[global_prediction_id] for probs in segment_probs
                    ]))
                else:
                    if "other" in self.label2id:
                        global_prediction_id = self.label2id["other"]
                    else:
                        global_prediction_id = min(self.negative_class_ids)
                    global_confidence = 0.0

        # Convert global prediction ID to label name.
        global_prediction_label = self.id2label[global_prediction_id]

        # Calculate per-class global probabilities.
        # For each class, only include segments whose local prediction
        # matches that class.
        #
        # class_means[class_id] is therefore:
        #   mean(whale-call likelihood for segments locally predicted as class_id)
        class_means: dict[int, float] = {}
        per_class_probabilities: dict[str, float] = {}

        for class_id, label in self.id2label.items():
            matching_confidences = [
                confidence
                for confidence, predicted_id in zip(
                    local_confidences,
                    local_predictions,
                )
                if predicted_id == class_id
            ]

            class_mean = (
                float(np.mean(matching_confidences))
                if matching_confidences
                else 0.0
            )

            class_means[class_id] = class_mean
            per_class_probabilities[label] = class_mean

        # Build the set of unique classes seen in local_predictions in first-seen order.
        seen_local = dict.fromkeys(local_predictions)
        unique_local_ids = list(seen_local.keys())

        # Define label groups by name and map to ids when available.
        NEGATIVE_LABEL_NAMES = {"jingle", "bird"}
        BACKGROUND_LABEL_NAMES = {"abiotic", "other", "water", "vessel", "human"}

        negative_ids_set = {self.label2id[l] for l in NEGATIVE_LABEL_NAMES if l in self.label2id}
        background_ids_set = {self.label2id[l] for l in BACKGROUND_LABEL_NAMES if l in self.label2id}
        # Positive ids are any seen ids not classified as negative or background.
        seen_ids_set = set(unique_local_ids)
        positive_ids_set = seen_ids_set - negative_ids_set - background_ids_set

        qualifying_ids: list[int] = list(qualifying_classes)
        for cid in unique_local_ids:
            if cid in qualifying_classes:
                continue
            if cid not in negative_ids_set and cid not in background_ids_set:
                continue
            mask = [1 if p == cid else 0 for p in local_predictions]
            non_adj_count = count_non_adjacent_positive_events(mask)
            if non_adj_count >= effective_threshold:
                qualifying_ids.append(cid)

        # Keep positive, negative, and background groups in priority order;
        # order each group by event count, mean confidence, then label name.
        positives = [cid for cid in qualifying_ids if cid in positive_ids_set]
        negatives = [cid for cid in qualifying_ids if cid in negative_ids_set]
        backgrounds = [cid for cid in qualifying_ids if cid in background_ids_set or (cid not in positive_ids_set and cid not in negative_ids_set)]
        class_event_counts = {}
        for cid in unique_local_ids:
            mask = [1 if p == cid else 0 for p in local_predictions]
            class_event_counts[cid] = count_non_adjacent_positive_events(mask)

        positives.sort(
            key=lambda cid: (
                -class_event_counts.get(cid, 0),
                -class_means.get(cid, 0.0),
                self.id2label[cid],
            )
        )
        negatives.sort(
            key=lambda cid: (
                -class_event_counts.get(cid, 0),
                -class_means.get(cid, 0.0),
                self.id2label[cid],
            )
        )
        backgrounds.sort(
            key=lambda cid: (
                -class_event_counts.get(cid, 0),
                -class_means.get(cid, 0.0),
                self.id2label[cid],
            )
        )

        ordered_ids = positives + negatives + backgrounds
        # Build `global_prediction_labels` from ordered_ids (unchanged).
        global_prediction_labels = [self.id2label.get(cid, str(cid)) for cid in ordered_ids]

        # Set the singular global prediction to the first label in the ordered list,
        # without modifying ordered_ids itself.
        if ordered_ids:
            global_prediction_id = ordered_ids[0]
            global_prediction_label = self.id2label.get(global_prediction_id, str(global_prediction_id))
            if global_prediction_id in self.negative_class_ids:
                global_confidence = float(np.mean([
                    probs[global_prediction_id] for probs in segment_probs
                ]))
            else:
                global_confidence = class_means.get(global_prediction_id, 0.0)
        # else keep previously computed global_prediction_id/global_prediction_label.

        return {
            "local_predictions": local_predictions,
            "local_confidences": local_confidences,
            "local_probs": segment_probs,
            "global_prediction": global_prediction_id,
            "global_prediction_label": global_prediction_label,
            "global_prediction_labels": global_prediction_labels,
            "global_confidence": global_confidence,
            "per_class_probabilities": per_class_probabilities,
            "hop_duration": float(hop_duration),
            "segment_duration": float(segment_duration),
        }

def get_podsai_inference(model_path: str, **kwargs) -> PodsAIInference:
    """
    Factory function to create PODS-AI inference instance.

    Args:
        model_path: Path to model directory or HuggingFace Hub model ID
        **kwargs: Additional arguments passed to PodsAIInference, e.g.
                  model_revision (str): pinned git commit hash for Hub models.

    Returns:
        PodsAIInference instance
    """
    return PodsAIInference(model_path, **kwargs)
