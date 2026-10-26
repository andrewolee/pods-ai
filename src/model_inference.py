#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Model inference module for scoring audio samples.

This module provides a simple interface for running model inference on audio files
to score segments using a sliding window approach. It can be extended to support
different model types.

For production use, this can be integrated with orcahello's
LiveInferenceOrchestrator.py which uses OrcaHelloSRKWDetectorV1 to download and score
audio from specific time ranges. See:
https://github.com/orcasound/orcahello/blob/main/InferenceSystem/src/LiveInferenceOrchestrator.py
https://github.com/orcasound/orcahello/blob/main/InferenceSystem/src/model/inference.py

The pretrained FastAI model is typically named "model.pkl" and should be placed in
a "model" directory.
"""

import gc
import os
import tempfile
import wave
import torch
import pandas as pd
from pydub import AudioSegment
from librosa import get_duration
from pathlib import Path
from numpy import floor
import torchaudio

from typing import Dict, List, Optional
import warnings


# Segment grouping size for scaling the positive calls threshold.
# For every SEGMENT_GROUP_SIZE segments, require at least 1 positive prediction.
SEGMENT_GROUP_SIZE = 10


# Monkey-patch torchaudio.load to avoid torchcodec dependency
# torchaudio 2.9.0+ defaults to torchcodec backend which requires additional installation
# This patch uses soundfile directly which is already installed
_original_torchaudio_load = torchaudio.load

def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Wrapper for torchaudio.load that uses soundfile directly instead of torchcodec"""
    import soundfile as sf
    import torch as t

    # Load audio using soundfile.
    data, samplerate = sf.read(str(filepath), dtype='float32')

    # Convert to torch tensor and ensure shape is (channels, samples).
    waveform = t.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))

    return waveform, samplerate

torchaudio.load = _patched_torchaudio_load


# Monkey-patch torchaudio.save to avoid torchcodec dependency.
# torchaudio 2.9.0+ defaults to torchcodec backend which requires additional installation.
# This patch uses soundfile directly which is already installed.
_original_torchaudio_save = torchaudio.save

def _patched_torchaudio_save(filepath, src, sample_rate, *args, **kwargs):
    """Wrapper for torchaudio.save that uses soundfile directly instead of torchcodec"""
    import soundfile as sf
    import numpy as np

    # Convert torch tensor to numpy array.
    # src is expected to be (channels, samples), soundfile expects (samples, channels).
    audio_data = src.numpy().T if src.ndim > 1 else src.numpy().reshape(-1, 1)

    # Save audio using soundfile.
    sf.write(str(filepath), audio_data, sample_rate)

torchaudio.save = _patched_torchaudio_save


# Monkey-patch torch.load to use weights_only=False for compatibility with fastai models.
# PyTorch 2.6+ changed the default to weights_only=True for security, but fastai models
# require weights_only=False to load functools.partial and other objects.
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Wrapper for torch.load that defaults to weights_only=False for fastai compatibility"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load


def load_model(mPath, mName="stg2-rn18.pkl"):
    from fastai.basic_train import load_learner
    return load_learner(mPath, mName)


def get_wave_file(wav_file):
    '''
    Function to load a wav file
    '''
    return AudioSegment.from_wav(wav_file)


def export_wave_file(audio, begin, end, dest):
    '''
    Function to extract a smaller wav file based start and end duration information
    '''
    sub_audio = audio[begin * 1000:end * 1000]
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Write PCM directly with the standard library, then publish atomically.
    # This avoids pydub's Python 3.14 WAV path-export incompatibility and keeps
    # downstream readers from observing a partial file.
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{dest_path.name}.", suffix=".tmp", dir=dest_path.parent
    )
    os.close(temp_fd)
    try:
        with wave.open(temp_name, "wb") as output_file:
            output_file.setnchannels(sub_audio.channels)
            output_file.setsampwidth(sub_audio.sample_width)
            output_file.setframerate(sub_audio.frame_rate)
            output_file.writeframes(sub_audio.raw_data)
        os.replace(temp_name, dest_path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def extract_segments(audioPath, sampleDict, destnPath, suffix):
    '''
    Function to extract segments given an audio path folder and proposal segments
    '''
    # Use Path objects to avoid string concatenation.
    audio_dir = Path(audioPath)
    dest_dir = Path(destnPath)
    for wav_file in sampleDict.keys():
        audio_file = get_wave_file(audio_dir / wav_file)
        for begin_time, end_time in sampleDict[wav_file]:
            output_file_name = wav_file.lower().replace(
                '.wav', '') + '_' + str(begin_time) + '_' + str(
                    end_time) + suffix + '.wav'
            output_file_path = dest_dir / output_file_name
            export_wave_file(audio_file, begin_time,
                             end_time, str(output_file_path))


class ModelInference:
    """
    Base class for model inference.

    This class provides an interface for scoring audio files using a sliding window
    approach. Subclasses should implement the actual model loading and inference logic.

    The inference uses overlapping segments (typically 3-second windows) to generate
    confidence scores. Different implementations may use different hop sizes:
    - FastAIModel: 1-second hop, produces per-second confidences
    - PodsAIInference: Configurable hop (default 2 seconds)

    The timestamp correction logic in extract_training_samples.py automatically infers
    the hop duration from the output length, so implementations are free to choose
    their own hop size.

    Supports both binary classification (whale vs other) and multi-class classification
    (species identification).
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model inference.

        Args:
            model_path: Optional path to the model file or directory.
        """
        self.model_path = model_path
        self.model = None

    def predict(self, wav_file_path: str) -> Dict:
        """
        Run inference on a wav file and return predictions.

        Uses a sliding window approach (typically 3-second segments) to generate predictions.
        The hop size between windows is implementation-specific:
        - FastAIModel uses 1-second hop, producing one confidence per second
        - PodsAIInference uses configurable hop (default 2 seconds)

        Args:
            wav_file_path: Path to the wav file to score.

        Returns:
            Dictionary containing:
                - local_predictions: List of predictions for each hop position.
                                    For binary models: 0 or 1 (0=other, 1=resident (SRKW))
                                    For multi-class models: class ID (e.g., 0-6)
                - local_confidences: List of confidence scores (0.0-1.0) for each hop position.
                                    For binary models: confidence that resident call is present
                                    For multi-class models: whale-call likelihood (1 - P(negative classes))
                                    Used by timestamp correction to locate whale calls.
                                    Note: The number of entries depends on hop_duration. FastAI uses
                                    1-second hop and produces ~N entries for N-second audio. PODS-AI
                                    uses 2-second hop and produces ~N/2 entries.
                - global_prediction: Overall prediction for the entire audio.
                                    For binary models: 0 or 1
                                    For multi-class models: class ID
                - global_prediction_label: (Optional) Human-readable label for global prediction.
                                          Only provided by multi-class models.
                - global_prediction_labels: (Optional) Ordered labels for all
                                           classes that independently meet the global
                                           evidence threshold. Older callers can use
                                           global_prediction_label as the primary label.
                - global_confidence: Overall confidence score (0.0-1.0) for the entire audio.
                                    Typically mean of positive local confidences (0.0-1.0).
                - hop_duration: Actual hop duration in seconds used by the model.
                               This eliminates the need for timestamp correction to infer hop size.
                - segment_duration: Actual segment duration in seconds used by the model.
        """
        raise NotImplementedError("Subclasses must implement predict()")


class FastAIModel(ModelInference):
    """
    FastAI model inference using the aifororcas model.

    This implementation uses the FastAI model architecture from aifororcas-livesystem.
    The model processes 3-second segments with 1-second hop to generate per-second scores.
    The model file (typically model.pkl) should be available in the model_path directory.
    """

    def __init__(self, model_path: str = "./model", model_name: str = "stg2-rn18.pkl",
                 threshold: float = 0.5, min_num_positive_calls_threshold: int = 3,
                 use_gpu: bool = True, smooth_predictions: bool = True,
                 batch_size: int = 32):
        """
        Initialize FastAI model inference.

        Args:
            model_path: Path to directory containing the model file
            model_name: Name of the model file (default: "model.pkl")
            threshold: Confidence threshold for positive predictions (default: 0.5)
            min_num_positive_calls_threshold: Minimum positive predictions for global positive classification.
                                             For short audio clips (< 30 segments), this is automatically
                                             scaled down (1 per 10 segments) to avoid requiring too many
                                             positives from very short clips. The effective threshold is
                                             min(scaled_threshold, min_num_positive_calls_threshold).
                                             Default: 3.
            use_gpu: If True, move the model to CUDA at initialization when a GPU is available.
                     Avoids repeated host-to-device transfers during inference.  If no GPU is
                     present, falls back to CPU automatically.  Default: True.
            smooth_predictions: If True, apply a rolling-average window to smooth per-segment
                                predictions before thresholding.  Default: True.
            batch_size: Number of audio segments to score in each DataLoader batch.
                        Larger values improve GPU throughput; reduce if you run out of memory.
                        Default: 32.
        """
        super().__init__(model_path=model_path)
        self.model = load_model(model_path, model_name)
        self.threshold = threshold
        self.min_num_positive_calls_threshold = min_num_positive_calls_threshold
        self.use_gpu = use_gpu
        self.smooth_predictions = smooth_predictions
        self.batch_size = batch_size

        # Perform GPU device placement once at initialization to avoid repeated
        # host-to-device transfers during inference (improvement from legacy code).
        if self.use_gpu and torch.cuda.is_available():
            self.model.model.cuda()
        else:
            self.model.model.cpu()

    def predict(self, wav_file_path):
        '''
        Function which generates local predictions using wavefile.

        Processes 3-second segments with 1-second hop, optionally applies rolling
        average smoothing, and returns per-second predictions (0.0-1.0) and a global
        confidence (0.0-1.0).
        '''
        wav_path = Path(wav_file_path)

        # Infer clip length.
        max_length = get_duration(path=wav_file_path)
        print(wav_path.name)
        print("Length of Audio Clip:{0}".format(max_length))

        # Generate 3 sec proposals with 1 sec hop length.
        max_start = max(0, int(floor(max_length) - 2))

        # If audio is shorter than 3 seconds, still create one segment from 0 to max_length.
        if max_start == 0 and max_length > 0:
            segments = [(0, min(3, max_length))]
        else:
            segments = [(i, i + 3) for i in range(max_start)]

        # Create a proposal dictionary.
        three_sec_dict = {wav_path.name: segments}

        # Build a mapping from segment filename stem → start_time_s during creation,
        # to avoid re-parsing filenames later.
        start_time_by_stem = {
            wav_path.stem.lower() + '_' + str(begin) + '_' + str(end): begin
            for begin, end in segments
        }

        # Create 3 sec segments from the defined wavefile using proposals built above.
        # "use_a_real_wavname.wav" will generate -> "use_a_real_wavname_0_3.wav", "use_a_real_wavname_1_4.wav" etc. files in local directory.
        # Use TemporaryDirectory for automatic cleanup on exit.
        with tempfile.TemporaryDirectory() as temp_dir:
            local_dir = Path(temp_dir)

            extract_segments(wav_path.parent, three_sec_dict, local_dir, "")

            # Define Audio config needed to create on the fly mel spectograms.
            from audio.data import AudioConfig, SpectrogramConfig, AudioList
            config = AudioConfig(standardize=False,
                                 sg_cfg=SpectrogramConfig(
                                     f_min=0.0,  # Minimum frequency to Display.
                                     f_max=10000,  # Maximum Frequency to Display.
                                     hop_length=256,
                                     n_fft=2560,  # Number of Samples for Fourier.
                                     n_mels=256,  # Mel bins.
                                     pad=0,
                                     to_db_scale=True,  # Converting to DB scale.
                                     top_db=100,  # Top decibel sound.
                                     win_length=None,
                                     n_mfcc=20)
                                 )
            config.duration = 4000  # 4 sec padding or snip.
            config.resample_to = 20000  # Every sample at 20000 frequency.
            config.downmix = True
            config.pad_mode = "zeros-after"  # Make deterministic: zeros at end only

            # Get sorted file list.
            path_list = sorted(
                [str(path) for path in local_dir.iterdir()],
                key=lambda path: Path(path).name,
            )

            # Create AudioList from sorted paths.
            tfms = None
            test = AudioList(path_list, path=local_dir, config=config).split_none().label_empty()
            testdb = test.transform(tfms).databunch(bs=self.batch_size)

            # Score each 3 second clip.
            predictions = []
            for item in testdb.x:
                predictions.append(self.model.predict(item)[2][1])

            # Explicitly release fastai objects to encourage immediate memory reclamation.
            del test
            del testdb
            gc.collect()
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Temp directory and segment files are automatically cleaned up here.

        # Aggregate predictions.

        if not path_list:
            raise RuntimeError(
                f"No audio segments could be extracted from '{wav_path}'. "
                "The file may be corrupt, too short, or pydub/ffmpeg failed to process it."
            )

        # Create a DataFrame.
        prediction = pd.DataFrame({'FilePath': path_list, 'confidence': predictions})

        # Convert prediction to float.
        prediction['confidence'] = prediction.confidence.astype(float)

        # Extract starting time from the pre-built lookup; fall back to filename parsing if needed.
        prediction['start_time_s'] = prediction.FilePath.apply(
            lambda x: start_time_by_stem.get(Path(x).stem, int(Path(x).stem.split('_')[-2]))
        )

        # Sort the file based on start_time_s.
        prediction = prediction.sort_values(
            ['start_time_s']).reset_index(drop=True)

        if self.smooth_predictions:
            # Rolling Window (to average at per second level).
            submission = pd.DataFrame(
                    {
                        'wav_filename': wav_path.name,
                        'duration_s': 1.0,
                        'confidence': list(prediction.rolling(2)['confidence'].mean().values)
                    }
                ).reset_index().rename(columns={'index': 'start_time_s'})

            # Update first row.
            submission.loc[0, 'confidence'] = prediction.confidence.iloc[0]

            # Add last row.
            lastLine = pd.DataFrame({
                'wav_filename': wav_path.name,
                'start_time_s': [submission.start_time_s.max()+1],
                'duration_s': 1.0,
                'confidence': [prediction.confidence.iloc[-1]]
                })
            submission = pd.concat([submission, lastLine], ignore_index=True)
            submission = submission[['wav_filename', 'start_time_s', 'duration_s', 'confidence']]
        else:
            # No smoothing — use raw per-segment predictions directly.
            submission = pd.DataFrame({
                'wav_filename': wav_path.name,
                'start_time_s': prediction['start_time_s'],
                'duration_s': 3.0,  # Each segment is 3 seconds.
                'confidence': prediction['confidence']
            })

        # Initialize output JSON.
        result_json = {}
        result_json = dict(
            submission=submission,
            local_predictions=list((submission['confidence'] > self.threshold).astype(int)),
            local_confidences=list(submission['confidence']),
            hop_duration=1.0,  # FastAI uses 1-second hop
            segment_duration=3.0  # FastAI uses 3-second segments
        )

        # Scale the positive calls threshold based on the number of segments.
        # For every SEGMENT_GROUP_SIZE segments, require at least 1 positive prediction.
        # Cap at min_num_positive_calls_threshold to avoid requiring too many for very long clips.
        total_segments = len(result_json["local_predictions"])
        scaled_threshold = max(1, (total_segments + SEGMENT_GROUP_SIZE - 1) // SEGMENT_GROUP_SIZE)
        effective_threshold = min(scaled_threshold, self.min_num_positive_calls_threshold)

        result_json['global_prediction'] = int(sum(result_json["local_predictions"]) >= effective_threshold)
        result_json['global_confidence'] = submission.loc[(submission['confidence'] > self.threshold), 'confidence'].mean()
        if pd.isnull(result_json["global_confidence"]):
            result_json["global_confidence"] = 0

        return result_json


class DummyModelInference(ModelInference):
    """
    Dummy model inference for testing purposes.

    This implementation returns mock predictions without actually running a model.
    It's useful for testing the timestamp correction logic.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize dummy model."""
        super().__init__(model_path)
        warnings.warn(
            "Using DummyModelInference which returns mock predictions. "
            "For production use, integrate a real model.",
            UserWarning
        )

    def predict(self, wav_file_path: str) -> Dict:
        """
        Generate predictions for testing.

        Returns mock predictions where the middle of the audio has the highest score (0.0-1.0)
        and a global_confidence (0.0-1.0).
        """
        import librosa

        # Load audio to determine duration.
        y, sr = librosa.load(wav_file_path, sr=None)
        duration = len(y) / sr

        # Number of per-second positions.
        num_segments = int(duration)

        if num_segments == 0:
            num_segments = 1

        # Generate mock predictions: highest score in the middle.
        local_predictions = []
        local_confidences = []

        middle_idx = num_segments // 2
        for i in range(num_segments):
            # Create a score that peaks in the middle.
            distance_from_middle = abs(i - middle_idx)
            confidence = max(0.3, 0.9 - (distance_from_middle * 0.1))
            prediction = 1 if confidence > 0.6 else 0

            local_predictions.append(prediction)
            local_confidences.append(round(confidence, 3))

        # Calculate global prediction.
        num_positive = sum(local_predictions)
        global_prediction = 1 if num_positive >= 3 else 0

        # Calculate global confidence.
        if num_positive > 0:
            positive_confidences = [c for p, c in zip(local_predictions, local_confidences) if p == 1]
            global_confidence = sum(positive_confidences) / len(positive_confidences)
        else:
            global_confidence = 0.0

        return {
            "local_predictions": local_predictions,
            "local_confidences": local_confidences,
            "global_prediction": global_prediction,
            "global_confidence": global_confidence
        }


def download_model_if_needed(model_path: str = "./model",
                            model_url: Optional[str] = None) -> bool:
    """
    Download the FastAI model from Azure Blob Storage if it doesn't exist.

    The default model is downloaded from:
    https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip

    Args:
        model_path: Directory where the model should be stored
        model_url: Optional custom URL for the model zip file. If not provided, uses the default model.
                  Can also be set via MODEL_URL environment variable.

    Returns:
        True if model is available (existed or downloaded successfully), False otherwise
    """
    import requests
    import zipfile
    import io

    model_dir = Path(model_path)
    model_file = model_dir / "model.pkl"

    # Check if model already exists.
    if model_file.exists():
        print(f"Model already exists at {model_file}")
        return True

    # Create model directory.
    model_dir.mkdir(parents=True, exist_ok=True)

    # Determine model URL.
    if model_url is None:
        # Check environment variable.
        model_url = os.environ.get(
            "MODEL_URL",
            "https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip"
        )

    print(f"Downloading model from {model_url}...")

    try:
        response = requests.get(model_url, timeout=120)
        response.raise_for_status()

        # Extract zip file with path validation to prevent directory traversal attacks.
        # The zip file contains a "model" directory, so we extract to model_dir.parent.
        print(f"Extracting model to {model_dir.parent}...")
        extract_target = model_dir.parent.resolve()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Validate all file paths before extraction to prevent zip slip attacks.
            for member in zip_ref.namelist():
                # Resolve the member path and ensure it's within the target directory.
                member_path = (extract_target / member).resolve()
                if not str(member_path).startswith(str(extract_target)):
                    raise ValueError(f"Zip file contains unsafe path: {member}")

            # Safe to extract after validation.
            zip_ref.extractall(extract_target)

        # Check if extraction was successful.
        if model_file.exists():
            print(f"Model downloaded and extracted successfully to {model_file}")
            return True
        else:
            print(f"Warning: Model file not found after extraction at {model_file}")
            return False

    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def get_model_inference(model_path: Optional[str] = None, model_type: str = "fastai",
                        auto_download: bool = False, model_url: Optional[str] = None, **kwargs) -> ModelInference:
    """
    Factory function to create a model inference instance.

    Args:
        model_path: Optional path to the model file or directory.
        model_type: Type of model to use. Supports:
            - "fastai": FastAI model from aifororcas-livesystem (default)
            - "podsai": PODS-AI multi-class audio model (supports spectrogram-based checkpoints)
            - "orcahello": OrcaHello SRKW detector (orcasound/orcahello-srkw-detector-v1)
            - "dummy": DummyModelInference for testing
        auto_download: If True and model_type is "fastai", automatically download model if not found
        model_url: Optional custom URL for downloading the model. If not provided, uses default model.
                  Can also be set via MODEL_URL environment variable.
        **kwargs: Additional arguments passed to model constructor (e.g., threshold, min_num_positive_calls_threshold)

    Returns:
        ModelInference instance

    Note:
        The FastAI model can be downloaded from (default):
        https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip

        To use a different model version, set the MODEL_URL environment variable:
        export MODEL_URL=https://trainedproductionmodels.blob.core.windows.net/dnnmodel/YOUR-MODEL.zip

        Example usage:
            # Use FastAI model (default).
            model = get_model_inference(model_path="./model")

            # Use FastAI model with auto-download.
            model = get_model_inference(model_path="./model", auto_download=True)

            # Use PODS-AI model for multi-class classification.
            model = get_model_inference(
                model_path="path/to/podsai/model",
                model_type="podsai",
                threshold=0.5,
                min_num_positive_calls_threshold=3
            )

            # Use OrcaHello SRKW detector (orcasound/orcahello-srkw-detector-v1).
            model = get_model_inference(
                model_path="orcasound/orcahello-srkw-detector-v1",
                model_type="orcahello",
            )

            # Use dummy model for testing.
            model = get_model_inference(model_type="dummy")

            # Use specific FastAI model version.
            model = get_model_inference(
                model_path="./model",
                model_type="fastai",
                auto_download=True,
                model_url="https://trainedproductionmodels.blob.core.windows.net/dnnmodel/NEW-MODEL.zip"
            )
    """
    if model_type == "dummy":
        return DummyModelInference(model_path)
    elif model_type == "podsai":
        # PODS-AI models require explicit model path (no fallback to base model).
        # The base model lacks id2label/label2id config required by PodsAIInference.
        if model_path is None:
            raise ValueError(
                "model_path is required for podsai model type. "
                "Provide a path to a fine-tuned model directory or HuggingFace Hub model ID. "
                "Train a model first using train_podsai_model.py or specify a Hub model."
            )

        # Lazy import to avoid circular dependency.
        # If the provided model_path is an ONNX file, use the ONNX Runtime wrapper.
        if isinstance(model_path, str) and model_path.lower().endswith('.onnx'):
            from podsai_onnx_inference import get_podsai_onnx_inference
            feature_extractor = kwargs.pop('feature_extractor', None)
            return get_podsai_onnx_inference(
                model_path,
                feature_extractor=feature_extractor,
                **kwargs,
            )

        from podsai_inference import get_podsai_inference

        return get_podsai_inference(model_path, **kwargs)
    elif model_type == "orcahello":
        # OrcaHello SRKW detector using new inference pipeline.
        # Defaults to orcasound/orcahello-srkw-detector-v1 on HuggingFace Hub.
        from orcahello_inference import get_orcahello_srkw_inference

        if model_path is None:
            from orcahello_inference import OrcaHelloSRKWInference
            model_path = OrcaHelloSRKWInference.DEFAULT_MODEL_PATH

        # Filter kwargs to only include parameters supported by OrcaHelloSRKWInference.
        orcahello_kwargs = {
            k: v for k, v in kwargs.items() if k in ("config", "model_revision")
        }

        return get_orcahello_srkw_inference(model_path, **orcahello_kwargs)
    elif model_type == "fastai":
        if model_path is None:
            model_path = "./model"

        # Check if model exists, download if requested.
        model_file = Path(model_path) / "model.pkl"
        if not model_file.exists() and auto_download:
            if not download_model_if_needed(model_path, model_url):
                raise FileNotFoundError(
                    f"Failed to download model. Please manually download from:\n"
                    f"{model_url or os.environ.get('MODEL_URL', 'https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip')}\n"
                    f"and extract to {model_path}"
                )

        # Filter kwargs to only include parameters supported by FastAIModel.
        # FastAIModel accepts: threshold, min_num_positive_calls_threshold, use_gpu, smooth_predictions, batch_size.
        fastai_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ('threshold', 'min_num_positive_calls_threshold', 'use_gpu', 'smooth_predictions', 'batch_size')
        }

        return FastAIModel(model_path=model_path, model_name="model.pkl", **fastai_kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: 'dummy' (for testing), 'fastai' (for production), "
            f"'podsai' (for PODS-AI audio-classification models), "
            f"'orcahello' (for OrcaHello SRKW detector). "
            f"For production use with FastAI model, set MODEL_TYPE=fastai and ensure model is available."
        )
