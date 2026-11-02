# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
# Adapted from:
# https://github.com/orcasound/orcahello/blob/main/InferenceSystem/src/spectrogram_visualizer.py
import gc
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf


# Fixed image dimensions for consistent output.
_VIZ_IMAGE_HEIGHT = 480
_VIZ_IMAGE_WIDTH = 1280
_FREQ_LABEL_FONT_SIZE = 8
_COLORMAP = "Blues"
_N_FFT = 4096
_HOP_LENGTH = 1024
_TOP_DB = 100


def _freq_label(hz):
    """Format a frequency value as a compact human-readable string."""
    if hz >= 1000:
        khz = hz / 1000
        return f"{khz:.0f}k" if khz == int(khz) else f"{khz:.1f}k"
    return f"{hz:.0f}"


def _pick_freq_ticks(f_min, f_max):
    """Choose ~5-8 log-spaced tick positions between f_min and f_max.

    Args:
        f_min: Minimum frequency in Hz. Must be positive.
        f_max: Maximum frequency in Hz.
    """
    candidates = [
        40, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 48000,
    ]
    ticks = [f for f in candidates if f_min <= f <= f_max]
    if not ticks:
        ticks = np.geomspace(max(f_min, 1), f_max, num=6).tolist()
    return ticks


def _render_spectrogram(
    spectrogram_np,
    times_np,
    freqs_np,
    output_path,
    width_px=_VIZ_IMAGE_WIDTH,
    height_px=_VIZ_IMAGE_HEIGHT,
    dpi=100,
):
    """Render a mel spectrogram array to a PNG file.

    Args:
        spectrogram_np: 2D numpy array (n_mels, n_frames), dB-scaled.
        times_np: 1D array of time values (seconds).
        freqs_np: 1D array of frequency values (Hz).
        output_path: Path to save PNG.
        width_px: Image width in pixels.
        height_px: Image height in pixels.
        dpi: Dots per inch.
    """
    fig, ax = plt.subplots(1, 1, figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    try:
        ax.axis("off")
        ax.set_position([0.0, 0.0, 1.0, 1.0])

        # Freqs are log-spaced; give each mel bin equal pixel height.
        bin_indices = np.arange(len(freqs_np))
        ax.pcolormesh(
            times_np,
            bin_indices,
            spectrogram_np,
            shading="auto",
            cmap=_COLORMAP,
        )

        f_min, f_max = float(freqs_np[0]), float(freqs_np[-1])
        ticks = _pick_freq_ticks(f_min, f_max)
        x_pos = times_np[0] + (times_np[-1] - times_np[0]) * 0.005

        for freq in ticks:
            bin_idx = float(np.searchsorted(freqs_np, freq))
            ax.text(
                x_pos,
                bin_idx,
                _freq_label(freq),
                color="white",
                fontsize=_FREQ_LABEL_FONT_SIZE,
                fontweight="bold",
                va="center",
                ha="left",
                path_effects=[
                    matplotlib.patheffects.Stroke(linewidth=2, foreground="black"),
                    matplotlib.patheffects.Normal(),
                ],
            )

        fig.savefig(output_path, bbox_inches=None, pad_inches=0)
    finally:
        plt.close(fig)


def _compute_mel_for_clip(wav_file_path, native_sr):
    """Load audio and compute a mel spectrogram using librosa.

    Args:
        wav_file_path: Path to WAV file.
        native_sr: Native sample rate to use (no resampling).

    Returns:
        Tuple of (spectrogram_np, times_np, freqs_np) where spectrogram_np is a
        2D dB-scaled mel spectrogram (n_mels, n_frames), times_np is a 1D array
        of frame times in seconds, and freqs_np is a 1D array of mel center
        frequencies in Hz.
    """
    n_mels = _VIZ_IMAGE_HEIGHT
    f_min = 20.0
    f_max = native_sr // 2

    y, _ = librosa.load(wav_file_path, sr=native_sr, mono=True)

    with warnings.catch_warnings():
        # High n_mels relative to n_fft can leave empty top bins; that is intentional
        # for 1:1 pixel rendering and simply renders as a low-intensity color.
        warnings.filterwarnings("ignore", message="At least one mel filterbank")
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=native_sr,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )

    spectrogram_np = librosa.power_to_db(mel, ref=np.max, top_db=_TOP_DB)
    n_frames = spectrogram_np.shape[1]
    times_np = librosa.frames_to_time(
        np.arange(n_frames), sr=native_sr, hop_length=_HOP_LENGTH, n_fft=_N_FFT
    )
    freqs_np = librosa.mel_frequencies(n_mels=n_mels, fmin=f_min, fmax=f_max)

    return spectrogram_np, times_np, freqs_np


def write_spectrogram(wav_file_path, output_path=None):
    """Generate a spectrogram PNG from a WAV file.

    Uses the native sample rate and visualization-optimized mel parameters
    for clear human-readable spectrogram output.

    Args:
        wav_file_path: Path to the input WAV file.
        output_path: Optional path for the output PNG file. If not provided,
            the PNG is saved alongside the WAV file with the same base name.

    Returns:
        Path to the output PNG file.
    """
    wav_file_path = str(wav_file_path)
    directory_name = os.path.dirname(wav_file_path)
    candidate_name = os.path.basename(wav_file_path)
    candidate_name_without_extension = os.path.splitext(candidate_name)[0]

    if output_path is not None:
        spec_output_path = str(output_path)
    else:
        spec_output_path = os.path.join(
            directory_name, candidate_name_without_extension + ".png"
        )

    native_sr = sf.info(wav_file_path).samplerate
    spectrogram_np, times_np, freqs_np = _compute_mel_for_clip(wav_file_path, native_sr)
    _render_spectrogram(spectrogram_np, times_np, freqs_np, spec_output_path)

    del spectrogram_np, times_np, freqs_np
    gc.collect()

    return spec_output_path
