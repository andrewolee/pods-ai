# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Script to generate spectrograms from .wav files.

This script traverses the output/wav directory (and its subdirectories),
finds all .wav files saved by download_wavs.py, and creates a spectrogram
(.png file) in the corresponding output/png subdirectory.

Usage:
    python src/make_spectrograms.py
"""
import argparse
import sys
from pathlib import Path
from typing import List


def find_wav_files(root_dir: Path) -> List[Path]:
    """
    Recursively find all .wav files in a directory and its subdirectories.
    
    Parameters:
        root_dir: Root directory to search.
    
    Returns:
        List[Path]: List of paths to .wav files.
    """
    wav_files = []
    for path in root_dir.rglob("*.wav"):
        if path.is_file():
            wav_files.append(path)
    return wav_files


def process_wav_files(wav_root: Path, png_root: Path, skip_existing: bool = True):
    """
    Process all .wav files in wav_root and generate spectrograms under png_root.

    For each .wav file found under wav_root, the corresponding .png spectrogram
    is saved under png_root in the same relative subdirectory.

    Parameters:
        wav_root: Root directory containing .wav files in subdirectories.
        png_root: Root directory where .png spectrograms will be saved.
        skip_existing: If True, skip .wav files that already have a .png spectrogram.
    """
    import spectrogram_visualizer

    wav_files = find_wav_files(wav_root)
    
    if not wav_files:
        print(f"No .wav files found in {wav_root}")
        return
    
    print(f"Found {len(wav_files)} .wav file(s) to process")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for wav_path in wav_files:
        # Compute the corresponding PNG path under png_root.
        relative = wav_path.relative_to(wav_root)
        png_path = png_root / relative.with_suffix('.png')
        if skip_existing and png_path.exists():
            print(f"Skipping (spectrogram already exists): {wav_path}")
            skipped_count += 1
            continue
        
        png_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Processing: {wav_path}")
            spec_path = spectrogram_visualizer.write_spectrogram(wav_path, output_path=png_path)
            print(f"  -> Created spectrogram: {spec_path}")
            processed_count += 1
        except Exception as e:
            print(f"  -> ERROR processing {wav_path}: {type(e).__name__}: {str(e)}")
            error_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")


def main() -> int:
    """Generate spectrograms for one WAV file or the default directory."""
    parser = argparse.ArgumentParser(
        description="Generate spectrograms for a WAV file or files in output/wav."
    )
    parser.add_argument(
        "wav_file",
        nargs="?",
        type=Path,
        help="Generate a spectrogram for this WAV file only.",
    )
    args = parser.parse_args()

    if args.wav_file is not None:
        if not args.wav_file.is_file():
            print(f"Error: WAV file not found: {args.wav_file}")
            return 1

        import spectrogram_visualizer

        try:
            print(f"Processing: {args.wav_file}")
            spec_path = spectrogram_visualizer.write_spectrogram(args.wav_file)
            print(f"  -> Created spectrogram: {spec_path}")
        except Exception as e:
            print(f"  -> ERROR processing {args.wav_file}: {type(e).__name__}: {str(e)}")
            return 1
        return 0

    wav_root = Path("output/wav")
    png_root = Path("output/png")

    if not wav_root.exists():
        print(f"Error: Directory not found: {wav_root}")
        print("Please ensure the output/wav directory exists.")
        return 1

    process_wav_files(wav_root, png_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
