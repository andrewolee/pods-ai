# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import make_spectrograms


def test_help_displays_usage(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["make_spectrograms.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        make_spectrograms.main()

    assert exc_info.value.code == 0
    help_output = capsys.readouterr().out
    assert "usage:" in help_output
    assert "Generate spectrograms for a WAV file or files in output/wav." in help_output
    assert "wav_file" in help_output


def test_main_processes_only_specified_wav_file(monkeypatch, tmp_path: Path) -> None:
    wav_file = tmp_path / "single.wav"
    wav_file.touch()
    processed_files = []
    monkeypatch.setattr("sys.argv", ["make_spectrograms.py", str(wav_file)])
    monkeypatch.setitem(
        sys.modules,
        "spectrogram_visualizer",
        SimpleNamespace(
            write_spectrogram=lambda path: processed_files.append(path) or "single.png"
        ),
    )

    assert make_spectrograms.main() == 0
    assert processed_files == [wav_file]


def test_main_processes_default_directory_without_wav_argument(
    monkeypatch, tmp_path: Path
) -> None:
    wav_root = tmp_path / "output" / "wav"
    wav_root.mkdir(parents=True)
    processed_roots = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["make_spectrograms.py"])
    monkeypatch.setattr(
        make_spectrograms,
        "process_wav_files",
        lambda wav_root, png_root: processed_roots.append((wav_root, png_root)),
    )

    assert make_spectrograms.main() == 0
    assert processed_roots == [(Path("output/wav"), Path("output/png"))]
