# make_spectrograms.py

Generate PNG spectrograms for WAV files under `output/wav`, or for one specified
WAV file. In batch mode, the script searches recursively and saves each
spectrogram under `output/png` in the matching relative subdirectory. Existing
spectrograms are skipped. For a single WAV file, the PNG is saved alongside the
input file.

Run from the repository root:

```bash
python src/make_spectrograms.py
```

Generate a spectrogram for only one WAV file:

```bash
python src/make_spectrograms.py path/to/audio.wav
```

Display command-line usage:

```bash
python src/make_spectrograms.py --help
```
