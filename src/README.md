# Helper Scripts

- **spectrogram_visualizer.py**: Adapted from [aifororcas-livesystem](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/spectrogram_visualizer.py)
- **model_inference.py**: Provides model inference interface for scoring audio samples
- **orcasite_feeds.py**: Lightweight module providing the `OrcasiteFeed` dataclass and
  `get_orcasite_feeds()` helper. Depends only on `requests` — no `azure-cosmos` — so
  scripts that only need the feeds REST API (e.g. `add_samples.py`) can import it.
- [**add_samples.py**](../docs/add-samples.md): Splits a WAV file into 3-second segments (2-second hop), saves each
  segment to a `new/` directory using the standard filename convention, and prints the
  predicted class for each segment. Useful for labelling new recordings and adding them
  to the training set.

Key dependencies:
- `boto3`: For accessing S3 audio files
- `ffmpeg-python`: For audio processing
- `librosa>=0.10.0`: For audio analysis
- `m3u8`: For HLS stream parsing
- `pytz`: For timezone handling
- `fastai==1.0.61`: For FastAI model support
- `torch>=2.1.0`: PyTorch deep learning framework
- `torchvision>=0.16.0`: Computer vision models and utilities
- `torchaudio>=2.1.0`: Audio processing for PyTorch
- `soundfile`: Audio file I/O
- `fastai_audio`: FastAI audio extensions (from GitHub)
- `pandas`, `pydub`: Data processing and audio manipulation
