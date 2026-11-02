# Copilot Instructions for PODS-AI

## Project Overview

PODS-AI (Programmatic Orca Detection System using Artificial Intelligence) is a Python project for detecting and classifying orca and other whale vocalizations from audio recordings provided by the [Orcasound](https://www.orcasound.net/) hydrophone network.

## Repository Layout

```
pods-ai/
├── .github/
│   ├── copilot-instructions.md   # This file
│   ├── dependabot.yml            # Dependabot updates for GitHub Actions
│   ├── renovate.json             # Renovate configuration
│   └── workflows/
│       ├── check_csv.yml                          # CI: regenerates and validates detections.csv
│       ├── generate_model_comparison.yml           # Regenerates model comparison and README.md
│       ├── LiveInferenceSystem.yaml               # CI: build and smoke-test the Docker container
│       ├── LiveInferenceSystem-deploy.yaml        # CD: push container image to ACR on tag
│       ├── LiveInferenceSystem-deploy-configmaps.yaml  # CD: apply K8s configmaps on change
│       ├── validate-readme.yml                    # CI: verifies README.md is generated from its inputs
│       └── validate-yaml.yml                      # CI: yamllint on all YAML files
├── external/                     # Git submodules such as OrcaHello
├── LiveInferenceSystem/          # Docker container for live inference (mirrors OrcaHello's InferenceSystem/)
│   ├── Dockerfile                # Container definition (build from repo root)
│   ├── .dockerignore
│   ├── pyproject.toml            # Production Python dependencies (uses uv)
│   ├── uv.lock                   # Locked dependency versions
│   ├── deploy/                   # Kubernetes manifests for each hydrophone location
│   │   ├── <location>.yaml          # Deployment spec
│   │   └── <location>-configmap.yaml  # Hydrophone-specific orchestrator config
│   └── tests/
│       └── orch_configs/         # Orchestrator config files for smoke tests
│           └── LiveHLS/
├── output/                       # Generated CSV, WAV, PNG, and model artifacts
├── tests/                        # Pytest test suite
├── LICENSE
├── README.md                     # User and contributor documentation
├── README.md.template            # README source with a model comparison placeholder
├── patch_fastai_audio.bat        # Windows helper for fastai_audio compatibility
├── patch_fastai_audio.sh         # Shell helper for fastai_audio compatibility
├── pods-ai.pyproj                # Visual Studio Python project metadata
├── pods-ai.sln                   # Visual Studio solution
├── requirements.txt
└── src/
    ├── add_samples.py              # Segment and label new samples
    ├── audio_utils.py              # Shared audio helpers
    ├── build-readme.py             # Generate README.md from its template and model comparison
    ├── compare_models.py           # Compare model performance on testing data
    ├── download_wavs.py            # Step 4: download wav files
    ├── extract_training_samples.py # Step 3: detections.csv → initial training/testing samples
    ├── get_best_timestamp.py       # Timestamp correction helper
    ├── LiveInferenceOrchestrator.py  # Live HLS inference loop (entry point for Docker container)
    ├── merge_training_samples.py   # Step 4: initial/manual samples → training samples
    ├── make_csv.py                 # Step 1: query APIs → detections.csv
    ├── make_spectrograms.py        # Step 6: wav → PNG spectrograms
    ├── manual_samples_utils.py     # Shared manual-sample CSV helpers
    ├── model_inference.py          # Common inference interface
    ├── orcahello_inference.py      # OrcaHello inference adapter
    ├── orcasite_feeds.py           # Orcasite feed helpers
    ├── podsai_inference.py         # PODS-AI inference adapter
    ├── process_false_negatives.py  # Helper to review false negatives
    ├── process_false_positives.py  # Helper to review false positives
    ├── process_humpback_wavs.py    # Step 2: process humpback source files
    ├── run_inference.py            # Run model inference on a WAV file
    ├── spectrogram_visualizer.py   # Spectrogram helper
    └── train_podsai_model.py       # Train a PODS-AI model
```

## Data Sources

- **Orcasite API** (`https://live.orcasound.net/api/json/`) – Human and machine detections, feed metadata.
- **OrcaHello** – Azure Cosmos DB (`aifororcasmetadatastore`) storing Southern Resident Killer Whale (SRKW) review results; accessed via the `COSMOS_KEY` secret.
- **Orcasound S3** (`audio-orcasound-net`, region `us-west-2`) – HLS audio streams and wav segments.

## Key Environment Variables / Secrets

| Variable | Default | Purpose |
|---|---|---|
| `COSMOS_URL` | `https://aifororcasmetadatastore.documents.azure.com:443/` | Cosmos DB endpoint |
| `COSMOS_KEY` | *(required secret)* | Cosmos DB primary key |
| `COSMOS_DB` | `predictions` | Cosmos DB database name |
| `COSMOS_CONTAINER` | `metadata` | Cosmos DB container name |

## Coding Conventions

Follow the coding conventions documented in [CONTRIBUTING.md](../CONTRIBUTING.md).
When reviewing code, ensure it follows those coding conventions.

## pods-ai Pipeline

The scripts in `src/` are meant to be run in order:

1. `make_csv.py` – Queries Orcasite and OrcaHello APIs; writes `output/csv/detections.csv`.
2. `process_humpback_wavs.py` – Processes humpback signal files from `signals-humpback_*.wav`; extracts 2-second segments.
3. `extract_training_samples.py` – Reads `detections.csv`; writes `output/csv/initial_training_samples.csv` and `output/csv/testing_samples.csv` (adjusts humpback count based on existing signal files).
4. `merge_training_samples.py` – Reads `initial_training_samples.csv` and `manual_samples.csv`; writes `output/csv/training_samples.csv`.
5. `download_wavs.py` – Reads `training_samples.csv`; downloads wav files into subdirectories under `output/wav/`.
6. `make_spectrograms.py` – Generates a PNG spectrogram alongside each wav file.

Detection labels: `resident`, `transient`, `humpback`, `other`.  
Classification kinds: `tp_human_only`, `tp_machine_only`, `fp_machine_only`, `tp_both`, `skip`.

## CI / CD

- FFmpeg setup is platform-specific until the Ubuntu action path in https://github.com/FedericoCarboni/setup-ffmpeg/issues/35 is resolved and revalidated (OrcaHello #642 currently validates native Ubuntu FFmpeg): use native `apt-get install -y ffmpeg` on Ubuntu, and the pinned `FedericoCarboni/setup-ffmpeg` v2 action with `github.token` on Windows.
- **check_csv.yml** – Runs on every PR; re-executes `make_csv.py`, `extract_training_samples.py`, and, when relevant files changed, `merge_training_samples.py`, then asserts no diff in the committed CSV files (requires `COSMOS_KEY` secret for `make_csv.py`).
- **LiveInferenceSystem.yaml** – Runs on every PR touching `LiveInferenceSystem/**` or the orchestrator source files; builds the Docker image from the repo root and runs a LiveHLS smoke test.
- **LiveInferenceSystem-deploy.yaml** – Triggered by a `LiveInferenceSystem.v#.#.#` tag push; builds and pushes the container image to `orcaconservancycr.azurecr.io/pods-ai-live-inference-system`.
- **LiveInferenceSystem-deploy-configmaps.yaml** – Triggered when `LiveInferenceSystem/deploy/*-configmap.yaml` files change on `main`; applies updated configmaps and restarts the affected deployments via `kubectl`.
- **generate_model_comparison.yml** – Runs model comparison and regenerates `README.md` from `README.md.template` and `model-comparison.txt`.
- **validate-readme.yml** – Runs on PRs changing `README.md.template` or `model-comparison.txt` and verifies that `README.md` matches `python src/build-readme.py`.
- **validate-yaml.yml** – Runs `yamllint` against all YAML files using the rules in `.yamllint.yml` (line-length disabled, truthy check-keys disabled).
- **dependabot.yml** – Weekly updates for GitHub Actions dependencies.
- **renovate.json** – Tracks additional dependency updates, including the pinned PODS-AI test model revision.

## LiveInferenceSystem Container

`LiveInferenceSystem/` mirrors the structure of OrcaHello's `InferenceSystem/` and packages the live inference orchestrator as a Docker container for AKS deployment.

- **Docker build context**: repo root (not `LiveInferenceSystem/`), so the Dockerfile accesses both `src/` and `external/orcahello/InferenceSystem/src/orcasound_hls/`.
- **Container image name**: `pods-ai-live-inference-system` (different from OrcaHello's `live-inference-system` to allow side-by-side coexistence).
- **K8s deployment name**: `pods-ai-inference-system` (different from OrcaHello's `inference-system`); both share the same namespaces (e.g. `bush-point`, `orcasound-lab`).
- **ConfigMap name**: `pods-ai-hydrophone-configs` (mounted at `/config/config.yml`).
- **Secrets**: `pods-ai-inference-system` secret per namespace with `AZURE_COSMOSDB_PRIMARY_KEY`, `AZURE_STORAGE_CONNECTION_STRING`, and `INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING`.
- The `external/orcahello` submodule must be initialized before building the Docker image.

## Dependencies

Install project requirements before running scripts:

```bash
pip install -r requirements.txt
```

Key packages: `azure-cosmos`, `boto3`, `librosa`, `ffmpeg-python`, `matplotlib`, `numpy`, `requests`, `pytz`.  
