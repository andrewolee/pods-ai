# run_inference.py

Run model inference on a wav file and display the global prediction, confidence score,
and per-class probabilities.

For PODS-AI models:

- For a whale-class winner, `global_confidence` is the average whale-call likelihood across windows predicted as that class.
- For a negative/background winner, `global_confidence` is the average model probability for that class across all windows.
- `per_class_probabilities` reports the average whale-call likelihood
(`local_confidence`) for windows predicted as each class.
- Whale-likelihood values are diagnostic statistics and should not be interpreted
as class probabilities.

For the FastAI binary model, `resident = global_confidence` and `other = 1 - global_confidence`.

```
usage: python run_inference.py [wav_file]
       [--node-name NODE_NAME]
       [--end-timestamp-str YYYY_MM_DD_HH_MM_SS_PST | --start-timestamp-utc YYYY-MM-DDTHH:MM:SSZ]
       [--set {training,testing}]
       [--model {podsai,fastai,orcahello}] [--type {ast,wav2vec2}] [--model-path PATH]
```

| Argument | Description |
|---|---|
| `wav_file` | Path to the wav file to score |
| `--node-name` | Hydrophone feed node name (for download mode) |
| `--end-timestamp-str` | PST **end** timestamp used with `--node-name` (format: `YYYY_MM_DD_HH_MM_SS_PST`) |
| `--start-timestamp-utc` | UTC **start** timestamp used with `--node-name` (format: `YYYY-MM-DDTHH:MM:SSZ`) |
| `--set` | With `--node-name`: `training` downloads a 3-second clip, `testing` (default) downloads a 60-second clip. Ignored when `wav_file` is given |
| `--model` | Model type: `podsai` (default), `fastai`, or `orcahello` |
| `--type` | PODS-AI model variant used with `--model podsai`: `ast` (default) or `wav2vec2` (older model variant). These map to the currently pinned revisions in `src/run_inference.py` |
| `--model-path` | Path to model directory, HuggingFace Hub model ID, or PODS-AI ONNX file. Defaults to `./model` for `fastai`; defaults to `orcasound/orcahello-srkw-detector-v1` for `orcahello`; defaults to `davethaler/whale-call-detector` for `podsai` |

When using `--node-name`, provide exactly one timestamp argument:
`--end-timestamp-str` or `--start-timestamp-utc`.

## ONNX inference

ONNX inference is supported for PODS-AI models. Install the optional dependencies
from the repository root:

```bash
pip install -r requirements-onnx.txt
```

Then pass a PODS-AI-compatible ONNX file to `--model-path`:

```bash
cd src
python run_inference.py sample.wav --model podsai --model-path ../podsai.onnx
```

**Example — PODS-AI model**

```bash
cd src
python run_inference.py sample.wav --model podsai
```

Output:
```
Model type: podsai
Global prediction: resident (confidence: 0.7000)
Prediction time: 1.23s

Average whale likelihood by predicted class:
  humpback: 0.0000
  human: 0.0000
  jingle: 0.0500
  resident: 0.7000
  transient: 0.0000
  vessel: 0.0800
  water: 0.0600
```

For multi-class PODS-AI inference, `global_prediction_labels` also reports every
class that independently meets the evidence threshold. The legacy
`global_prediction_label` remains the primary label for compatibility.

## Compatibility / Migration

- Use `global_prediction_labels` for new consumers; it is an ordered list and may
  contain multiple classes.
- `global_prediction_label` is retained for single-label compatibility and selects
  the first class in the priority order: whale classes, then bird/jingle, then
  background classes. Within each group, labels are ordered by event count, then
  average whale-call likelihood, then label name.
- Empty or error responses always return `global_prediction_labels: []`, which
  clients should interpret as no classes.

**Example — FastAI model**

```bash
cd src
python run_inference.py sample.wav --model fastai --model-path ../model
```

Output:
```
Model type: fastai
Global prediction: resident (confidence: 0.7500)
Prediction time: 0.85s

Per-class probabilities:
  other: 0.2500
  resident: 0.7500
```

**Example — OrcaHello SRKW Detector**

Uses the [`orcasound/orcahello-srkw-detector-v1`](https://huggingface.co/orcasound/orcahello-srkw-detector-v1)
model from HuggingFace Hub. This is a binary SRKW (Southern Resident Killer Whale) detector
based on the new OrcaHello inference pipeline (ResNet50 + mel spectrograms, no fastai_audio dependency).

The model implementation is loaded from the `orcasound/orcahello` submodule. Initialize it first:

```bash
git submodule update --init external/orcahello
```

Then run inference:

```bash
cd src
python run_inference.py sample.wav --model orcahello
```

Output:
```
Model type: orcahello
Global prediction: resident (confidence: 0.8000)
Prediction time: 0.92s

Per-class probabilities:
  other: 0.2000
  resident: 0.8000
```

You can compare results between models by running each on the
same file and comparing the output.
