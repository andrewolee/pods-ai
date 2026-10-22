# `export_podsai_onnx.py`

## Purpose

`export_podsai_onnx.py` exports a PODS-AI Hugging Face model to an ONNX file for
CPU inference. The current exporter supports AST spectrogram-input models; raw
audio models such as Wav2Vec2 are not supported. The exported model has dynamic
batch size and fixed spectrogram dimensions.

## Requirements

Install the optional ONNX dependencies and the project requirements:

```bash
pip install -r requirements.txt
pip install -r requirements-onnx.txt
```

## Usage

Run the command from the repository root:

```bash
python src/export_podsai_onnx.py \
    --model davethaler/whale-call-detector \
    --revision <revision> \
    --out podsai.onnx
```

The exporter defaults to the latest model at the Hugging Face Hub, writes
`podsai.onnx` in the current directory, and exports a representative input
shape of 1 batch × 300 frames × 128 mel bins. These frame dimensions match the
default 3-second PODS-AI inference window; the batch dimension remains dynamic.
Pin `--revision` to a specific model commit when reproducible exports are
needed.

| Option | Description |
| --- | --- |
| `--model` | Hugging Face model ID or local model path. Default: `davethaler/whale-call-detector`. |
| `--revision` | Optional Hugging Face model revision. |
| `--out` | Output ONNX file path. Default: `podsai.onnx`. |
| `--device` | Export device: `cpu` (default) or `cuda`. |
| `--batch-size` | Representative batch size. Default: `1`. |
| `--frames` | Representative number of spectrogram frames. Default: `300`. |
| `--mel-bins` | Representative number of mel bins. Default: `128`. |

The model's time and mel dimensions are fixed at export time. If you change
`--frames` or `--mel-bins`, inference inputs must use those same dimensions.

## Using the exported model

Use the ONNX file with `run_inference.py`:

```bash
python src/run_inference.py sample.wav --model podsai --model-path podsai.onnx
```

ONNX inference also loads the feature-extractor configuration from the matching
Hugging Face model. Install `requirements-onnx.txt` before running inference.
