# `compare_models.py`

## Scope

`compare_models.py` evaluates model predictions for WAV files below
`output/testing-wav`. Each immediate subdirectory is treated as the expected
category. The command reports accuracy, macro F1 for whale classes,
per-class false-positive and false-negative rates, prediction time, and a
primary-label confusion matrix.

The command performs end-to-end inference on 60-second WAV files. Its results
are not the trainer's held-out split metrics.

## Command syntax

```text
python src/compare_models.py [options]
```

| Option | Description |
| --- | --- |
| `--wav-dir PATH` | Root directory of category subdirectories. Default: `output/testing-wav`. |
| `--models MODEL_LIST` | Comma-separated built-in names. Include `custom` to evaluate the custom PODS-AI model. Default: `fastai,orcahello,wav2vec2,oldpodsai,podsai`. |
| `--custom-model-path PATH_OR_HF_ID` | Local custom model file/directory or Hugging Face repository ID. Required when `custom` is in `--models`; supplying it without `custom` adds `custom` automatically. |
| `--custom-model-revision REVISION` | Optional revision for a custom Hugging Face model. Ignored for local paths. |
| `--fastai-model-path PATH` | FastAI model directory. Default: `model`. |
| `--orcahello-model-path PATH_OR_HF_ID` | OrcaHello model path or Hub ID. Default: `orcasound/orcahello-srkw-detector-v1`. |
| `--podsai-model-path PATH_OR_HF_ID` | PODS-AI model path or Hub ID for `wav2vec2`, `oldpodsai`, and `podsai`. Default: `davethaler/whale-call-detector`. |
| `--podsai-model-revision REVISION` | Revision for the built-in `podsai` model. |
| `--max-samples N` | Evaluate at most `N` WAV files. `N` must be positive. |
| `--category NAME` | Evaluate only the named category. |

Built-in model names map to these inference backends:

| Name | Backend | Revision behavior |
| --- | --- | --- |
| `fastai` | FastAI | No revision |
| `orcahello` | OrcaHello | No revision |
| `wav2vec2` | PODS-AI | Pinned Wav2Vec2 revision |
| `oldpodsai` | PODS-AI | Pinned previous revision |
| `podsai` | PODS-AI | `--podsai-model-revision` |

Unknown names cause an error before model loading. The WAV directory must
exist and contain at least one category WAV file. Supported background
categories include `other`, `abiotic`, `water`, `vessel`, `jingle`, `human`,
and `bird`.

## Custom model specification

The `custom` model name selects the PODS-AI backend. Its value is supplied by
`--custom-model-path`; the option accepts one local path or one Hugging Face
repository ID. A command that lists `custom` without a path is rejected. A
path supplied without `custom` adds `custom` to the model list. Inline values
such as `custom:./models/my-detector` are not supported.

Custom models are PODS-AI-compatible multi-class models. They use the same
category-membership evaluation as `wav2vec2`, `oldpodsai`, and `podsai`,
including the confusion matrix, F1, and FP/FN calculations.

### Inference contract

The model artifact must be loadable by:

```python
get_model_inference(model_type="podsai", model_path=VALUE)
```

The loaded PODS-AI backend must expose a callable `predict()` method. Its
input and result contract is:

- Input audio is decoded as mono at 16 kHz.
- Analysis uses 3-second windows with a 2-second hop.
- Inference uses a confidence threshold of `0.5` by default. The default
  positive-event threshold is 2, scaled down for shorter clips.
- The Hugging Face audio-classification configuration provides `id2label` and
  `label2id` and includes at least one supported background label: `other`,
  `abiotic`, `water`, `vessel`, `jingle`, `human`, or `bird`.
- The prediction mapping contains `global_prediction_label`,
  `global_confidence`, `local_predictions`, `local_confidences`, and
  `per_class_probabilities`. `global_prediction_labels` may also be present.
- Confidence values are numeric values in the inclusive range `[0, 1]`.

An artifact that cannot be loaded by this backend or a factory result without
`predict()` is rejected before any test sample is evaluated.

### Model value resolution

1. If `VALUE` resolves to an existing file or directory, it is passed to the
   loader as a local model path. The command expands `~` before loading.
2. A missing value beginning with `./`, `../`, `~`, `/`, `\`, or a Windows
   drive letter is treated as a missing local path and rejected before loading.
3. Other missing values are passed to the Hugging Face loader as repository
   IDs. `--custom-model-revision` is forwarded for these values; it is ignored
   for local paths.

Invalid repository IDs, unavailable repositories, permission failures, and
incompatible artifacts are reported as a single custom-model loading error
that includes the loader exception type and message. A Python traceback is
not printed by the CLI.

## Evaluation definitions

- `fastai` and `orcahello` are binary resident-versus-other models. A sample
  is correct when resident status matches the expected category.
- `wav2vec2`, `oldpodsai`, `podsai`, and `custom` use the expected category's
  membership in `global_prediction_labels`; older output falls back to
  `global_prediction_label`.
- Macro F1 covers `resident`, `transient`, and `humpback` classes present in
  the evaluated samples, either as the correct class or the predicted class.
- `[R|T|H]FP%` is the fraction of non-class samples predicted as that class.
- `[R|T|H]FN%` is the fraction of class samples predicted as another class.
- The displayed confusion matrix uses `global_prediction_label`. Multi-label
  F1 and FP/FN calculations use every emitted global label.

## Examples

Evaluate the built-in models on all available samples:

```bash
python src/compare_models.py
```

Evaluate a local custom model alongside FastAI:

```bash
python src/compare_models.py --models fastai,custom --custom-model-path ./models/my-detector
```

Evaluate a Hugging Face custom model with a pinned revision:

```bash
python src/compare_models.py \
    --models podsai,custom \
    --custom-model-path my-org/my-whale-detector \
    --custom-model-revision d66c75205e3d229aa5fe875cb27bc90304654fcd
```

Limit evaluation to resident samples and ten files:

```bash
python src/compare_models.py --category resident --max-samples 10
```
