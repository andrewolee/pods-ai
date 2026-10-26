# process_testing_set_mispredictions.py

Scans `output/csv/testing_60s_samples.csv`,
runs PODS-AI on each corresponding `output/testing-wav/<category>/...wav`, and
proposes `training_3s_samples.csv` rows plus `testing_60s_samples.csv` rows to
remove when non-adjacent high-confidence (>0.80) whale segments are found.

```
usage: process_testing_set_mispredictions.py [-h] [--testing-csv TESTING_CSV] [--wav-dir WAV_DIR]
                                             [--model-path MODEL_PATH] [--model-revision MODEL_REVISION]
                                             [--min-confidence MIN_CONFIDENCE] [--category CATEGORY]

Propose training_3s_samples rows from testing_60s_samples mispredictions and list testing rows to remove.

options:
  -h, --help            show this help message and exit
  --testing-csv TESTING_CSV
                        Path to testing_60s_samples.csv.
  --wav-dir WAV_DIR     Root directory containing downloaded testing WAV files.
  --model-path MODEL_PATH
                        PODS-AI model path or HuggingFace model ID.
  --model-revision MODEL_REVISION
                        Optional model revision for HuggingFace model IDs.
  --min-confidence MIN_CONFIDENCE
                        Minimum local segment confidence required for proposal generation.
  --category CATEGORY   Process only detections whose actual category matches this value.
```
