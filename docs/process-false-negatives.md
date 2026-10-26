# process_false_negatives.py

Re-checks confirmed machine detections by downloading the 60-second WAV and
running PODS-AI inference. By default, it runs OrcaHello segment inference and
appends 3-second segments where OrcaHello predicts resident but PODS-AI does not
to `output/csv/new_manual_training_samples.csv` with corrected class `resident`. Use
`--set testing` to instead add a 60-second testing sample when PODS-AI does not
predict resident; these rows are appended to
`output/csv/new_manual_testing_samples.csv`. Supports `--category CATEGORY` to
process only detections whose PODS-AI predicted category matches the provided
value.

```
usage: process_false_negatives.py [-h] [--set SET] [--feed FEED] [--start YYYY_MM_DD_HH_MM_SS_PST]
                                  [--end YYYY_MM_DD_HH_MM_SS_PST|now] [--manual-samples-csv MANUAL_SAMPLES_CSV]
                                  [--output-dir OUTPUT_DIR] [--model-path MODEL_PATH]
                                  [--orcahello-model-path ORCAHELLO_MODEL_PATH] [--detections-csv DETECTIONS_CSV]
                                  [--category CATEGORY]

Process confirmed machine detections, find 60-second false negatives where PODS-AI misses resident calls, and append
corrected resident sub-segments or testing samples to the corresponding manual-samples CSV.

options:
  -h, --help            show this help message and exit
  --set SET             training or testing (default: training).
  --feed FEED           Process only this feed (by node_name, e.g., rpi_sunset_bay).
  --start YYYY_MM_DD_HH_MM_SS_PST
                        Include only detections with timestamp >= this value.
  --end YYYY_MM_DD_HH_MM_SS_PST|now
                        Include only detections with timestamp <= this value. Use 'now' to remove the upper bound.
  --manual-samples-csv MANUAL_SAMPLES_CSV
                        Path to the manual samples CSV for the selected set.
  --output-dir OUTPUT_DIR
                        Directory where add_samples.py should write segment WAV files.
  --model-path MODEL_PATH
                        PODS-AI model path or HuggingFace model ID.
  --orcahello-model-path ORCAHELLO_MODEL_PATH
                        OrcaHello model path or HuggingFace model ID.
  --detections-csv DETECTIONS_CSV
                        Path to detections.csv for add_samples.py metadata lookups.
  --category CATEGORY   Process only detections whose PODS-AI predicted category matches this value.
```

For example, to find false negatives for the testing set:

```bash
python src/process_false_negatives.py --set testing
```
