# process_false_positives.py

Re-checks rejected machine-predicted (whether from PODS-AI or another model)
resident detections by downloading the 60-second WAV, re-running PODS-AI, and
appending candidate rows to the selected manual-samples CSV (training defaults to
`output/csv/new_manual_training_samples.csv`; use `--set testing` for
`output/csv/new_manual_testing_samples.csv`).

The corrected class is inferred from the human-authored portion of the moderation
comments (auto-generated "AI: …" lines are ignored).  Explicit negations in the
comments are understood: "No humpback" suppresses the humpback match, and
"No humpback nor vessel" resolves the corrected class to `water`.
Supports `--category CATEGORY` to process only detections whose inferred
actual category matches the provided value.

```
usage: process_false_positives.py [-h] [--set SET] [--feed FEED] [--start YYYY_MM_DD_HH_MM_SS_PST]
                                  [--end YYYY_MM_DD_HH_MM_SS_PST|now] [--manual-samples-csv MANUAL_SAMPLES_CSV]
                                  [--output-dir OUTPUT_DIR] [--model-path MODEL_PATH]
                                  [--detections-csv DETECTIONS_CSV] [--category CATEGORY]

Process rejected OrcaHello resident detections looking for new training or testing samples. For training samples, re-
run PODS-AI on the 60-second WAV, and append mismatched whale-class sub-segments to new_manual_training_samples.csv
with a corrected class. For testing, append one corrected 60-second sample row to new_manual_testing_samples.csv.

options:
  -h, --help            show this help message and exit
  --set SET             training or testing (default: training).
  --feed FEED           Process only this feed (by node_name, e.g., rpi_sunset_bay).
  --start YYYY_MM_DD_HH_MM_SS_PST
                        Include only detections with timestamp >= this value.
  --end YYYY_MM_DD_HH_MM_SS_PST|now
                        Include only detections with timestamp <= this value. Use 'now' to remove the upper bound.
  --manual-samples-csv MANUAL_SAMPLES_CSV
                        Path to new manual samples csv.
  --output-dir OUTPUT_DIR
                        Directory where add_samples.py should write segment WAV files.
  --model-path MODEL_PATH
                        PODS-AI model path or HuggingFace model ID.
  --detections-csv DETECTIONS_CSV
                        Path to detections.csv for add_samples.py metadata lookups.
  --category CATEGORY   Process only detections whose inferred actual category matches this value.
```
