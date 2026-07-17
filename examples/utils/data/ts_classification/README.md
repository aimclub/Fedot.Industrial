# Time Series Classification Data

Contains UCR/UEA-style train/test fixtures used by classification examples and
loader tests.

## Fixtures

- `ItalyPowerDemand_fake/`: tiny fake classification dataset with `.arff`,
  `.ts`, `.tsv`, and `.txt` train/test files for parser smoke tests.

## Local Inputs

- `eeg/`: optional local EEG arrays (`sig_data.npy`, `sig_target.npy`) for the
  harmful brain activity classification notebook.
- `hand_recognition/`: optional local CSV files for hand-recognition
  classification scenarios.

Local inputs are not committed. Keep scripts and notebooks pointed at this task
folder through package-local defaults instead of hard-coding legacy
legacy pre-`utils` real-world paths.
