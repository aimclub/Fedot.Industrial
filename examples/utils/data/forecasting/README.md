# Forecasting Data

Contains small forecasting samples and local real-world time-series files.

## Fixtures

- `nbeats/`: daily train/test CSV sample for forecasting tutorials.
- `ice_forecasting/`: Arctic sea ice area series used by the real-world
  forecasting example.

## Local Inputs

- `benchmark_forecasting/`: optional local M4/Monash files used by historical
  forecasting notebooks.
- `kaggle_inventory/`: optional local Kaggle forecasting files
  (`train.csv`, `test.csv`, `submission.csv`).
- `debet/`: optional local oil-field/debet forecasting config, source CSVs,
  checkpoints, and prediction outputs.

Local inputs are not committed; wrappers expose config or context previews so
the expected paths are visible before running a full experiment.
