# Debet Forecasting

Applied forecasting scenario for local oil-field/debet data.

The current entrypoint is `main.py`. It exposes a context preview and does not
load data, train neural models, save checkpoints, or render plots by default.

## Local Inputs

Expected local data and config live under `examples/utils/data/forecasting/debet`:

- `config.json`: scenario configuration.
- source CSV files referenced by the config.
- optional model checkpoints for inference or finetuning.

These files are local inputs and are not committed.

## Outputs

Generated checkpoints, plots, and prediction folders are local outputs. Keep
them untracked. After a real benchmark run, prefer shared publication helpers
from `benchmark.industrial.evaluation` for aggregate tables and visual reports.
