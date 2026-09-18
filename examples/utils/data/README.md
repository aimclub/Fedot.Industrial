# Examples Data

Small tracked fixtures are organized by task type. Larger local datasets and
generated outputs may exist in this directory during experiments, but they are
not part of the repository contract and should remain untracked.

## Layout

| Path | Task type | Use case |
| --- | --- | --- |
| `ts_classification/` | time series classification | UCR/UEA-style train/test fixtures. |
| `ts_regression/` | time series regression | TSER train/test fixtures. |
| `forecasting/` | forecasting | Forecasting CSV samples and local real-world series. |
| `anomaly_detection/` | anomaly detection | SKAB and local anomaly-detection fixtures. |

When adding data for a new example, place it in the task folder first and then
document the expected reader and file format in that folder's README.

Legacy local directories such as `benchmark/` or `real_world/` may exist in a
developer workspace as untracked inputs from previous experiments. New examples
should not depend on those paths; move or mirror local data into the appropriate
task folder when preparing a reproducible scenario.
