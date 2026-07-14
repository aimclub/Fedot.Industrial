# Real World Examples

Status: current wrappers, reusable analytics, and local scenario references.

Benchmark-like Python scripts now delegate to `current_api.py`, which builds
typed `benchmark.industrial` configs or lightweight publication contexts.
Domain notebooks are preserved as applied scenarios because many depend on
local datasets or optional packages.

## Layout

| Path | Purpose |
| --- | --- |
| `benchmark_example/classification/` | TSC benchmark wrappers based on current typed configs. |
| `benchmark_example/regression/` | TSER benchmark wrappers based on current typed configs. |
| `benchmark_example/forecasting/` | Forecasting benchmark and Kaggle-style local contexts. |
| `benchmark_example/detection/` | SKAB anomaly-detection context using the task-based data path. |
| `benchmark_example/analysis_of_results/` | Thin benchmark-analysis notebooks backed by `benchmark.industrial` result analytics. |
| `industrial_examples/` | Domain use cases grouped by business domain and task. |
| `industrial_examples/eeg/classification/` | Migrated EEG classification notebook scenario. |
| `../utils/data/benchmark_history/` | Historical result/data manifests used by benchmark analysis. |

## Relationship To `tools_example`

Keep `tools_example` and `real_world_examples` separate for now. The former is
the MCP-ready action surface for external agents; the latter is the applied
showcase with local data manifests, benchmark comparison artifacts, and domain
notebooks. Shared implementation should move into `benchmark.industrial` or
`examples/utils/current_api`, not be duplicated between the two example
packages.

## Local Inputs And Notebooks

Constants for external Kaggle/EEG inputs live in `real_world_defaults.json`.
`external_data_manifest.json` describes local raw inputs, historical benchmark
results, expected paths, optional DVC locations, and the public Yandex Disk archive link. Keep datasets, checkpoints, and high-churn image outputs untracked.
Report-style benchmark/domain assets live under
`examples/artifacts/cloud_bundle` when they are deterministic and lightweight.

Data delivery contract:

```powershell
python -m pip install dvc
# Optional: configure a private/local DVC remote outside git, then run:
dvc pull
```

Users without DVC can download the public Yandex Disk archive listed in
`external_data_manifest.json`: https://disk.yandex.ru/d/Ch_7K26rukpAWw and unpack them into each source `expected_path`.
Do not commit credentials, OAuth tokens, `.dvc/config.local`, raw datasets, or
historical result dumps.

Some preserved domain notebooks still carry historical pipeline cells. Treat
them as applied references, not as the current smoke contract, until they are
rewritten around the task-local wrappers in this package or the typed
`benchmark.industrial` suite/config APIs.

Forecasting notebooks should render both metric comparison packs and
multi-model forecast plots. Classification and regression notebooks should show
Kernel Learning reference models, UCR two-stage context when relevant, and
single-generator downstream baselines alongside Industrial and SoTA comparisons.
