# Industrial Examples

This directory is split by reproducibility role. The current entrypoints use
`benchmark.industrial` typed configs and keep optional local data outside the
repository contract.

## Status Audit

| Path | Status | Notes |
| --- | --- | --- |
| `utils/current_api/` | current | Lightweight examples backed by `benchmark.industrial` typed configs. These are the recommended starting point. |
| `utils/data/` | current support data | Small local fixtures used by examples and tests. Keep these deterministic and repository-local. |
| `tools_example/` | tool-ready wrappers | JSON-friendly Industrial scenarios that expose `list_tool_specs()` and `invoke_tool()` for external agents or MCP tool adapters. Local inputs are documented in `tool_defaults.json` and README. |
| `real_world_examples/` | current wrappers plus local scenarios | Domain examples and benchmark-like scripts with current API wrappers. Notebooks with local data are preserved and documented as local scenarios. |
| `real_world_examples/benchmark_example/rkhs_okhs/` | current forecasting scenario | RKHS/OKHS forecasting example integrated as a benchmark-style forecasting scenario. |
| `models_example/` | notebook reference | Model notebooks preserved as scenario references; they require optional runtime/data checks before joining the smoke contract. |

## Recommended Entry Points

- `python -m examples.utils.current_api`
- `python -m benchmark.industrial --manifest examples/utils/current_api/manifests/toy_tser_suite.json`

The supervised current examples avoid external downloads by default and use
in-memory records. Forecasting, Kernel Learning, RKHS/OKHS, Kaggle, EEG, and
LIMAN scenarios expose typed config previews or local data contexts first;
execute full experiments only in an environment with the optional runtime and
documented local datasets installed.

## Local Inputs And Outputs

Untracked datasets and generated artifacts inside `examples/` are treated as
local inputs or outputs. Do not add them to git. Prefer task-based data paths
under `examples/utils/data/{ts_classification,ts_regression,forecasting,anomaly_detection}`
and document any external dataset in the corresponding README.

Tracked notebooks that still depend on optional packages or private/local data
are preserved as scenario references. Their READMEs document the expected input
paths instead of replacing missing data with silent stubs.

## Notebook Execution Notes

Use the `industrialts-venv39` Jupyter kernel, or an equivalent environment with
`fedot_ind`, optional model runtimes, and local scenario data installed, when
rerunning notebooks. The current reproducibility contract is covered by
`examples/utils/current_api` and migrated `.py` entrypoints. Historical notebooks that
still contain legacy pipeline cells are kept as applied references and should be
rewritten into task-local current API notebooks before they are added to smoke
execution.
