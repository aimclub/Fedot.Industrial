# Benchmark Result Analysis

Status: current thin notebooks and local-data manifests.

Reusable aggregation, ranking, comparison, and visualization logic lives in
`benchmark.industrial`. Notebooks in this package are presentation entrypoints:
they import `current_api.py` and render publication packs into the ignored local
`examples/artifacts/cloud_bundle/benchmark_showcase/analysis_of_results/<notebook_stem>/`
handoff path for external publication.

## Entry Points

- `analysis_uni_clf.ipynb`: univariate TSC comparison.
- `analysis_multi_clf.ipynb`: multivariate TSC comparison.
- `analysis_regr.ipynb`: TSER comparison.
- `m4_analysis.ipynb`: forecasting/M4 comparison.
- `pdl_uni_benchmark.ipynb`: PDL-focused TSC slice.
- `pipeline_population.ipynb`: evolutionary composition dynamics from local
  history data under `examples/utils/data/benchmark_history`.

## Local Inputs

`analysis_defaults.json` describes source-backed result tables, incremental run
artifacts, expected coverage, and local/DVC inputs. Full benchmark result
directories and historical data are local or DVC inputs, not git-tracked
payloads. If a local source is unavailable, notebooks must show the preflight
status rather than silently replacing it with fake data.
