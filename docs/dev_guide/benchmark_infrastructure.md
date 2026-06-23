# Industrial Benchmark Infrastructure

This document is the development contract for benchmark code, experiment
entrypoints, result artifacts, and public comparison tables.

Use it when adding a new benchmark direction, changing Kernel Learning
experiments, or updating `benchmark/results`.

## Canonical Runtime Package

The canonical benchmark runtime package is:

```python
benchmark.industrial
```

New benchmark code must import from `benchmark.industrial` or from a thematic
package below it. Do not add compatibility imports for removed versioned
packages. Historical result folders such as `benchmark/results/v2_kernel_learning`
can be used as data sources, but they are not runtime packages.

The current package layout is:

```text
benchmark/industrial/
  datasets/        dataset discovery and local split loading
  evaluation/      metrics, result analysis, aggregation helpers, markdown
  experiments/     manifests, persistence, registries, presets, progress
  legacy/          old wrappers that delegate to the canonical runtime
  models/          model specs and adapter factories
  visualization/   reusable plotting helpers
  api.py           public suite helpers
  classification.py
  regression.py
  forecasting.py
  core.py          typed benchmark contracts
  cli.py
```

Implementation should live next to the thematic public entrypoint. A package
`__init__.py` may expose a short public index, but it should not become the only
place that gathers implementations from unrelated modules.

## Typed Config Contract

All suite-style benchmark runs should converge to:

```python
typed_config.build_suite_config() -> run_registered_suite(...)
```

The shared runtime types live in `benchmark.industrial.core`:

- `BenchmarkSuiteConfig`
- `DatasetSpec`
- `ModelSpec`
- `RunSpec`
- `ArtifactSpec`
- `TaskType`

Shell scripts may parse CLI arguments or environment variables, but they should
normalize those inputs into typed config objects before the run starts. The
runner should not spread defaults across several scripts.

Data-like defaults belong in JSON files next to the thematic package when that
keeps implementation code smaller and easier to review. Use this for model
lists, metrics, dataset ids, run metadata, and output templates. Implementation
modules should load, validate, normalize, and convert JSON payloads into typed
records.

Current defaults and manifest entrypoints:

```text
benchmark/experiments/kernel_learning/defaults.json
benchmark/industrial/experiments/preset_defaults.json
benchmark/industrial/examples/*_manifest.json
benchmark/results/showcase/showcase_manifest.json
```

Manifest-driven runs should use:

```python
from benchmark.industrial.experiments.manifests import build_suite_config_from_manifest
```

and registry-backed runs should use:

```python
from benchmark.industrial.experiments.registry import run_registered_suite
```

## Experiment Entrypoint Layout

Experiment runners are grouped by task or use case. The current Kernel Learning
layout is the reference structure:

```text
benchmark/experiments/kernel_learning/
  classification/
    run_ucr.py
    run_ucr_two_stage.py
  regression/
    run_tser.py
  forecasting/
    run_m4.py
  analysis/
    analyze_stage1.py
  configs.py
  controls.py
  defaults.json
```

Rules for new runners:

- Put the executable script in the task package.
- Keep it thin: parse user input, build a typed config, call shared runtime.
- Reuse `benchmark.industrial.experiments` for persistence, progress, registry
  entries, and resolved configs.
- Reuse `benchmark.industrial.evaluation` and
  `benchmark.industrial.visualization` for aggregation and plots.
- Do not write result files from ad-hoc helpers when a shared writer already
  exists.

## Result Storage Contract

`benchmark/results` has three different responsibilities. Keep them separate.

```text
benchmark/results/
  kernel_learning/          current resumable run artifacts
  v2_kernel_learning/       historical Kernel Learning reference artifacts
  showcase/                 canonical comparison tables and report packs
  server_results/           historical raw artifacts, archive candidate
  ts_*                      historical raw artifacts, archive candidates
```

The canonical result entrypoint is:

```bash
python -m benchmark.results.showcase
```

The showcase reads `benchmark/results/showcase/showcase_manifest.json` and
builds:

```text
benchmark/results/showcase/
  README.md
  showcase_manifest.resolved.json
  tables/
    source_inventory.csv
    benchmark_overview.csv
    current_best_per_dataset.csv
    archive_candidates.csv
  <benchmark_group>/
    summary.md
    tables/
    plots/
```

Use these tables first:

- `tables/benchmark_overview.csv` for coverage by benchmark direction.
- `tables/source_inventory.csv` for source provenance and archive status.
- `tables/current_best_per_dataset.csv` for the best current Industrial result
  per dataset.

Raw run folders, checkpoints, logs, and historical fitted pipelines must not be
mixed into public comparison tables directly. If they are needed for a report,
add them to the showcase manifest as sources or archive candidates.

## Expected Run Artifact Layout

Registered suite runs should persist enough information to rebuild aggregate
tables without rerunning training:

```text
benchmark/results/<family>/<suite>/<run_id>/
  records/
    runs.jsonl
    metrics.jsonl
    predictions.jsonl
    kernel_diagnostics.jsonl       optional Kernel Learning artifact
    kernel_selection.jsonl          optional Kernel Learning artifact
  aggregate/
    runs.csv
    metrics.csv
    predictions.csv
    leaderboard.csv
    run_metadata.json
    summary.md
  resolved_config.json
  resolved_manifest.json
  run_summary.json
  artifact_manifest.json
```

Task-specific additions are allowed, but the shared files above are the stable
contract for aggregation and showcase generation.

## Aggregation And Visualization Rules

The canonical aggregation layer is:

```python
benchmark.industrial.evaluation.aggregation
```

Use it when a runner or notebook needs to rebuild reference-style
`aggregate/*` artifacts from saved `records/*` files. The public entrypoints
are:

- `resolve_task_aggregation_rule` for task-specific metric direction,
  grouping, count column, and prediction index rules;
- `load_benchmark_artifact_frames` for loading `runs`, `metrics`,
  `predictions`, `errors`, and optional Kernel Learning diagnostics;
- `build_benchmark_aggregate_tables` for deterministic in-memory aggregate
  tables;
- `render_benchmark_aggregate_artifacts` for writing `leaderboard.csv`,
  `metrics.csv`, `predictions.csv`, `runs.csv`, `run_metadata.json`,
  `summary.md`, and `artifact_manifest.json`.

Use `benchmark.industrial.evaluation.result_analysis` for result ingestion and
comparison logic:

- `load_result_sources`
- `normalize_result_table`
- `build_best_per_dataset_frame`
- `build_mean_rank_frame`
- `build_topk_summary_frame`
- `build_source_delta_frame`
- `build_model_diagnostics_frame`

Use `benchmark.industrial.visualization.benchmark_results` for report packs:

```python
from benchmark.industrial.visualization.benchmark_results import (
    render_benchmark_result_analysis_pack,
)
```

Metric direction must be explicit for each benchmark direction:

- classification: usually `accuracy`, `higher`;
- regression: usually `rmse`, `lower`;
- forecasting: usually `mae`, `lower`, or the task-specific public metric.

Failed and skipped runs should stay in run records with a status and reason.
Canonical comparison tables should use successful metric rows only unless the
report explicitly analyses failures.

## Kernel Learning Reference Pipeline

Kernel Learning is the first reference benchmark pipeline for this
infrastructure.

Typed config builders live in:

```python
benchmark.experiments.kernel_learning.configs
```

Task entrypoints:

```text
benchmark/experiments/kernel_learning/classification/run_ucr.py
benchmark/experiments/kernel_learning/classification/run_ucr_two_stage.py
benchmark/experiments/kernel_learning/regression/run_tser.py
benchmark/experiments/kernel_learning/forecasting/run_m4.py
```

Default model specs, generator sets, metrics, and output templates live in:

```text
benchmark/experiments/kernel_learning/defaults.json
```

Current full-run artifacts are expected under:

```text
benchmark/results/kernel_learning/
```

Historical v2 Kernel Learning artifacts remain valid reference inputs under:

```text
benchmark/results/v2_kernel_learning/
```

Do not import or recreate the old v2 runtime. Treat it as result data only.

## Adding A New Benchmark Direction

Use this checklist for every new benchmark direction.

1. Create or reuse a thematic package under `benchmark/industrial` for runtime
   code.
2. Add typed config builders or manifest support that returns
   `BenchmarkSuiteConfig`.
3. Put default model lists, metrics, dataset ids, and output templates in a
   package-local JSON file when they are table-like data.
4. Add a thin runner under `benchmark/experiments/<family>/<task>/`.
5. Make the runner call `run_registered_suite` or another shared
   `benchmark.industrial` suite function.
6. Persist `records`, `aggregate`, resolved config, run summary, and artifact
   manifest.
7. Reuse shared aggregation and visualization helpers for final tables and
   plots.
8. Add the result source to `benchmark/results/showcase/showcase_manifest.json`
   if it should appear in public comparisons.
9. Add tests for config normalization, manifest validation, result aggregation,
   and at least one smoke run or dry-run path.
10. Update this document only when the infrastructure contract changes.

## What Not To Do

- Do not add new ad-hoc `run_*.py` files at the repository root or directly
  under `benchmark` when they belong to a task package.
- Do not store raw generated outputs inside `benchmark.industrial`.
- Do not hard-code long model catalogs or dataset lists in implementation files
  when a JSON defaults file is clearer.
- Do not copy large historical result folders into showcase directories.
- Do not treat `benchmark/results/v2_kernel_learning` as a runtime package.
- Do not add a large `__init__.py` that hides implementations scattered across
  unrelated files.

## Verification

Recommended local checks after benchmark infrastructure changes:

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& "D:\data_old\WORK\Repo\Industiral\IndustrialTS\venv_3.9_new\Scripts\python.exe" -m pytest -q `
  tests/unit/benchmark/test_results_showcase.py `
  tests/unit/examples/test_real_world_result_analysis.py
```

For Kernel Learning-specific changes, also use:

```powershell
& "D:\data_old\WORK\Repo\Industiral\IndustrialTS\venv_3.9_new\Scripts\python.exe" -m pytest -q `
  tests/unit/core/kernel_learning `
  tests/unit/models/test_kernel_learning_experiment_scripts.py
```

Regenerate the result showcase after updating result sources:

```powershell
& "D:\data_old\WORK\Repo\Industiral\IndustrialTS\venv_3.9_new\Scripts\python.exe" -m benchmark.results.showcase
```
