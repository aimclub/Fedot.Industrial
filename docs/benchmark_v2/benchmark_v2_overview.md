# Benchmark V2 Overview

## Purpose

`benchmark/v2` is the new benchmark stack for time series forecasting, classification, and regression in this
repository.

It replaces the old benchmark layer with a typed, reproducible, artifact-oriented workflow that supports:

- forecasting on local `M4` and `Monash` subsets,
- time series classification on local `UCR/UEA` datasets,
- time series regression on local TSER datasets,
- publication-ready artifact packs,
- manifest-driven execution,
- registered runs and run-to-run comparison.

## Scope

The current `benchmark/v2` implementation covers:

- `forecasting`
- `ts_classification`
- `ts_regression`

with a shared schema and shared orchestration vocabulary.

## Main Modules

- [core.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/core.py)
  Common dataclasses and task-agnostic result schema.
- [forecasting.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/forecasting.py)
  Forecasting adapters, metrics, model adapters, and suite runner.
- [classification.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/classification.py)
  TSC adapters, metrics, baselines, and artifact rendering.
- [regression.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/regression.py)
  TSER adapters, metrics, baselines, and artifact rendering.
- [analytics.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/analytics.py)
  Series-level comparison and publication pack generation.
- [presets.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/presets.py)
  Ready-to-run local benchmark presets.
- [manifests.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/manifests.py)
  JSON/YAML manifest loading, validation, and execution.
- [registry.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/registry.py)
  Registered runs, run bundles, and registry persistence.
- [run_compare.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/run_compare.py)
  Run-to-run comparison on top of the registry layer.
- [cli.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/cli.py)
  Package-level CLI used by `python -m benchmark.v2`.

## Execution Modes

`benchmark/v2` supports four practical ways to run benchmarks.

### 1. Direct Suite Config

Use `BenchmarkSuiteConfig` when you need full programmatic control.

### 2. Presets

Use presets for standard local runs:

- `m4`
- `monash`
- `ucr`
- `tser`

This is the fastest way to launch a benchmark without building a large config object by hand.

### 3. Manifests

Use JSON/YAML manifests when you need reproducible experiment descriptions that can be versioned in Git.

### 4. Registered Runs

Use the registry-backed functions when you want benchmark runs to persist:

- run summary,
- resolved config,
- input payload,
- artifact manifest,
- registry entry,
- registry index.

This is the preferred mode for experiment studies and paper-oriented workflows.

## Artifact Layout

For persisted runs, output is organized under:

```text
<output_dir>/
  <run_id>/
    aggregate/
    series/
    run_summary.json
    resolved_config.json
    artifact_manifest.json
    input_payload.json
    resolved_manifest.json
  _registry/
    <run_id>.json
    run_registry.jsonl
    run_registry.csv
    run_registry.md
```

Not every file is present for every execution mode.

For example:

- `input_payload.json` is present for manifest and preset executions,
- `resolved_manifest.json` is present for manifest executions,
- `resolved_config.json` is always written for registered runs.

## Current Baseline Coverage

### Forecasting

- `OKHS`
- `NaiveLastValue`
- `NaiveMean`
- `NaiveDrift`
- `MovingAverage`
- `LinearTrend`
- `ClassicalDMD`
- optional external scaffolds for `AutoGluon`, `N-BEATS`, `TFT`

### Classification

- `MajorityClass`
- `NearestCentroid`

### Regression

- `MeanRegressor`
- `LinearRegressor`

## Current Dataset Coverage

### Forecasting

- local `M4` CSV subsets
  from [examples/data/m4/datasets](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/examples/data/m4/datasets)
- local `Monash` CSV files
  from [examples/data/benchmark/forecasting/monash_benchmark](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/examples/data/benchmark/forecasting/monash_benchmark)

### Classification and Regression

- local `*.tsv`, `*.csv`, and `*.ts` datasets
  from [fedot_ind/data](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/data)

## Recommended Usage

Use:

- presets for quick local validation,
- manifests for reproducible benchmark definitions,
- registered runs for experiment studies,
- run comparison for final analysis across multiple benchmark jobs.

## Related Documents

- [Quickstart](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/benchmark_v2/quickstart.md)
- [Migration Guide](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/benchmark_v2/migration_guide.md)
- [Release Notes](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/benchmark_v2/release_notes.md)

