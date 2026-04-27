# Benchmark V2 Release Notes

## Summary

`benchmark/v2` is the new benchmark stack for forecasting, time series classification, and time series regression in
this repository.

This release replaces the old benchmark workflow with a typed, reproducible, artifact-oriented pipeline.

## What Is Included

- Unified benchmark schema for `forecasting`, `ts_classification`, and `ts_regression`
- Forecasting support for local `M4` and `Monash` subsets
- Local dataset adapters for `UCR/UEA`-style classification datasets
- Local dataset adapters for TSER datasets from `fedot_ind/data`
- Publication-ready artifact generation for forecasting, classification, and regression
- Preset-based execution for standard local benchmark runs
- JSON/YAML manifest-driven execution
- Registered runs with run bundle persistence and registry indexing
- Run-to-run comparison on top of the registry layer
- Package-level CLI via `python -m benchmark.v2`
- Backward-compatible delegation from legacy benchmark entrypoints

## Main User-Facing Entry Points

- `run_forecasting_benchmark_suite(...)`
- `run_tsc_benchmark_suite(...)`
- `run_tser_benchmark_suite(...)`
- `run_local_benchmark_preset(...)`
- `run_manifest(...)`
- `run_manifest_path(...)`
- `run_registered_preset(...)`
- `run_registered_manifest(...)`
- `run_registered_manifest_path(...)`
- `compare_registered_runs(...)`

## Key Improvements Over Legacy Benchmarking

- Typed configs and typed result objects instead of loosely-coupled benchmark scripts
- Shared vocabulary across forecasting, classification, and regression
- Reproducible manifests and resolved configs
- Stable artifact layout for paper-oriented workflows
- Registry-backed experiment tracking
- Built-in run comparison instead of manual CSV merging

## Current Limitations

- External deep baselines are still optional/scaffolded and may be reported as `skipped` or `not_available`
- Some legacy benchmark examples still exist as compatibility paths
- The new benchmark layer is optimized for local reproducible runs and artifact generation, not dashboard-first UX

## Validation Status

The current `benchmark/v2` target test suite is green in the repository and covers:

- forecasting adapters and publication pack generation
- classification and regression task suites
- presets
- manifests
- registered runs
- run-to-run comparison

## Related Documents

- [Overview](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/benchmark_v2/benchmark_v2_overview.md)
- [Quickstart](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/benchmark_v2/quickstart.md)
- [Migration Guide](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/benchmark_v2/migration_guide.md)
