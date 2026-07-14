# Benchmark Industrial Overview

This page is kept in `docs/benchmark_v2` as a migration landing page for older links. The active runtime package is `benchmark.industrial`.

## Purpose

`benchmark.industrial` is the benchmark stack for time series forecasting, classification, and regression in this repository. It provides a typed, reproducible, artifact-oriented workflow with:

- forecasting on local `M4` and `Monash` subsets;
- time series classification on local `UCR/UEA` datasets;
- time series regression on local TSER datasets;
- publication-ready artifact packs;
- manifest-driven execution;
- registered runs and run-to-run comparison;
- canonical result showcase generation.

## Scope

The current implementation covers:

- `forecasting`;
- `ts_classification`;
- `ts_regression`.

## Main Modules

- `benchmark/industrial/core.py`: shared dataclasses and task-agnostic result schema.
- `benchmark/industrial/forecasting.py`: forecasting adapters, metrics, model adapters, and suite runner.
- `benchmark/industrial/classification.py`: TSC adapters, metrics, baselines, and artifact rendering.
- `benchmark/industrial/regression.py`: TSER adapters, metrics, baselines, and artifact rendering.
- `benchmark/industrial/evaluation/`: aggregation, result analysis, diagnostics, and evolution analysis.
- `benchmark/industrial/visualization/`: reusable report and forecast-comparison plots.
- `benchmark/industrial/experiments/`: manifests, presets, registry, persistence, and run comparison.
- `benchmark/industrial/cli.py`: package CLI used by `python -m benchmark.industrial`.

## Execution Modes

1. Direct `BenchmarkSuiteConfig` for full programmatic control.
2. Presets for quick local runs: `m4`, `monash`, `ucr`, `tser`.
3. JSON manifests for reproducible experiment descriptions.
4. Registered runs for persisted studies and paper-oriented workflows.

## Artifact Layout

Registered suite runs should write:

```text
<output_dir>/<run_id>/
  records/
  aggregate/
  resolved_config.json
  resolved_manifest.json
  run_summary.json
  artifact_manifest.json
```

The public benchmark window is generated from:

```bash
python -m benchmark.results.showcase
```

## Related Documents

- [Benchmark infrastructure guide](../dev_guide/benchmark_infrastructure.md)
- [Kernel Learning benchmark runbook](../dev_guide/kernel_learning_benchmark_runbook.md)
- [Quickstart](quickstart.md)
- [Migration Guide](migration_guide.md)
