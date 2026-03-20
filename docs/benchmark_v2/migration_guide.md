# Benchmark V2 Migration Guide

## Goal

This guide explains how to migrate from the legacy benchmark entrypoints:

- `benchmark/benchmark_TSF.py`
- `benchmark/benchmark_TSC.py`
- `benchmark/benchmark_TSER.py`

to the new `benchmark/v2` stack.

## Migration Principle

The old benchmark modules should now be treated as compatibility shells.

For new work, prefer:

- typed suite configs,
- presets,
- manifests,
- registered runs,
- registry-based run comparison.

## Old to New Mapping

### Forecasting

Old:

```python
from benchmark.benchmark_TSF import BenchmarkTSF
```

New:

```python
from benchmark.v2 import run_forecasting_benchmark_suite
from benchmark.v2 import run_local_benchmark_preset
from benchmark.v2 import run_manifest_path
```

### Classification

Old:

```python
from benchmark.benchmark_TSC import BenchmarkTSC
```

New:

```python
from benchmark.v2 import run_tsc_benchmark_suite
from benchmark.v2 import run_local_benchmark_preset
```

### Regression

Old:

```python
from benchmark.benchmark_TSER import BenchmarkTSER
```

New:

```python
from benchmark.v2 import run_tser_benchmark_suite
from benchmark.v2 import run_local_benchmark_preset
```

## Recommended Migration Path

### 1. Stop Adding New Logic to Legacy Benchmarks

Do not extend:

- [benchmark_TSF.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/benchmark_TSF.py)
- [benchmark_TSC.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/benchmark_TSC.py)
- [benchmark_TSER.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/benchmark_TSER.py)

These files now exist mainly for backward compatibility.

### 2. Move New Runs to `benchmark.v2`

Choose the narrowest entrypoint that matches your workflow:

- direct suite config for full programmatic control,
- preset for quick local runs,
- manifest for reproducible experiment definitions,
- registered runs for study-oriented execution.

### 3. Move Comparison Logic to Registry-Based Analysis

If you previously compared CSV files by hand or used ad hoc notebooks, prefer:

```python
from benchmark.v2 import compare_registered_runs
```

This gives one comparison layer for all run types.

## Common Migration Targets

### Legacy Forecasting Config to Suite Config

Legacy shape:

```python
experiment_setup = {
    'dataset_specs': [...],
    'model_specs': [...],
    'output_dir': '...',
}
```

New shape:

```python
from benchmark.v2 import ArtifactSpec, BenchmarkSuiteConfig, DatasetSpec, ModelSpec, RunSpec, TaskType

config = BenchmarkSuiteConfig(
    task_type=TaskType.FORECASTING,
    datasets=(DatasetSpec(...),),
    models=(ModelSpec(...),),
    artifact_spec=ArtifactSpec(output_dir='benchmark/results/v2_demo'),
    run_spec=RunSpec(run_name='forecasting_demo', primary_metric='mae'),
)
```

### Legacy Quick Run to Preset

Old pattern:

```python
benchmark = BenchmarkTSF(experiment_setup=..., custom_datasets=[])
result = benchmark.run()
```

New pattern:

```python
from benchmark.v2 import run_local_benchmark_preset

result = run_local_benchmark_preset(
    'm4',
    subset='daily',
    sample_size=2,
    persist_on_run=True,
)
```

### Legacy Batch Execution to Manifest

If a run definition should be versioned, prefer a manifest file and run it with:

```python
from benchmark.v2 import run_manifest_path

result = run_manifest_path('examples/benchmark_v2/manifests/m4_daily_preset.yaml')
```

## Compatibility Status

Current legacy modules still delegate to `benchmark/v2` when `use_benchmark_v2=True` is passed in their setup.

This means you can migrate incrementally:

1. keep old entrypoint,
2. switch execution mode to `use_benchmark_v2=True`,
3. replace call sites with direct `benchmark.v2` imports,
4. remove legacy wrapper usage.

## What Is Better in `benchmark/v2`

Compared to the legacy layer, `benchmark/v2` gives you:

- typed configs and results,
- real local dataset adapters,
- publication-ready artifact packs,
- manifest-based reproducibility,
- registry-backed experiment tracking,
- run-to-run comparison,
- one vocabulary across forecasting, classification, and regression.

## What Still Stays Out of Scope

`benchmark/v2` is not yet a full dashboard product.

It is intentionally optimized for:

- local reproducible runs,
- paper-oriented artifact generation,
- structured study comparisons,
- benchmark refactoring and migration.

## Migration Checklist

- Move imports from legacy benchmark modules to `benchmark.v2`.
- Replace raw dict orchestration with `BenchmarkSuiteConfig`, presets, or manifests.
- Persist important runs through registered execution.
- Compare experiments through the registry instead of manual CSV merging.
- Keep legacy wrappers only where backward compatibility is still required.

## Related Documents

- [Overview](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/benchmark_v2/benchmark_v2_overview.md)
- [Quickstart](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/benchmark_v2/quickstart.md)
