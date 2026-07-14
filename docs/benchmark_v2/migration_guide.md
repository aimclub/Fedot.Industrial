# Migration Guide: Legacy Benchmarks To `benchmark.industrial`

This document replaces the old Benchmark V2 migration guide. The canonical runtime package is now `benchmark.industrial`.

Use historical `benchmark/results/v2_kernel_learning` folders only as result data. Do not import the removed versioned runtime package in new code.

## Goal

Migrate from legacy benchmark entrypoints:

- `benchmark/benchmark_TSF.py`
- `benchmark/benchmark_TSC.py`
- `benchmark/benchmark_TSER.py`

into the current typed benchmark stack:

- typed suite configs;
- JSON defaults and manifests;
- registered runs;
- shared aggregation and visualization;
- showcase-driven public comparisons.

## Old To New Mapping

### Forecasting

Old:

```python
from benchmark.benchmark_TSF import BenchmarkTSF
```

New:

```python
from benchmark.industrial import run_forecasting_benchmark_suite
from benchmark.industrial import run_local_benchmark_preset
from benchmark.industrial import run_manifest_path
```

### Classification

Old:

```python
from benchmark.benchmark_TSC import BenchmarkTSC
```

New:

```python
from benchmark.industrial import run_tsc_benchmark_suite
from benchmark.industrial import run_local_benchmark_preset
```

### Regression

Old:

```python
from benchmark.benchmark_TSER import BenchmarkTSER
```

New:

```python
from benchmark.industrial import run_tser_benchmark_suite
from benchmark.industrial import run_local_benchmark_preset
```

## Recommended Migration Path

1. Stop adding new logic to root-level legacy benchmark files.
2. Put new runtime code under the relevant `benchmark/industrial/<domain>` package.
3. Normalize inputs into `BenchmarkSuiteConfig`, `DatasetSpec`, `ModelSpec`, `RunSpec`, and `ArtifactSpec`.
4. Keep model lists, dataset lists, metrics, and output templates in package-local JSON defaults when they are catalog-like data.
5. Use `run_registered_suite`, `run_registered_preset`, or `run_registered_manifest_path` for persisted studies.
6. Use `benchmark.industrial.evaluation` and `benchmark.industrial.visualization` for aggregation and plots.
7. Add public result sources to `benchmark/results/showcase/showcase_manifest.json`.

## Suite Config Example

```python
from benchmark.industrial import ArtifactSpec, BenchmarkSuiteConfig, DatasetSpec, ModelSpec, RunSpec, TaskType

config = BenchmarkSuiteConfig(
    task_type=TaskType.FORECASTING,
    datasets=(DatasetSpec(dataset_name="M4_Daily", source="local_m4", params={"subset": "daily", "sample_size": 2}),),
    models=(ModelSpec(adapter_name="naive_last_value", display_name="NaiveLastValue"),),
    artifact_spec=ArtifactSpec(output_dir="benchmark/results/industrial_demo"),
    run_spec=RunSpec(run_name="forecasting_demo", primary_metric="mae"),
)
```

## Manifest Example

```python
from benchmark.industrial import run_manifest_path

result = run_manifest_path("examples/utils/current_api/manifests/m4_daily_preset.json")
```

## Compatibility Status

`benchmark.industrial.legacy` contains compatibility wrappers for old flows that still need them. These wrappers may delegate to the current runtime internally, but new code should not copy their `ApiTemplate` usage or old root-script style.

## Related Documents

- [Benchmark infrastructure guide](../dev_guide/benchmark_infrastructure.md)
- [Kernel Learning benchmark runbook](../dev_guide/kernel_learning_benchmark_runbook.md)
- [Quickstart](quickstart.md)
