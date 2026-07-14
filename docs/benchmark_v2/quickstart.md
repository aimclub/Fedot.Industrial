# Benchmark Industrial Quickstart

This page replaces the old `benchmark.v2` quickstart. The runtime package for new benchmark work is now `benchmark.industrial`.

Historical folders such as `benchmark/results/v2_kernel_learning` may still be used as data sources, but they are not import targets.

## Fastest Path

The simplest way to run the benchmark stack is through presets.

### Python API

```python
from benchmark.industrial import run_local_benchmark_preset

result = run_local_benchmark_preset(
    "m4",
    subset="daily",
    sample_size=2,
    persist_on_run=True,
    output_dir="benchmark/results/industrial_demo/m4_daily",
)
```

### Package CLI

```bash
python -m benchmark.industrial m4 --subset daily --sample-size 2 --output-dir benchmark/results/industrial_demo/m4_daily
```

## Forecasting Presets

Supported preset names:

- `m4`
- `monash`

Example:

```python
from benchmark.industrial import run_local_benchmark_preset

result = run_local_benchmark_preset(
    "monash",
    dataset_name="Bitcoin",
    subset="daily",
    sample_size=2,
    persist_on_run=True,
    output_dir="benchmark/results/industrial_demo/monash_bitcoin",
)
```

## Classification And Regression Presets

```python
from benchmark.industrial import run_local_benchmark_preset

tsc_result = run_local_benchmark_preset(
    "ucr",
    dataset_name="Lightning7",
    persist_on_run=True,
    output_dir="benchmark/results/industrial_demo/ucr_lightning7",
)

tser_result = run_local_benchmark_preset(
    "tser",
    dataset_name="NaturalGasPricesSentiment",
    persist_on_run=True,
    output_dir="benchmark/results/industrial_demo/tser_natural_gas",
)
```

## Registered Runs

If every run should be indexed automatically, use the registered entrypoints.

```python
from benchmark.industrial import run_registered_preset

bundle = run_registered_preset(
    "ucr",
    dataset_name="Lightning7",
    output_dir="benchmark/results/industrial_study/ucr_lightning7",
    persist_on_run=True,
)

print(bundle.registry_entry_path)
print(bundle.summary_path)
```

## Manifest Workflow

Manifest examples now live under `examples/utils/current_api/manifests` and use JSON by default.

```python
from benchmark.industrial import run_manifest_path

result = run_manifest_path("examples/utils/current_api/manifests/m4_daily_preset.json")
```

To inspect the resolved manifest without running training:

```python
from benchmark.industrial import load_manifest, render_resolved_manifest

payload = load_manifest("examples/utils/current_api/manifests/m4_daily_preset.json")
resolved = render_resolved_manifest(payload)
```

CLI usage:

```bash
python -m benchmark.industrial --manifest examples/utils/current_api/manifests/m4_daily_preset.json
python -m benchmark.industrial --manifest examples/utils/current_api/manifests/m4_daily_preset.json --print-resolved-manifest
```

## Run Comparison

To compare multiple registered runs in the same output root:

```python
from benchmark.industrial import compare_registered_runs

comparison = compare_registered_runs(
    "benchmark/results/industrial_study/ucr_lightning7",
    output_dir="benchmark/results/industrial_study/ucr_lightning7/_comparison",
)
```

This produces registry overview tables, best-model-per-run tables, model-vs-run matrices, markdown summaries, and plots.

## Standard Entry Points

Programmatic entrypoints:

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

CLI:

```bash
python -m benchmark.industrial <preset> [options]
python -m benchmark.industrial --manifest <path> [options]
```

## Useful Files

- [Benchmark infrastructure guide](../dev_guide/benchmark_infrastructure.md)
- [Kernel Learning benchmark runbook](../dev_guide/kernel_learning_benchmark_runbook.md)
- [Current API manifests](../../examples/utils/current_api/manifests)
