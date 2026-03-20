# Benchmark V2 Quickstart

## Fastest Path

The simplest way to run the new benchmark stack is through presets.

### Python API

```python
from benchmark.v2 import run_local_benchmark_preset

result = run_local_benchmark_preset(
    'm4',
    subset='daily',
    sample_size=2,
    persist_on_run=True,
    output_dir='benchmark/results/v2_demo/m4_daily',
)
```

### Package CLI

```bash
python -m benchmark.v2 m4 --subset daily --sample-size 2 --output-dir benchmark/results/v2_demo/m4_daily
```

## Forecasting Presets

Supported preset names:

- `m4`
- `monash`

Example:

```python
from benchmark.v2 import run_local_benchmark_preset

result = run_local_benchmark_preset(
    'monash',
    dataset_name='Bitcoin',
    subset='daily',
    sample_size=2,
    persist_on_run=True,
    output_dir='benchmark/results/v2_demo/monash_bitcoin',
)
```

## Classification and Regression Presets

```python
from benchmark.v2 import run_local_benchmark_preset

tsc_result = run_local_benchmark_preset(
    'ucr',
    dataset_name='Lightning7',
    persist_on_run=True,
    output_dir='benchmark/results/v2_demo/ucr_lightning7',
)

tser_result = run_local_benchmark_preset(
    'tser',
    dataset_name='NaturalGasPricesSentiment',
    persist_on_run=True,
    output_dir='benchmark/results/v2_demo/tser_natural_gas',
)
```

## Registered Runs

If you want every run to be indexed automatically, use the registered entrypoints.

```python
from benchmark.v2 import run_registered_preset

bundle = run_registered_preset(
    'ucr',
    dataset_name='Lightning7',
    output_dir='benchmark/results/v2_study/ucr_lightning7',
    persist_on_run=True,
)

print(bundle.registry_entry_path)
print(bundle.summary_path)
```

### Registered Manifest Run

```python
from benchmark.v2 import run_registered_manifest_path

bundle = run_registered_manifest_path(
    'examples/benchmark_v2/manifests/m4_daily_preset.yaml'
)
```

## Manifest Workflow

### Run From Manifest

```python
from benchmark.v2 import run_manifest_path

result = run_manifest_path('examples/benchmark_v2/manifests/m4_daily_preset.yaml')
```

### Inspect Resolved Manifest

```python
from benchmark.v2 import load_manifest, render_resolved_manifest

payload = load_manifest('examples/benchmark_v2/manifests/m4_daily_preset.yaml')
resolved = render_resolved_manifest(payload)
```

### CLI Manifest Run

```bash
python -m benchmark.v2 --manifest examples/benchmark_v2/manifests/m4_daily_preset.yaml
```

### CLI Resolved Manifest Print

```bash
python -m benchmark.v2 --manifest examples/benchmark_v2/manifests/m4_daily_preset.yaml --print-resolved-manifest
```

## Run Comparison

To compare multiple registered runs in the same output root:

```python
from benchmark.v2 import compare_registered_runs

comparison = compare_registered_runs(
    'benchmark/results/v2_study/ucr_lightning7',
    output_dir='benchmark/results/v2_study/ucr_lightning7/_comparison',
)
```

This produces:

- registry overview tables,
- best-model-per-run table,
- model-vs-run metric matrix,
- comparison summary markdown,
- comparison plots.

## Standard Entry Points

### Programmatic

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

### CLI

```bash
python -m benchmark.v2 <preset> [options]
python -m benchmark.v2 --manifest <path> [options]
```

## Useful Files

- [Preset manifest example](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/examples/benchmark_v2/manifests/m4_daily_preset.yaml)
- [Overview](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/benchmark_v2/benchmark_v2_overview.md)
- [Migration Guide](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/benchmark_v2/migration_guide.md)
