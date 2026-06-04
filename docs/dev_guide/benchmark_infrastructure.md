# Industrial Benchmark Infrastructure

## Canonical Module

The canonical benchmark implementation lives in:

```python
benchmark.industrial
```

`benchmark.industrial` is the only valid benchmark runtime package. New code,
tests, scripts, manifests, and notebooks should import from it directly. The
old versioned package has been removed instead of kept as a compatibility
facade.

## Package Layout

The package keeps the stable runtime modules and exposes faceted entry points:

- `benchmark.industrial.datasets` for dataset discovery and local split loading.
- `benchmark.industrial.evaluation` for metrics and publication packs.
- `benchmark.industrial.experiments` for suite configs, manifests, presets, and registries.
- `benchmark.industrial.models` for model specs and adapter factories.
- `benchmark.industrial.visualization` for visualization helpers.
- `benchmark.industrial.legacy` for old benchmark wrappers that still delegate
  to the canonical Industrial runtime.

Only public runtime shells stay flat at the package root:

- `benchmark.industrial.core`
- `benchmark.industrial.api`
- `benchmark.industrial.classification`
- `benchmark.industrial.regression`
- `benchmark.industrial.forecasting`
- `benchmark.industrial.cli`
- `benchmark.industrial.errors`

Helper implementations live in thematic packages instead of being collected by
large `__init__.py` files:

```text
benchmark/industrial/datasets/
  discovery.py
  local_io.py
benchmark/industrial/evaluation/
  analytics.py
  kernel_learning.py
  markdown.py
  okhs_quality.py
benchmark/industrial/experiments/
  artifacts.py
  incremental_persistence.py
  manifests.py
  presets.py
  progress.py
  registry.py
  run_compare.py
  verbosity.py
benchmark/industrial/visualization/
  forecasting.py
```

## Experiment Layout

Kernel-learning experiment runners are grouped by task:

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

The executable `run_*.py` files are thin shell entrypoints. They may read CLI
arguments or environment variables, but typed experiment defaults and model
lists live in `benchmark/experiments/kernel_learning/defaults.json` and are
normalized by `benchmark.experiments.kernel_learning.configs`.

Suite-style runners should follow the shared path:

```python
typed_config.build_suite_config() -> run_registered_suite(...)
```

This keeps resolved configs, final metrics, registry entries, aggregation, and
run summaries on one persistence path. Shared script controls such as
environment-variable parsing live in
`benchmark.experiments.kernel_learning.controls`.

## Constants Layout

As a benchmark module rule of thumb, table-like constants should live outside
implementation files when they can be represented as data. Prefer a small JSON
file next to the thematic package entrypoint for:

- default `ModelSpec` lists;
- benchmark metrics and publication metric sets;
- default dataset or series id lists;
- output directory templates and run metadata defaults.

Implementation modules should load, validate, normalize, and convert that data
into typed records. They should not become long catalogs of model names,
adapter parameters, or experiment constants.

Current constants files:

```text
benchmark/experiments/kernel_learning/defaults.json
benchmark/industrial/experiments/preset_defaults.json
```

## Model Layout

Model implementations should live next to the model package entry point:

```text
benchmark/industrial/models/
  classification.py
  kernel_artifacts.py
  regression.py
  forecasting.py
  specs.py
```

Avoid adding a package whose `__init__.py` is only a long list of imports from
unrelated task modules. The `__init__.py` may expose the public surface, but the
implementation that a developer needs to edit should be in the same package.

## Removed Legacy Areas

The old helper package, versioned runtime package, and root-level legacy
benchmark files are removed from the code tree.

Legacy wrappers that still matter live under:

```python
benchmark.industrial.legacy
```

Generated outputs must stay outside the code package:

- `benchmark/results/` stores benchmark result artifacts.
- `benchmark/benchmark/` is treated as a local generated artifact path and is
  ignored by Git.
- `benchmark.industrial` contains benchmark code, not saved experiment outputs.

## Migration Rule

When touching benchmark code, use:

```python
from benchmark.industrial import ModelSpec, run_local_benchmark_preset
```

Do not add compatibility imports for the removed versioned package. If a
downstream notebook or script still depends on old paths, migrate it to
`benchmark.industrial` as part of the change.
