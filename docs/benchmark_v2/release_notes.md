# Benchmark Industrial Release Notes

These notes supersede the old Benchmark V2 release notes. The active runtime package is `benchmark.industrial`.

## Summary

The benchmark layer now uses one canonical package for forecasting, time series classification, and time series regression. The old versioned runtime name should not be used in new imports.

## What Is Included

- Unified benchmark schema for `forecasting`, `ts_classification`, and `ts_regression`.
- Local dataset adapters for M4, Monash, UCR/UEA-style classification, and TSER data.
- JSON manifest-driven execution.
- Preset-based execution for standard local runs.
- Registered run bundles with result persistence.
- Shared aggregation, diagnostics, and visualization modules.
- Public result showcase under `benchmark/results/showcase`.
- Compatibility wrappers under `benchmark.industrial.legacy` where older flows still need them.

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

## CLI

```bash
python -m benchmark.industrial <preset> [options]
python -m benchmark.industrial --manifest <path> [options]
```

## Current Limitations

- External deep baselines are optional and may be reported as `skipped` or `not_available`.
- Historical result folders can be used as reference data, but they are not runtime modules.
- The benchmark layer is optimized for local reproducible runs and artifact generation, not as a dashboard product.

## Related Documents

- [Benchmark infrastructure guide](../dev_guide/benchmark_infrastructure.md)
- [Quickstart](quickstart.md)
- [Migration Guide](migration_guide.md)
