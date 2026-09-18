# Regression Benchmark Examples

Current wrappers build task-typed `benchmark.industrial` TSER configs and keep
historical local result directories outside git. Use `results_manifest.json` to
connect local TSER dumps to analysis code.

## Files

| File | Use case |
| --- | --- |
| `PDL_multi.py` | PDL regression wrapper context. |
| `SOTA_multi.py` | SOTA comparison wrapper context. |
| `results_manifest.json` | Expected local result directories and metric semantics. |
