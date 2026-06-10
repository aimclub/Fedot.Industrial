# Classification Benchmark Examples

Current wrappers build task-typed `benchmark.industrial` classification configs
and current Industrial experiment entrypoints. Historical local UCR result
directories remain useful for comparison notebooks, but they are external inputs
described in `results_manifest.json`.

## Files

| File | Use case |
| --- | --- |
| `PDL_uni.py`, `PDL_multi.py` | PDL classification wrapper contexts. |
| `SOTA_uni.py`, `SOTA_multi.py` | SOTA comparison wrapper contexts. |
| `results_manifest.json` | Expected local result directories and metric semantics. |
