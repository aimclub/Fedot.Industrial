# Benchmark Archive Inputs

This directory is a manifest boundary for local archive data used by
`examples.real_world_examples.benchmark_example`. The raw contents are useful
for analysis notebooks, but they are not reproducible source code and must stay
outside git.

## Inputs

| Path | Use case |
| --- | --- |
| `composition_results/` | Evolution notebook: generation-level fitness dynamics, model complexity, and notable composite pipelines. |
| `data/m3/`, `data/m4/` | Forecasting notebooks and full local benchmark reruns. |

Use `examples/real_world_examples/external_data_manifest.json` for DVC paths and
public Google Drive placeholders.
