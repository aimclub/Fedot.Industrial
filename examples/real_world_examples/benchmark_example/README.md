# Benchmark Example Scenarios

This package keeps real-world benchmark wrappers and historical result analysis
separate from raw local data. Python entrypoints build current
`benchmark.industrial` configs; notebooks in `analysis_of_results` render
publication-style artifacts from reusable analytics functions into the central
`examples/artifacts/cloud_bundle` hub.

## Layout

| Path | Task | Purpose |
| --- | --- | --- |
| `analysis_of_results/` | mixed | Thin notebooks for leaderboard, rank, delta, and evolution views. |
| `classification/` | ts_classification | UCR and PDL/SOTA wrapper scripts plus local result manifest. |
| `regression/` | ts_regression | TSER wrapper scripts plus local result manifest. |
| `detection/` | anomaly_detection | SKAB wrapper script plus local result manifest. |
| `forecasting/` | forecasting | M4/Kaggle/Monash wrapper scripts plus local result manifest. |
| `full_runs/` | mixed | Resumable full-run configs for UCR, TSER, M4, and two-stage UCR. |

Raw result directories are intentionally outside git and are described by
`results_manifest.json` or the historical manifests under
`examples/utils/data/benchmark_history`.
