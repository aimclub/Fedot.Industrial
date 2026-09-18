# Full Benchmark Runs

This package defines resumable full Industrial benchmark runs used by
`benchmark_example/analysis_of_results`.

`full_run_defaults.json` stores run-level constants. `current_api.py` builds the
actual configs from existing Kernel Learning experiment APIs so dataset
discovery stays source-backed instead of hard-coded.

```powershell
python -c "from examples.real_world_examples.benchmark_example.full_runs import run_full_benchmark; run_full_benchmark('ucr_full')"
python -c "from examples.real_world_examples.benchmark_example.full_runs import run_full_benchmark; run_full_benchmark('tser_full')"
python -c "from examples.real_world_examples.benchmark_example.full_runs import run_full_benchmark; run_full_benchmark('m4_full')"
```

Runs use `resume_enabled` from the underlying `benchmark.experiments` configs
and write artifacts under `benchmark/results/kernel_learning/...`.
