# Benchmark Results

`benchmark/results/showcase` is the canonical entrypoint for benchmark result
comparison tables used by the Kernel Learning PR.

Benchmark infrastructure rules, typed config conventions, artifact layout, and
the checklist for adding a new benchmark direction are documented in the
project wiki:
https://github.com/aimclub/Fedot.Industrial/wiki/Benchmark-Infrastructure

Run:

```bash
python -m benchmark.results.showcase
```

The showcase reads the manifest in
`benchmark/results/showcase/showcase_manifest.json`, indexes current Industrial
Kernel Learning runs, selected `v2_kernel_learning` reference runs, and legacy
SoTA comparison CSV files. Raw historical folders remain in place and are listed
as archive candidates instead of being duplicated or silently mixed into the
current comparison tables.

Use `benchmark/results/showcase/tables/benchmark_overview.csv` and
`benchmark/results/showcase/tables/current_best_per_dataset.csv` as the first
tables to inspect.
