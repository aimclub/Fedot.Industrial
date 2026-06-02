# Kernel Learning Benchmark Runbook

## Scope

This runbook covers the MVP kernel-learning benchmark flow for supervised UCR time-series classification and the
two-stage warm-start experiment.

The public estimator API remains:

```python
from fedot_ind.core.kernel_learning import KernelEnsembleClassifier, KernelEnsembleRegressor
```

Experiment orchestration helpers live in:

```python
from fedot_ind.core.kernel_learning.experiments_api import (
    KernelLearningStage1Runner,
    KernelLearningStage2Runner,
)
```

Do not import `experiments_api` from `fedot_ind.core.kernel_learning.__init__`: it intentionally depends
on `benchmark.v2` and should stay out of the lightweight estimator import path.

## Stage 1: Kernel Learning On UCR

Use the UCR script when you want to train the kernel ensemble and persist kernel diagnostics after each dataset/model
run.

```powershell
python benchmark/run_kernel_learning_ucr.py
```

Important defaults:

- `UCR_DATA_ROOT = PROJECT_ROOT / "data"`;
- `UCR_DATASETS = ()` means all locally discovered datasets from the configured UCR benchmark allow-list;
- missing explicitly requested datasets are downloaded through the existing repository UCR loader
  when `download_if_missing=True`;
- topology is excluded from the default MVP generator set.

The UCR script now compares four kernel-learning variants:

- `KernelEnsembleClassifier_score_baseline_summary`: score-based selector baseline on summary features;
- `KernelEnsembleClassifier_adaptive_all_non_topological`: projected-gradient sparse MKL over the non-topological
  generator set;
- `KernelEnsembleClassifier_shapelet_motif_rbf`: shapelet/local-pattern features plus statistical summary features;
- `KernelEnsembleClassifier_embedding_nystrom`: embedding-style features with opt-in Nystrom kernel approximation.

Expected artifacts:

```text
benchmark/results/v2_kernel_learning/<run_id>/
  runs/<dataset>/<model>/
    run.json
    metrics.json
    predictions.csv
    kernel_diagnostics.json
    kernel_selection.json
  records/
    runs.jsonl
    metrics.jsonl
    predictions.jsonl
    kernel_diagnostics.jsonl
```

The per-run JSON artifacts are written incrementally, so completed datasets remain available even if a later dataset
fails.

## Stage 2: Warm-Start FEDOT Evolution

The two-stage script can either run stage 1 or load an existing stage 1 run.

Load an existing stage 1 run and execute stage 2:

```powershell
python benchmark/run_kernel_learning_ucr_two_stage.py `
  --stage1-run-id kernel_learning_ucr_stage1_ba419d49e4
```

Run stage 1 first, then stage 2:

```powershell
python benchmark/run_kernel_learning_ucr_two_stage.py --run-stage-1
```

Run stage 1 for selected datasets:

```powershell
python benchmark/run_kernel_learning_ucr_two_stage.py `
  --run-stage-1 `
  --datasets Coffee Lightning7
```

Load or run stage 1 only:

```powershell
python benchmark/run_kernel_learning_ucr_two_stage.py --skip-stage-2
```

Useful stage 2 controls:

```powershell
python benchmark/run_kernel_learning_ucr_two_stage.py `
  --stage2-output-dir benchmark/results/v2_kernel_learning/ucr_two_stage_optim_debug `
  --timeout-minutes 5 `
  --pop-size 5
```

Stage 2 writes one directory per dataset:

```text
benchmark/results/v2_kernel_learning/ucr_two_stage_optim_<date>/
  <dataset>/
    initial_population_specs.json
    fedot_config.json
    metrics.json
    predictions.csv
    optimizer_summary.json
  stage2_summary.json
```

Runtime initial assumptions are passed to `FedotIndustrial` as lazy `PipelineBuilder` objects. They are built after the
industrial operation repository is activated, so operations such as `wavelet_basis` resolve as data operations rather
than models.

## TSER Regression Suite

Use the TSER script to validate the regression path and compare score-based selection against adaptive MKL:

```powershell
python benchmark/run_kernel_learning_tser.py
```

The TSER script compares:

- `KernelEnsembleRegressor_score_linear_summary`;
- `KernelEnsembleRegressor_adaptive_rbf_summary`;
- `KernelEnsembleRegressor_shapelet_rbf`;
- `KernelEnsembleRegressor_embedding_nystrom`.

Expected artifacts follow the same `benchmark/results/v2_kernel_learning/<run_id>/` layout and include
`kernel_selection` plus `kernel_diagnostics` for each successful model run.

## Forecasting Suite

Use the forecasting script to validate the public `KernelEnsembleForecaster` through the benchmark-v2 forecasting
adapter:

```powershell
python benchmark/run_kernel_learning_forecasting.py
```

The forecasting script uses local M4 CSV files and a small sample size by default. It compares:

- `NaiveLastValue`;
- `LaggedRidgeForecaster`;
- `KernelEnsembleForecaster_identity_shapelet`;
- `KernelEnsembleForecaster_embedding_nystrom_okhs`.

The kernel forecasting adapter turns each train series into supervised lag windows, fits `KernelEnsembleForecaster`,
and stores kernel-learning artifacts in the run metadata.

## Stage 1 Analysis

Use the analyzer script to summarize saved kernel diagnostics and selected generators:

```powershell
python benchmark/analyze_kernel_learning_stage1.py `
  --run-dir benchmark/results/v2_kernel_learning/ucr_two_stage_140526/kernel_learning_ucr_stage1_ba419d49e4
```

The summary report and visualizations are useful for selecting datasets for stage 2 and for reviewing which feature
generators were most informative.

## Targeted Test Command

In the current Windows environment, pytest may try to autoload unavailable third-party plugins. Disable plugin autoload
for targeted kernel-learning checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& "D:\data_old\WORK\Repo\Industiral\IndustrialTS\venv_3.9_new\Scripts\python.exe" -m pytest -q `
  tests\unit\core\kernel_learning `
  tests\unit\api\utils\test_kernel_warm_start_strategy.py `
  tests\unit\models\test_benchmark_v2_kernel_learning.py `
  tests\unit\models\test_benchmark_v2_incremental_artifacts.py `
  tests\unit\models\test_kernel_learning_experiment_scripts.py `
  tests\unit\models\test_kernel_learning_stage1_analysis.py
```

## Pre-Merge Checklist

- Targeted kernel-learning pytest passes.
- `python -m py_compile` passes for benchmark scripts and `fedot_ind/core/kernel_learning`.
- Stage 1 artifacts can be loaded with `RUN_STAGE_1=False`.
- Stage 2 initial assumptions are lazy `PipelineBuilder` objects.
- `wavelet_basis`, `fourier_basis`, and `eigen_basis` resolve as data operations after industrial repository
  initialization.
- No topology generator is included in default MVP benchmark presets.
