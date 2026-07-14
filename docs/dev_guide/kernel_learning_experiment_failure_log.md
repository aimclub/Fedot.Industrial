# Kernel Learning Experiment Failure Log

This document tracks runtime failures found while running the `benchmark/run_kernel_*` experiment entrypoints.

## Rules

- Log every experiment failure that is not already explained by unit tests.
- Do not recompute successful completed items on rerun; use benchmark resume artifacts.
- Stop after the same unresolved failure repeats across three reruns.
- Keep `kernel_selection`, `kernel_diagnostics`, and run metadata artifacts for every completed item.

## Runs

### 2026-06-02 - forecasting full-mode attempt 1

- Command: `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_forecasting.py`
- Status: fixed locally, rerun pending.
- Error: direct script execution failed before suite construction
  with `ModuleNotFoundError: No module named 'benchmark.v2'`.
- Cause: `benchmark/run_kernel_learning_forecasting.py` imported `benchmark.v2` before adding the repository root
  to `sys.path`; the same import-order bug was present in the UCR and TSER kernel learning entrypoints.
- Fix: moved repository-root bootstrap before all local benchmark imports
  in `benchmark/run_kernel_learning_forecasting.py`, `benchmark/run_kernel_learning_tser.py`,
  and `benchmark/run_kernel_learning_ucr.py`; added source-level tests that assert the bootstrap happens
  before the old versioned benchmark import.

### 2026-06-02 - forecasting full-mode attempt 2

- Command: `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_forecasting.py`
- Status: fixed locally, rerun pending.
- Error: M4 local loader failed
  with `BenchmarkConfigurationError: Local M4 files were not found: ...examples\data\m4\datasets\Monthly-train.csv and ...Monthly-test.csv`.
- Cause: the repository contains local M4 files in long-frame layouts such as `fedot_ind/data/M100/M4Monthly.csv`
  and `examples/data/benchmark/forecasting/M4Monthly.csv`, while the v2 M4 adapter only searched for
  wide `Monthly-train.csv` / `Monthly-test.csv`.
- Fix: added a typed `LocalM4FileLayout` resolver in `benchmark/v2/forecasting.py` that keeps the wide train/test layout
  as primary and falls back to local long-frame `M4{subset}.csv` files with tail-holdout splitting; added a unit test
  for the long local layout.

### 2026-06-03 - forecasting full-mode attempt 3

- Command: `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_forecasting.py`
- Status: fixed locally, rerun pending.
- Error: all 20 forecasting item evaluations finished with `ok=20 fail=0`, but publication pack rendering failed
  with `KeyError: 'metric_name'`.
- Cause: the run resumed from an existing empty progress index; `_load_resume_state()`
  replaced `ForecastingSuiteRunner.metric_records` and `prediction_records` with new lists
  while `ForecastingSeriesArtifactsRecorder` still pointed at the old lists. Fresh metrics were computed
  for `metrics_summary` but were not persisted into item-level `metric_records` / `prediction_records`.
- Fix: synchronized the artifacts recorder after resume-state loading, added stable empty-frame schemas
  in `benchmark/v2/analytics.py`, and added a resume fallback in `benchmark/v2/incremental_persistence.py` that
  reconstructs aggregate metric rows from `run_record.metrics_summary` for already completed success items whose
  detailed metric rows were lost before this fix.

### 2026-06-03 - TSER full-mode attempt 1

- Command: `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_tser.py`
- Status: fixed locally, rerun pending.
-
Error: `BenchmarkRegressionError: Could not resolve local TRAIN/TEST files for NaturalGasPricesSentiment in ...\data\NaturalGasPricesSentiment`.
- Cause: `benchmark/run_kernel_learning_tser.py` pointed `TSER_DATA_ROOT` at `PROJECT_ROOT/data`, where these dataset
  folders contain old experiment artifacts rather than Monash/UEA regression split files. The actual
  local `*_TRAIN.ts` / `*_TEST.ts` files are under `fedot_ind/data/<dataset>`.
- Fix: changed the TSER entrypoint to use `PROJECT_ROOT / "fedot_ind" / "data"` as its local data root while
  preserving `download_if_missing=False`.

### 2026-06-03 - TSER full-mode attempt 2

- Command: `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_tser.py`
- Status: fixed locally, rerun pending.
- Error: `NaturalGasPricesSentiment` completed with 4 successful model runs, then loading `AppliancesEnergy_TRAIN.ts`
  failed with `Timestamped .ts files are not supported yet`.
- Cause: `benchmark/v2/local_io.py` rejected `@timestamps true` `.ts` files and split dimensions with a
  plain `series_part.split(':')`, which is incompatible with timestamp values such as `17:00:00`.
- Fix: added timestamp-aware `.ts` parsing with depth-aware dimension splitting and value extraction
  from `(timestamp,value)` pairs; added a unit test for timestamped multivariate regression `.ts` splits.

### 2026-06-03 - UCR full-mode attempt 1

- Command: `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_ucr.py`
- Status: fixed locally, rerun pending.
- Error: the first successful item for `ACSF1` was persisted,
  then `KernelEnsembleClassifier_adaptive_all_non_topological` failed
  with `_detect_knee_point() missing 1 required positional argument: 'indices'`.
- Cause: `MatrixDecomposer` registered the channel-filter helper `_detect_knee_point(values, indices)` directly as the
  one-argument spectrum regularizer for `rank_regularization='knee_point'`. The adaptive UCR path reaches this through
  eigen-basis decomposition while preparing feature operators.
- Fix: added a spectrum-specific adapter in `fedot_ind/core/operation/decomposition/matrix_decomposition/decomposer.py`
  that supplies stable component indices to `_detect_knee_point` and preserves the one-argument regularizer contract;
  added focused unit tests for direct spectrum regularization and `MatrixDecomposer.apply`.

### 2026-06-03 - UCR full-mode attempt 2

- Command: `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_ucr.py`
- Status: fixed locally, rerun pending.
- Error: the resumed UCR run advanced to 38 persisted items,
  then `KernelEnsembleClassifier_adaptive_all_non_topological` failed on `ArrowHead` and `Car` with `StandardScaler`
  feature-width mismatches in `tabular_extractor`; `ChlorineConcentration` later failed
  with `[Errno 22] Invalid argument` during the same timed run.
- Cause: `TabularExtractor` fits `StandardScaler/PCA` on the raw train feature matrix but some internal feature
  pipelines can produce a different raw feature width for test splits before PCA projection. That violates the
  train/test feature-space contract expected by kernel matrix construction.
- Fix: made `TabularExtractor` remember the fitted raw train feature width and align transform-time raw features by
  deterministic truncation/padding before applying the fitted scaler/PCA; added focused reducer tests for both short and
  wide transform matrices.

### 2026-06-03 - UCR full-mode attempt 3

- Command: background `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_ucr.py` with stdout/stderr
  redirected to `benchmark/results/v2_kernel_learning/ucr_suite_020626/ucr_full_resume_*.log`.
- Status: stopped intentionally, fixed locally, rerun pending.
- Error: after the width-alignment fix, `ArrowHead` and `Car` adaptive reruns became successful,
  but `ChlorineConcentration` adaptive entered a very long `tabular_extractor` feature extraction over 3840 samples.
  Progress was still around 22% after several minutes, implying that full UCR across 111 datasets would become
  impractically long with unbounded tabular extraction.
- Cause: `tabular_extractor` is a heavy composite feature generator and was registered without a budget policy, unlike
  the topology generator. Large UCR datasets can dominate full-mode runtime before the selector can decide whether the
  generated kernel is useful.
- Fix: registered `tabular_extractor` through `BudgetedRepositoryFeatureGeneratorAdapter`
  with `max_samples=400`, `max_cells=50000`, and a `statistical_summary` fallback; extended lightweight fallback
  construction and added registry/fallback tests. The initial looser budget still allowed `ChlorineConcentration` train
  data through and only became expensive on transform, so the default budget was tightened before rerunning full mode.

### 2026-06-03 - UCR full-mode attempt 4

- Command: background `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_ucr.py` with stdout/stderr
  redirected to `benchmark/results/v2_kernel_learning/ucr_suite_020626/ucr_full_resume_budget2_*.log`.
- Status: stopped intentionally, fixed locally, rerun pending.
- Error: the run advanced to 125 successful items, but `FacesUCR` exposed the same train/test asymmetry in a
  smaller-train/larger-test form: train passed the budget while test-side tabular extraction over 2050 samples became
  slow.
- Cause: UCR datasets can have highly asymmetric train/test split sizes, so a budget that only admits moderately sized
  train matrices can still permit expensive transform-time tabular extraction.
- Fix: tightened the default `tabular_extractor` budget to `max_samples=150` and `max_cells=20000`, keeping true tabular
  extraction for small UCR datasets and routing larger/asymmetric datasets to the `statistical_summary` fallback before
  kernel fitting.

### 2026-06-03 - UCR full-mode attempt 5

- Command: background `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_ucr.py` with stdout/stderr
  redirected to `benchmark/results/v2_kernel_learning/ucr_suite_020626/ucr_full_resume_budget3_*.log`.
- Status: stopped intentionally, fixed locally, rerun pending.
- Error: the run advanced to 149 successful items, then `FreezerSmallTrain` showed an even smaller train/larger test
  asymmetry: train around 28 samples still passed the budget, while test-side tabular extraction over 2850 samples
  became slow.
- Cause: because UCR split asymmetry can be extreme, train-only admission for a heavy stateful tabular generator must be
  conservative enough to mean "tiny only" unless the benchmark supplies both train and test sizes to the generator
  budget.
- Fix: tightened the default `tabular_extractor` budget to `max_samples=25` and `max_cells=10000`, making tabular
  extraction a tiny-dataset path and routing larger/asymmetric UCR datasets to the `statistical_summary` fallback before
  kernel fitting.

### 2026-06-03 - UCR full-mode final resume

- Command: background `.\venv_3.9_new\Scripts\python.exe benchmark\run_kernel_learning_ucr.py` with stdout/stderr
  redirected to `benchmark/results/v2_kernel_learning/ucr_suite_020626/ucr_full_resume_budget4_*.log`.
- Status: complete.
- Evidence: the resumed run finished all 444 UCR model evaluations successfully under run
  id `kernel_learning_ucr_suite_c44f2a9f6c`.
- Artifacts: aggregate publication files were refreshed
  in `benchmark/results/v2_kernel_learning/ucr_suite_020626/kernel_learning_ucr_suite_c44f2a9f6c/aggregate` with 444 run
  rows, 1332 metric rows, 462896 prediction rows, and `status_counts: {"success": 444}` in `run_metadata.json`.
- Diagnostics: incremental JSONL records for kernel diagnostics, kernel selection, metrics, predictions, and runs were
  updated in `benchmark/results/v2_kernel_learning/ucr_suite_020626/kernel_learning_ucr_suite_c44f2a9f6c/records`.
