# Radical Forecasting Refactor Plan

## Purpose

This document is the source-of-truth for the first radical refactor wave of the forecasting stack in `Fedot.Industrial`.

The target architecture is:

- forecasting-first rather than generic-FEDOT-first;
- tensor-native inside the forecasting runtime;
- composable by primitive stages instead of monolithic forecaster classes;
- benchmark-aligned with `benchmark/v2`;
- intentionally willing to simplify or retire legacy compatibility paths.

## Why The Refactor Is Needed

The current forecasting stack contains several architectural bottlenecks:

- forecasting models mix trajectory construction, decomposition, rank selection, head fitting, diagnostics, and
  orchestration in one class;
- model tuning is often delegated to generic FEDOT loops that are better suited for tabular or generic pipeline cases
  than for multi-horizon forecasting;
- `industrial_preprocessing_strategy` and `industrial_model_strategy` are overloaded with legacy dispatch logic and old
  compatibility assumptions;
- regime diagnostics and benchmark artifacts are richer than the forecasting runtime itself, which creates a mismatch
  between observability and implementation structure;
- models such as `okhs_forecasting` and `lagged_forecaster` are harder to tune and reason about than their mathematical
  pipeline suggests.

## Architectural Defaults

- Scope of wave 1: only the forecasting stack.
- Canonical evaluation layer: `benchmark/v2`.
- Canonical internal data format: `torch.Tensor`.
- FEDOT `InputData` / `OutputData` remain boundary adapters, not the internal forecasting runtime format.
- Public composites in v1 are named models, not a generic builder-first API.
- Backward compatibility is not a primary constraint.

## Primitive Graph

The new forecasting stack is built around a primitive graph:

1. `trajectory_transform`
2. `decomposition`
3. `rank_truncation`
4. `forecast_head`

### Primitive Families

#### Trajectory Transform

- `hankelisation`
- later: `page_embedding`
- later: `tensor_hankelisation`

#### Decomposition

- `svd_decomposition`
- `randomized_svd_decomposition`
- later: `tensor_decomposition`

#### Rank Truncation

- `explained_variance_truncation`
- `statistical_rank_truncation`
- `expert_rank_truncation`

#### Forecast Heads

- `ridge_forecasting_head`
- `okhs_fdmd_head`
- `havok_head`
- `weighted_average_head`
- later: `deepar_head`, `nbeats_head`

## Runtime Types

Wave 1 introduces explicit typed runtime objects:

- `ForecastTensorBatch`
- `TrajectoryTransformResult`
- `DecompositionResult`
- `RankTruncationResult`
- `ForecastHeadResult`
- `ForecastingSplitSpec`
- `ForecastingEvaluationResult`
- `ForecastingOperationCapability`
- `TensorDevicePolicy`

These types are designed to carry tensor-native forecasting state without depending on generic FEDOT dispatch semantics.

## Model Migration Strategy

### Thin Shell Principle

Forecasting models should become thin shells that:

- define sensible defaults;
- assemble primitive stages;
- aggregate diagnostics for API and benchmark consumers.

The heavy lifting must move into deterministic runtime helpers and primitive stages.

### Migration Targets

#### `lagged_forecaster`

Long-term source-of-truth:

- `hankelisation -> ridge_forecasting_head`

#### `okhs_forecasting`

Long-term source-of-truth:

- `hankelisation -> decomposition -> rank_truncation -> okhs_fdmd_head`

#### `ssa_forecaster` / `mssa_forecaster`

Long-term source-of-truth:

- `page_embedding -> decomposition -> rank_truncation -> linear_head`

#### `havok_forecaster`

Long-term source-of-truth:

- `hankelisation -> svd_decomposition -> havok_head`

## Named Composite Models

Wave 1 adds three named composite models:

### `lagged_ridge_forecaster`

- `hankelisation -> ridge_forecasting_head`

### `low_rank_lagged_ridge_forecaster`

- `hankelisation -> svd_decomposition -> rank_truncation -> ridge_forecasting_head`

### `hybrid_ensemble_forecaster`

- branch A: `lagged_ridge_forecaster`
- branch B: `low_rank_lagged_ridge_forecaster`
- branch C: a complex branch such as `havok` or `okhs`
- final head: `weighted_average_head`

## Explicit Forecasting Runtime

The generic FEDOT forecasting split/evaluation loop is not the target runtime for new industrial forecasting models.

Wave 1 introduces an explicit industrial forecasting runtime with:

- holdout split semantics;
- rolling-origin-ready typed contracts;
- multi-horizon metric evaluation;
- stage-aware diagnostics collection;
- direct tensor-native state flow.

## Benchmark Alignment

`benchmark/v2` remains the canonical evaluation layer.

The refactor should ensure:

- `regime_diagnostics` are always persisted;
- model metadata is stage-aware;
- composite models preserve branch-level diagnostics;
- routing evaluation can be compared against winner model families.

## Current Wave-1 Implementation Status

### Implemented In This Slice

- `docs/dev` source-of-truth for the refactor;
- tensor-native runtime foundation for forecasting primitives;
- first named composite models;
- a new shell-first operator entrypoint:
  - `okhs_fdmd_forecaster`
- extracted pure helper layer for OKHS orchestration:
  - fit-plan construction
  - stage-diagnostics construction
  - prediction-plan helpers
  - anti-smoothing postprocess helpers
- benchmark adapters for named composite models;
- model family mapping for routing-aware metadata.
- stage-aware benchmark artifacts for `okhs_fdmd_forecaster`:
  - per-series `okhs_fdmd_stage_diagnostics.json`
  - aggregate `okhs_fdmd_stage_diagnostics` tables in the publication pack
- forecasting decomposition and rank-truncation primitives as separate industrial operations:
  - `svd_decomposition`
  - `randomized_svd_decomposition`
  - `tensor_decomposition`
  - `explained_variance_truncation`
  - `statistical_rank_truncation`
  - `expert_rank_truncation`
- tuning search-space entries for named composite models and primitive forecasting stages;
- an M4 benchmark example for the new composite forecasting stack.
- dedicated thin forecasting runtime strategies:
  - `IndustrialForecastingModelRuntimeStrategy`
  - `IndustrialForecastingPreprocessingRuntimeStrategy`
- repository forecasting metadata redirected away from legacy multidimensional dispatch and toward forecasting-only
  runtime entrypoints.
- legacy classes
  - `IndustrialSkLearnForecastingStrategy`
  - `IndustrialForecastingPreprocessingStrategy`
    now act as compatibility wrappers over the new forecasting runtime strategies.

### Deferred To Later Slices

- full removal or reduction of the legacy forecasting codepaths still living inside
  `industrial_preprocessing_strategy` and `industrial_model_strategy`;
- full decomposition of `okhs_forecasting` into FEDOT pipeline-compatible primitive nodes;
- full primitive-node registration in industrial JSON repositories;
- explicit rolling-origin tuning orchestration across all forecasting models;
- neural forecast heads as first-class primitives.

## Delivery Phases

### Phase 1. Runtime Foundation

- introduce typed tensor-native runtime;
- keep FEDOT objects only at boundaries;
- stabilize `hankelisation` as the reference trajectory transform.

### Phase 2. Primitive Operations

- introduce decomposition and rank-truncation primitives;
- normalize stage diagnostics.

### Phase 3. First Migrated Models

- migrate lagged and low-rank lagged composites;
- begin shell-first migration path for operator models.
- `okhs_fdmd_forecaster` now provides a stage-aware forecasting shell over the existing OKHS/fDMD backend.

### Phase 4. Composite Models

- add explainable weighted ensembles;
- preserve branch-level diagnostics.

### Phase 5. Strategy Rewrite

- narrow legacy industrial strategies to thin shells;
- detach forecasting runtime from old multidimensional compatibility flags.

Current status:

- started;
- forecasting repository metadata now points to dedicated forecasting runtime strategies;
- remaining work is to migrate lingering forecasting-specific logic out of the legacy strategy modules entirely.

### Phase 6. Benchmark And Routing Adoption

- align benchmark artifacts with primitive vocabulary;
- compare routing recommendation against winner families on real cohorts.

Current status:

- in progress;
- `benchmark/v2` now persists stage-aware series and aggregate artifacts for `okhs_fdmd_forecaster`;
- next useful step is to extend the same primitive-stage artifact contract to additional model families beyond OKHS.

## Practical Rule For New Development

Any new forecasting model added after this refactor wave should answer three questions explicitly:

1. What is the trajectory transform?
2. What is the latent/decomposition stage?
3. What is the forecast head?

If those three pieces are not separable, the implementation should be treated as incomplete.
