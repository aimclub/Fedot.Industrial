# Forecasting Refactor Phase 2 Roadmap

## Purpose

This document is the source-of-truth for the second refactor cycle of the forecasting stack in
`Fedot.Industrial`.

Phase 1 established the core direction:

- forecasting-first runtime;
- tensor-native internal contracts;
- primitive-stage vocabulary;
- named shell/composite models;
- benchmark-aligned diagnostics.

Phase 2 is about turning that foundation into a cleaner and more explicit execution model:

- finish shrinking legacy forecasting dependencies;
- move from model-level orchestration to primitive-graph orchestration;
- make tuning and evaluation explicit at the stage level;
- reduce the remaining hidden coupling between shell models and legacy FEDOT infrastructure.

## Entry State

Phase 2 starts from the following already-implemented baseline:

- `ForecastTensorBatch` and tensor-native forecasting runtime are in place.
- Primitive stages already exist:
    - `hankelisation`
    - `svd_decomposition`
    - `randomized_svd_decomposition`
    - `tensor_decomposition`
    - `explained_variance_truncation`
    - `statistical_rank_truncation`
    - `expert_rank_truncation`
- Named shell/composite models already exist:
    - `lagged_ridge_forecaster`
    - `low_rank_lagged_ridge_forecaster`
    - `hybrid_ensemble_forecaster`
    - `okhs_fdmd_forecaster`
    - `mssa_forecaster`
    - `havok_forecaster`
    - `ssa_forecaster` as compatibility wrapper
- `benchmark/v2` already persists:
    - regime diagnostics
    - generic stage diagnostics
    - OKHS stage diagnostics
    - hybrid ensemble diagnostics
- Legacy strategy entrypoints already redirect forecasting operations toward the new runtime in key paths.

## Phase 2 Goal

The main goal of Phase 2 is:

**make the primitive graph the real source-of-truth for forecasting training, tuning, and execution, rather than only
the diagnostic vocabulary around model wrappers.**

In practice this means:

- shell models become thinner;
- `okhs_forecasting` stops being a hidden backend-monolith;
- tuning is performed over explicit stage groups;
- legacy strategy modules stop containing meaningful forecasting logic;
- repository metadata and benchmark workflows speak the same architectural language.

## Architectural Priorities

### 1. Finish Legacy Boundary Contraction

The current legacy strategy modules are already much thinner, but they still remain a hidden dependency surface.

Phase 2 should make them clearly secondary:

- forecasting-specific logic should no longer live inside
    - `industrial_preprocessing_strategy.py`
    - `industrial_model_strategy.py`
- these modules should retain only:
    - compatibility shims
    - redirects
    - minimal delegation

Desired result:

- the forecasting runtime can be reasoned about without reading legacy multidimensional dispatch code.

### 2. Move OKHS to a Primitive-First Runtime

`okhs_fdmd_forecaster` already exposes a shell-first contract, but the kernel-level backend still concentrates too much
orchestration.

Phase 2 should continue extracting:

- trajectory planning
- decomposition representation decisions
- rank selection decisions
- prediction orchestration
- anti-smoothing logic

Desired result:

- `okhs_forecasting` becomes a compatibility shell over explicit primitive/runtime helpers;
- `okhs_fdmd_head` becomes a realistic next candidate for standalone primitive execution.

### 3. Make Stage-Level Tuning Explicit

Current runtime and repository layers already expose stage params, but tuning still happens mostly at the wrapper/model
level.

Phase 2 should introduce a more explicit tuning contract:

- trajectory stage first
- decomposition/rank stage second
- forecast head stage third

Desired default:

- `SequentialTuner` for complex models and composite pipelines
- `SimultaneousTuner` only for small/stable spaces

### 4. Unify Repository Vocabulary Around Stage-First Forecasting

The repository already knows about primitives and shell models, but not every first-class forecasting path is fully
normalized yet.

Phase 2 should ensure:

- repository metadata
- defaults
- search space
- primary forecasting model lists
- benchmark adapter aliases

all reflect the same canonical forecasting vocabulary.

### 5. Prepare Neural Heads as True Stage Citizens

Neural models are still closer to standalone worlds than to primitive-stage members.

Phase 2 should prepare the bridge, without trying to fully solve deep forecasting in one pass:

- define what a neural forecast head should look like in the primitive graph
- clarify how it plugs into:
    - trajectory transforms
    - decomposition/rank stages
    - composite ensembles

## PR Stack

### PR-1. Legacy Forecasting Boundary Cleanup

Goal:

- reduce forecasting responsibility inside legacy strategy modules to the minimum possible shell.

Scope:

- remove or isolate remaining forecasting-specific branches from:
    - `industrial_model_strategy.py`
    - `industrial_preprocessing_strategy.py`
- keep only:
    - compatibility wrappers
    - redirect helpers
    - narrow public entrypoints

Acceptance:

- no new forecasting behavior depends on legacy multidim dispatch internals;
- forecasting redirects are explicit and tested.

Current progress:

- centralized redirect helpers now live in `forecasting_runtime_strategy.py`;
- legacy model and preprocessing strategies delegate through the shared redirect policy instead of duplicating it
  locally;
- forecasting wrappers continue to exist, but the redirect contract is now source-of-truth and test-covered;
- the legacy redirect `__new__` logic is now extracted into dedicated forecasting boundary mixins, so the old strategy
  modules are thinner and more explicit;
- neural forecasting strategy was extracted from `industrial_model_strategy.py` into a dedicated forecasting-oriented
  module, which further reduces forecasting-specific responsibility inside the legacy strategy layer.

### PR-2. OKHS Primitive-Orchestration Extraction

Goal:

- continue decomposing `okhs_forecasting` from backend-monolith into explicit runtime helpers.

Scope:

- separate fit-plan and predict-plan responsibilities further;
- isolate remaining decomposition/rank/trajectory coordination;
- reduce direct orchestration living in the kernel class;
- keep compatibility behavior stable for benchmark and shell users.

Acceptance:

- `OKHSForecaster` becomes mostly lifecycle/state shell;
- extracted helpers cover the substantive stage decisions.

Current progress:

- fit-plan, prediction-plan and optimization-info construction are now extracted into `okhs_runtime.py`;
- DMD model instantiation fallback and projected-vs-dense prediction execution now live in pure helpers;
- direct OKHS fit and recursive direct prediction now also delegate through runtime helpers, so the kernel-level
  forecaster mostly coordinates lifecycle, state and compatibility behavior;
- the remaining shell-thinning work is now mostly about reducing the last pieces of inline orchestration and aligning
  the extracted helper layer with future standalone forecast-head primitives.

### PR-3. Explicit Stage-Tuning Contract

Goal:

- implement a practical tuning flow that matches the primitive graph.

Scope:

- introduce stage-group tuning helpers:
    - trajectory stage
    - decomposition/rank stage
    - forecast head stage
- define per-model defaults for sequential tuning
- make these flows usable by:
    - shell models
    - composites
    - benchmark experiments

Acceptance:

- tuning no longer depends on accidental wrapper-level parameter blobs;
- stage tuning order is explicit and reproducible.

Current progress:

- a typed stage-tuning contract now exists for stage-native forecasting shells and compatibility wrappers;
- canonical groups are published in the same order across models: trajectory, decomposition/rank, forecast head and
  ensemble where applicable;
- implementations can already expose machine-readable tuning plans, which prepares the next cycle for actual sequential
  tuning execution;
- stage-specific search-space slices are now derivable from the same contract, so wrapper-level tuning no longer needs
  to guess parameter ownership per stage;
- a pure sequential stage-tuning executor now exists and records applied vs ignored parameters per stage, which gives
  the next cycle a stable foundation for real tuner orchestration;
- implementations can already expose `run_stage_tuning(...)` over the new executor, so the next refactor step can focus
  on plugging this layer into real FEDOT/objective runtimes instead of inventing stage orchestration from scratch;
- a dedicated stage-tuning runtime bridge now evaluates stage candidates on real forecasting series through explicit
  holdout splits and forecasting metrics, which means stage tuning is no longer limited to abstract objective callbacks;
- stage-native implementations can now expose `run_stage_tuning_on_series(...)`, returning `baseline vs tuned`
  evaluation reports with real model diagnostics.

### PR-4. Repository and Search-Space Consolidation

Goal:

- align metadata, defaults, aliases, and optimization space with the actual stage-first forecasting stack.

Scope:

- normalize model names and aliases;
- expand or prune `PRIMARY_FORECASTING_MODELS` using the new architecture;
- ensure all first-class forecasting shells have:
    - defaults
    - repository entries
    - search-space presence when appropriate

Acceptance:

- repository vocabulary matches actual supported forecasting shells and primitives;
- benchmark adapters and routing aliases stay consistent.

Current progress:

- canonical forecasting alias normalization now lives in a dedicated repository helper module;
- default parameter lookup, benchmark adapter resolution and routing-family mapping share the same alias policy;
- stage-native forecasting shells now have explicit search-space coverage, including canonical short aliases for `mssa`
  and `havok`.

### PR-5. Benchmark Stage Adoption Completion

Goal:

- make `benchmark/v2` a full architectural mirror of the stage-first stack.

Scope:

- ensure every stage-native forecasting model persists meaningful stage artifacts;
- add missing family-level summaries where useful;
- improve model-family comparison using the primitive vocabulary rather than raw model names.

Acceptance:

- publication pack can compare forecasting families and stage behaviors consistently across shells/composites.

Current progress:

- `benchmark/v2` can now persist optional `stage_tuning_report` payloads for stage-native forecasting models without
  changing the default fast path;
- a dedicated benchmark artifact flow now publishes:
    - series-level `*_forecasting_stage_tuning.json`
    - aggregate `forecasting_stage_tuning.{csv,tex,parquet}`
    - aggregate `forecasting_stage_tuning_family_summary.{csv,tex,parquet}`
- benchmark-side tuning reports already include `baseline vs tuned` metric comparison and the chosen best-parameter
  payload, which makes cohort-level tuning analysis possible in the existing publication pack;
- family-level summaries now expose tuning improvement rate, routing-family match rate and average gain, so the
  benchmark can already compare whether tuning helps each forecasting family on a cohort rather than only on isolated
  series;
- routing publication artifacts now also include family-level winner comparisons, which
  makes `recommended family vs actual winner family` analysis available at cohort level without reconstructing it from
  raw run records.

### PR-6. Neural Forecast Head Bridge

Goal:

- define the first clean integration path for neural forecasting into the primitive graph.

Scope:

- specify neural head interface expectations;
- connect at least one neural forecaster conceptually to:
    - stage diagnostics
    - benchmark metadata
    - composite branch participation

Acceptance:

- neural forecasting is no longer treated as an architectural exception;
- a future Phase 3 can add first-class neural primitive heads without redesigning the runtime again.

Current progress:

- repo-native neural forecasting models now have an explicit bridge layer through `neural_forecast_head_bridge.py`;
- neural heads already expose the same high-level stage vocabulary:
    - `trajectory_transform`
    - `decomposition`
    - `rank_truncation`
    - `forecast_head`
- stage-tuning plans and stage-search spaces now support:
    - `patch_tst_model`
    - `tcn_model`
    - `deepar_model`
    - `nbeats_model`
- benchmark-v2 adapter resolution now understands these neural runtime heads, so they can participate in the same
  stage-aware diagnostics pipeline as the rest of the forecasting stack;
- benchmark-v2 execution is now covered by a focused regression path for native neural heads:
    - stage diagnostics artifacts
    - stage-tuning artifacts
    - family-level stage-tuning summary rows for `neural_forecaster`
- there is now a dedicated runnable benchmark-v2 example for the neural bridge path:
    - `benchmark/v2/examples/neural_stage_suite_140426.py`
- there is now a family-level runnable suite for native neural heads:
    - `benchmark/v2/examples/m4_neural_family_suite_140426.py`
- routing family mapping also recognizes native neural heads as `neural_forecaster`, which prepares the next cycle for
  routing- and benchmark-level family comparisons that include neural models as first-class citizens.

Close signal for `PR-6`:

- native neural heads are now validated across:
    - stage vocabulary
    - stage-tuning contracts
    - runtime bridge on real holdout evaluation
    - benchmark-v2 adapter execution
    - benchmark-v2 stage artifacts and family summaries
- this is enough to treat `PR-6` as closed for Phase 2, even though a future phase may still replace the bridge with
  first-class neural primitive heads.

## Dependencies Between PRs

Recommended order:

1. `PR-1` Legacy Forecasting Boundary Cleanup
2. `PR-2` OKHS Primitive-Orchestration Extraction
3. `PR-3` Explicit Stage-Tuning Contract
4. `PR-4` Repository and Search-Space Consolidation
5. `PR-5` Benchmark Stage Adoption Completion
6. `PR-6` Neural Forecast Head Bridge

Rationale:

- `PR-1` and `PR-2` reduce hidden coupling before tuning work gets deeper.
- `PR-3` should sit on top of a cleaner runtime and cleaner OKHS shell.
- `PR-4` keeps repository/search-space honest after the tuning contract is clarified.
- `PR-5` is stronger when the vocabulary and runtime contract are already stabilized.
- `PR-6` should land only after the non-neural primitive graph is coherent.

## Testing Strategy

Phase 2 should keep the same testing philosophy as Phase 1:

- facade/boundary tests for public shells and adapters;
- pure-helper tests for extracted planning/decision logic;
- invariant-oriented tests for:
    - deterministic routing
    - stable stage metadata
    - stage artifact persistence
    - repository alias consistency
    - tuning-order determinism

Priority test slices:

- targeted `pytest` slices for:
    - `test_forecasting_runtime_strategy.py`
    - `test_forecasting_composites.py`
    - `test_okhs_runtime.py`
    - `test_okhs_forecasting_api.py`
    - `test_benchmark_v2.py`
    - repository/search-space contract tests where added

## Exit Criteria For Phase 2

Phase 2 can be considered complete when all of the following are true:

1. Legacy strategy modules no longer contain meaningful forecasting business logic.
2. `okhs_forecasting` is primarily a shell over extracted primitive/runtime helpers.
3. Tuning over the primitive graph is explicit and stage-aware.
4. Repository/default/search-space vocabulary matches the actual shell-first forecasting stack.
5. `benchmark/v2` compares forecasting families using the same stage vocabulary used by runtime and repository layers.
6. Neural forecasting has a defined bridge into the primitive graph, even if not yet fully expanded.

## What Phase 2 Does Not Try To Solve

To keep scope realistic, Phase 2 does **not** aim to:

- redesign all deep forecasting models completely;
- replace every remaining FEDOT boundary object;
- fully remove every legacy compatibility shim;
- solve all forecasting quality issues model-by-model.

Instead, it aims to finish the architectural transition so later model-quality work happens on a cleaner base.

## Practical Planning Note

Phase 2 should be executed as a **PR stack**, not as one giant branch-wide rewrite.

Reason:

- the architecture is already moving in the right direction;
- the remaining risk is mostly hidden coupling and vocabulary drift;
- smaller PRs make it easier to validate that the new primitive graph remains the real source-of-truth.
