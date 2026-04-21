# Forecasting Refactor Phase 3 Roadmap

## Purpose

This document is the source-of-truth for the third refactor cycle of the forecasting stack in
`Fedot.Industrial`.

Phase 2 completed the architectural bridge work:

- legacy forecasting boundaries were contracted;
- stage vocabulary became explicit and benchmark-visible;
- stage-tuning contracts became runtime-capable;
- `okhs` moved closer to a shell-first runtime;
- native neural heads became stage citizens through a bridge layer.

Phase 3 is about turning that bridge-based architecture into a **true primitive-execution architecture**:

- convert remaining bridge/shell layers into first-class primitive nodes;
- move from shell-level orchestration toward graph-native forecasting execution;
- reduce the remaining compatibility-only layers around `okhs`, composites, and neural heads;
- make benchmark, routing, and tuning operate over model families and primitive graphs, not wrapper-specific logic.

## Entry State

Phase 3 starts from the following already-implemented baseline:

- tensor-native forecasting runtime is in place;
- stage vocabulary is stable:
    - `trajectory_transform`
    - `decomposition`
    - `rank_truncation`
    - `forecast_head`
    - `ensemble`
- primitive-stage tuning is implemented through:
    - stage plans
    - stage search spaces
    - sequential stage execution
    - runtime evaluation on holdout forecasting series
- `benchmark/v2` already persists:
    - regime diagnostics
    - stage diagnostics
    - stage tuning reports
    - family-level stage tuning summaries
    - routing-family summaries
- native neural heads already have:
    - bridge layer
    - adapter support
    - stage diagnostics
    - stage tuning support
    - benchmark execution coverage
- `okhs_fdmd_forecaster` and related runtime helpers are already strongly extracted, but not yet fully represented as
  standalone primitive nodes.

## Phase 3 Goal

The main goal of Phase 3 is:

**make primitive forecasting nodes and graph execution the actual runtime substrate, so shell models become optional
packaging rather than the main execution mechanism.**

In practice this means:

- `okhs`, low-rank linear models, and neural heads become more graph-native;
- bridge layers shrink or disappear where they only proxy primitive behavior;
- benchmark and routing reason about model families and stage graphs directly;
- future forecasting work can be added by composing primitives, not by creating new monolith wrappers.

## Architectural Priorities

### 1. Convert Neural Bridge Into First-Class Primitive Heads

Phase 2 gave neural models a clean bridge into the forecasting stage vocabulary.

Phase 3 should convert that bridge into a more native primitive-head layer:

- define first-class neural head primitives for:
    - `patch_tst_model`
    - `tcn_model`
    - `deepar_model`
    - `nbeats_model`
- reduce reliance on bridge-only wrappers where they merely adapt params and diagnostics;
- keep a thin compatibility shell only when it is useful for repository or user-facing naming.

Desired result:

- neural forecasting is represented like other forecast-head families, not as a special-case bridge.

### 2. Make OKHS a Real Primitive Graph

`okhs_forecasting` is now much thinner, but it still owns too much kernel/runtime coordination.

Phase 3 should continue extraction toward explicit primitives:

- `hankelisation`
- decomposition representation selection
- rank truncation
- `okhs_fdmd_head`
- direct kernel head where still relevant

Desired result:

- `okhs_fdmd_forecaster` becomes a named composition over true primitives;
- kernel-level compatibility shells stop being the place where substantive orchestration lives.

### 3. Introduce Graph-Native Forecasting Execution

Stage tuning and diagnostics are already explicit, but execution still routes through shell models in many places.

Phase 3 should introduce a graph-native execution path:

- explicit primitive graph execution for forecasting pipelines;
- graph-level runtime metadata;
- graph-level stage tuning and evaluation;
- clear mapping from primitive graph to benchmark artifacts.

Desired result:

- we can evaluate and tune a forecasting graph without requiring a bespoke shell model per composition.

### 4. Unify Family-Level Benchmark and Routing Logic

Benchmark and routing already understand families, but still depend on adapter-centric resolution in many cases.

Phase 3 should shift the center of gravity from adapter names to family/graph semantics:

- routing should recommend families and preferred primitive graph patterns;
- benchmark should compare:
    - family baselines
    - primitive graph variants
    - tuned vs untuned graph paths
- artifacts should clearly separate:
    - shell-level metadata
    - graph-level metadata
    - family-level evaluation summaries

Desired result:

- family comparisons become the main unit of analysis, and shell names become secondary presentation labels.

### 5. Reduce Compatibility-Only Layers to Stable Packaging Shells

By this point, compatibility wrappers should still exist where useful, but they should stop hiding runtime logic.

Phase 3 should narrow:

- bridge-only forecasting wrappers
- legacy shell-only parameter normalization
- adapter-specific orchestration that duplicates graph logic

Desired result:

- compatibility layers are easy to reason about because they package behavior instead of implementing it.

## PR Stack

### PR-1. Neural Primitive Head Extraction

Goal:

- replace bridge-first neural forecasting integration with first-class neural forecast-head primitives.

Scope:

- define neural primitive-head execution contracts;
- migrate `patch_tst_model`, `tcn_model`, `deepar_model`, `nbeats_model` away from bridge-centric runtime paths;
- keep stage diagnostics and stage tuning behavior stable.

Acceptance:

- native neural heads are executable without relying on a separate bridge abstraction for core behavior;
- benchmark and tuning still see the same family/stage semantics.

Current progress:

- primitive-oriented source-of-truth for native neural heads now lives in `neural_forecast_head.py`;
- the following concerns are already extracted out of the bridge layer:
    - model registry
    - typed primitive spec
    - FEDOT-compatible implementation layer
    - input-data construction
    - prediction normalization
    - stage diagnostics
    - fit/predict execution contract
- native neural execution now also has a dedicated run-result helper, so benchmark/runtime code can consume a shared
  primitive execution contract instead of rebuilding neural-head runs manually;
- `NeuralForecastHeadBridge` is now only a compatibility shell over `NeuralForecastHead`;
- benchmark-v2 and stage-tuning runtime already use the new primitive-oriented neural head as the main execution path;
- native neural forecasting operations are now wired through `ts_model` repository/runtime metadata instead of the old
  forecasting-only neural strategy path.

Close signal for `PR-1`:

- primitive source-of-truth exists;
- typed spec and typed run-result exist;
- FEDOT runtime implementation layer exists;
- benchmark/runtime/tuning all execute through the new primitive path;
- compatibility bridge remains available as a thin shell.

### PR-2. OKHS Primitive Graph Completion

Goal:

- complete the transition of `okhs` from extracted runtime helpers to true primitive-graph execution.

Scope:

- continue breaking kernel-level orchestration into standalone primitive-capable layers;
- formalize `okhs_fdmd_head` and direct-kernel head contracts;
- reduce remaining stateful orchestration inside compatibility shells.

Acceptance:

- `okhs_fdmd_forecaster` becomes a thin named graph wrapper over stable primitives;
- kernel-level classes are lifecycle-oriented compatibility shells, not orchestration centers.

Current progress:

- `OKHS` runtime already has a strongly extracted helper layer in `okhs_runtime.py`;
- `okhs_fdmd_forecaster` now also has:
    - typed spec
    - typed run-result
    - shared execution helper for benchmark/runtime usage
    - shared parameter normalization via `normalize_okhs_fdmd_params(...)`
    - canonical spec builder via `build_okhs_fdmd_spec(...)`;
- `stage_tuning_runtime` now instantiates `OKHS` through the shared builder instead of direct class construction;
- benchmark-v2 now resolves one normalized `OKHS` spec and reuses it for execution, metadata, and stage-tuning base
  parameters;
- `OKHS` now also has shared runtime helpers for prediction normalization and runtime diagnostics packaging, closer to
  the neural primitive-head pattern;
- targeted regression slice for the new typed/spec-based path is green: `8 passed`.

### PR-3. Graph-Native Forecasting Execution Runtime

Goal:

- introduce the first real graph-execution path for forecasting primitives.

Scope:

- execute forecasting graphs composed from primitive nodes;
- expose graph-level diagnostics and graph-level optimization metadata;
- support graph-level tuning entrypoints.

Acceptance:

- at least one non-trivial forecasting graph can execute end-to-end without bespoke model-shell orchestration.

### PR-4. Graph-Aware Tuning and Evaluation

Goal:

- move tuning and evaluation from shell wrappers toward graph-aware runtime flows.

Scope:

- stage tuning over graph nodes rather than only shell-declared groups;
- graph-level holdout and rolling-origin evaluation;
- reusable tuning/evaluation reports for benchmark and developer workflows.

Acceptance:

- tuned primitive graphs can be compared directly against untuned graphs and shell-packaged baselines.

### PR-5. Family-First Benchmark and Routing Consolidation

Goal:

- make family/graph analysis the default benchmark and routing perspective.

Scope:

- expand family-level routing evaluation;
- compare graph variants inside model families;
- make shell-name artifacts secondary to family/graph artifacts where appropriate.

Acceptance:

- benchmark reports can answer:
    - which family was recommended
    - which family actually won
    - which graph variant inside the family performed best

### PR-6. Compatibility Packaging Cleanup

Goal:

- reduce remaining compatibility wrappers to stable public packaging shells.

Scope:

- remove duplicated orchestration from bridge/wrapper classes;
- narrow wrapper responsibilities to naming, defaults, and public API continuity;
- mark truly legacy wrappers where appropriate.

Acceptance:

- compatibility shells remain useful, but no longer drive forecasting runtime semantics.

## Dependencies Between PRs

Recommended order:

1. `PR-1` Neural Primitive Head Extraction
2. `PR-2` OKHS Primitive Graph Completion
3. `PR-3` Graph-Native Forecasting Execution Runtime
4. `PR-4` Graph-Aware Tuning and Evaluation
5. `PR-5` Family-First Benchmark and Routing Consolidation
6. `PR-6` Compatibility Packaging Cleanup

Rationale:

- neural and OKHS are the two biggest remaining architecture-specific islands;
- graph execution should sit on top of primitive-ready heads, not precede them;
- graph-aware tuning is stronger once execution is graph-native;
- benchmark/routing consolidation should follow stable graph semantics;
- packaging cleanup is safest after the real runtime substrate has shifted.

## Testing Strategy

Phase 3 should preserve the testing style of earlier waves:

- pure-helper tests for extracted graph planning and execution logic;
- facade/boundary tests for public shells, adapters, and compatibility layers;
- invariant-oriented checks for:
    - graph determinism
    - stage-to-graph metadata consistency
    - family routing stability
    - tuned-vs-untuned reproducibility
    - artifact persistence and schema stability

Priority test slices:

- `test_stage_tuning.py`
- `test_stage_tuning_runtime.py`
- `test_forecasting_runtime.py`
- `test_forecasting_composites.py`
- `test_okhs_runtime.py`
- `test_okhs_forecasting_api.py`
- `test_neural_forecast_head_bridge.py` until bridge layers are fully replaced
- `test_benchmark_v2.py`

## Exit Criteria For Phase 3

Phase 3 can be considered complete when all of the following are true:

1. Native neural forecasting heads are represented as first-class primitive heads or their equivalent graph-native
   runtime nodes.
2. `okhs` execution is primarily graph-native, with compatibility shells owning little more than lifecycle and
   packaging.
3. At least one multi-stage forecasting graph executes directly through a graph runtime rather than a bespoke shell
   model.
4. Stage tuning and forecasting evaluation can operate over primitive graphs as first-class runtime objects.
5. `benchmark/v2` can compare graph variants inside families and publish family/graph-first artifacts.
6. Compatibility wrappers no longer contain meaningful forecasting runtime logic.

## What Phase 3 Does Not Try To Solve

To keep scope realistic, Phase 3 does **not** aim to:

- redesign every neural architecture internals;
- fully remove all FEDOT boundary objects;
- solve model-quality questions for every forecasting family;
- replace all shell models immediately with generic builders.

Instead, it aims to make **primitive-graph execution** the stable center of the forecasting stack.

## Practical Planning Note

Phase 3 should still be executed as a **PR stack**, not as one giant rewrite.

Reason:

- the stack is already partially modernized;
- the remaining risk is concentrated in graph execution semantics and hidden compatibility logic;
- smaller PRs will make it much easier to validate that primitive graphs, not wrappers, have become the actual
  source-of-truth.
