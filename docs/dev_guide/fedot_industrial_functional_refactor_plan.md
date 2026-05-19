# FedotIndustrial Functional Flow Refactor Plan

## Summary

This document describes a staged refactor plan for `FedotIndustrial`.
The goal is to preserve the Pymonad-inspired functional style where it is useful, but move it into explicit helper
layers and keep `FedotIndustrial` itself as a readable orchestration facade.

The target shape is:

```text
FedotIndustrial facade
  -> typed request and plan helpers
  -> monadic flow helpers for pure transformations and expected branches
  -> services for effects: repository, Dask runtime, solver, prediction, metrics
```

The public API of `FedotIndustrial` should remain stable. The refactor should be delivered as several small PRs with
focused tests.

## Motivation

Current methods in `FedotIndustrial` actively use Pymonad chains such as:

```python
Either.insert(value).then(...).then(...)
```

This style is attractive because it reads like a pipeline of data transformations. However, several current chains mix
pure transformations with side effects:

- mutation of `self` or `self.manager`;
- repository activation;
- Dask server startup;
- solver construction;
- model fitting;
- conditional branches hidden in lambdas;
- generic exceptions instead of typed expected outcomes.

As a result, the code can look mathematical while being hard to debug. The refactor should make the style more honest:

- use monads for explicit domain alternatives and value transformations;
- use typed plans for branching and configuration decisions;
- keep lifecycle effects visible in services or in the facade orchestration.

## Design Rules

- Do not use `.then(lambda ...)` for hidden mutation of `self`.
- Effects should be named and isolated in services or explicit `tap`-style helpers.
- `Either` should represent expected alternatives or domain failures, not replace every `if`.
- `Maybe` or optional values should represent normal absence only.
- Raw config dictionaries should be converted into typed records near the boundary.
- `FedotIndustrial` should remain a facade and orchestrator, not a deep business-logic container.
- Pymonad chains should appear in pure or mostly-pure helper modules, not around large runtime effects.

## PR 1. RFC And Functional Flow Rules

### Goal

Document the coding rules before changing implementation.

### Changes

- Add developer documentation for functional flow usage.
- Describe where Pymonad is encouraged and where it should be avoided.
- Define rules for side effects, typed plans, expected errors, and service extraction.

### Tests

No runtime tests required. This is a documentation PR.

### Acceptance Criteria

- The document clearly explains allowed and discouraged Pymonad patterns.
- Future refactor PRs can reference the rules.

## PR 2. Monadic Helper Layer

### Goal

Introduce a small helper layer over Pymonad so chains become readable and debuggable.

### Suggested Location

- `fedot_ind/api/flow/monadic.py`
- `tests/unit/api/flow/test_monadic.py`

### Changes

Add helpers such as:

- `pipe(value, *steps)`;
- `tap(effect_fn)`;
- `branch(predicate, left, right)`;
- `unwrap_or_raise(either, error_mapper=None)`;
- `to_either(value, predicate, error)`;
- optional `named_step(name, fn)` for debug traces.

### Tests

- `tap` returns the original value after running the effect.
- branch helpers choose the expected side deterministically.
- error unwrapping preserves useful messages.
- helper composition does not mutate inputs unless the named effect does it explicitly.

### Acceptance Criteria

- Helpers have no dependency on `FedotIndustrial`.
- New code can express monadic flows without anonymous side-effect lambdas.

## PR 3. Typed Domain Objects For Industrial Runtime

### Goal

Replace repeated raw strings, booleans, and loose dictionaries with explicit records for runtime planning.

### Suggested Location

- `fedot_ind/api/flow/domain.py`
- `tests/unit/api/flow/test_domain.py`

### Candidate Objects

- `IndustrialTaskKind`;
- `InitialAssumptionSource`;
- `InitialAssumptionPlan`;
- `SolverInitPlan`;
- `PredictionModePlan`;
- `ProcessedInputBundle`;
- structured validation/error records for expected invalid states.

### Tests

- task strings normalize into stable enum values;
- invalid task names produce a stable domain error;
- plan records serialize or represent themselves clearly for diagnostics;
- default values are deterministic.

### Acceptance Criteria

- Runtime decisions no longer depend on scattered raw strings where a closed set is known.
- Tests can assert on typed plans without starting Dask or FEDOT.

## PR 4. Initial Assumption Planner

### Goal

Extract and harden initial assumption normalization. This is especially important for kernel warm-start,
where `PipelineBuilder` should be built only after the industrial repository is active.

### Suggested Location

- `fedot_ind/api/flow/initial_assumption.py`
- or `fedot_ind/api/services/initial_assumption.py`

### Changes

Extract:

- current `_normalize_initial_assumption`;
- source selection between `automl_config.initial_assumption` and `industrial_config.initial_assumption`;
- lazy build rules for `PipelineBuilder`;
- ordered handling of `list` and `tuple` assumptions.

### Tests

- `None` stays `None`;
- ready `Pipeline` objects are not rebuilt;
- `PipelineBuilder` is built only during normalization;
- sequence order is preserved;
- industrial config is used as fallback when automl config has no initial assumption.

### Acceptance Criteria

- `FedotIndustrial` no longer owns raw assumption normalization logic.
- Kernel warm-start initial assumptions remain lazy until repository activation.

## PR 5. Input Processing Service

### Goal

Move input conversion, `DataCheck`, target encoder extraction, and default-context shape adaptation out of the facade.

### Suggested Location

- `fedot_ind/api/services/input_processing.py`
- `tests/unit/api/services/test_input_processing.py`

### Changes

Introduce `IndustrialInputProcessor` with a method such as:

```python
process(input_data, task, task_params, fit_stage, industrial_task_params, default_fedot_context) -> ProcessedInputBundle
```

Internal steps may still use monadic composition, but each step should be named:

- `copy_input`;
- `build_data_check`;
- `run_data_check`;
- `extract_target_encoder`;
- `adapt_default_context_shape`.

### Tests

- target encoder is preserved;
- default FEDOT context still squeezes features as before;
- fit-stage flags reach `DataCheck`;
- input data is copied before validation.

### Acceptance Criteria

- `FedotIndustrial._process_input_data` becomes a thin wrapper or disappears behind the service.
- Behavior is unchanged at the public boundary.

## PR 6. Repository And Dask Runtime Services

### Goal

Separate repository activation and Dask startup from solver construction.

### Suggested Locations

- `fedot_ind/api/services/repository.py`
- `fedot_ind/api/services/dask_runtime.py`

### Changes

Add services such as:

- `IndustrialRepositoryInitializer.activate(context, backend)`;
- `DaskRuntimeInitializer.start(distributed_config)`.

Repository initialization should explicitly happen before initial assumption normalization that builds lazy pipeline
builders.

### Tests

- default FEDOT context activates default repository;
- industrial context activates industrial repository;
- optimizer strategy is selected as before;
- Dask runtime object returns client and cluster for manager assignment.

### Acceptance Criteria

- Repository and Dask side effects are visible named steps.
- Solver initialization no longer hides repository activation order.

## PR 7. Solver Factory Service

### Goal

Move `Fedot(...)` construction into a dedicated factory fed by a typed plan.

### Suggested Location

- `fedot_ind/api/services/solver_factory.py`
- `tests/unit/api/services/test_solver_factory.py`

### Changes

Add a `SolverFactory` that accepts a `SolverInitPlan` and creates the FEDOT solver.

The plan should include:

- learning strategy params;
- optimisation loss;
- task name;
- task params;
- optimizer;
- available operations;
- normalized initial assumption.

### Tests

- classification/regression task params are passed as before;
- forecasting task params use industrial forecasting params;
- available operations are preserved;
- optimizer partial for industrial optimization is preserved;
- normalized initial assumptions are passed into `Fedot`.

### Acceptance Criteria

- `FedotIndustrial.__init_solver` no longer manually assembles the `Fedot(...)` call from nested dictionaries.
- Solver creation is testable without running full `fit`.

## PR 8. Refactor `FedotIndustrial.fit` Into Explicit Orchestrator

### Goal

Make `fit` read as a lifecycle script, not a monadic chain around side effects.

### Target Shape

```python
def fit(self, input_data, **kwargs):
    with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
        input_bundle = self.input_processor.process(...)
        self.repository_initializer.activate(...)
        self.dask_runtime.start(...)
        self.solver = self.solver_factory.create(...)
        self.fit_service.fit(self.manager, input_bundle.data)
```

### Tests

- public `fit` path still wires collaborators in the expected order;
- industrial repository is active before lazy initial assumptions are built;
- kernel warm-start regression scenario is covered;
- fallback strategy behavior is preserved.

### Acceptance Criteria

- `FedotIndustrial.fit` is short and explicit.
- Pymonad is not used to hide runtime side effects.
- Public API remains unchanged.

## PR 9. Prediction Service

### Goal

Extract prediction routing and target decoding from `FedotIndustrial`.

### Suggested Location

- `fedot_ind/api/services/prediction.py`
- `tests/unit/api/services/test_prediction.py`

### Changes

Move logic from:

- `__abstract_predict`;
- `predict`;
- `predict_proba`.

The service should explicitly model:

- solver family: FEDOT, Pipeline, or custom strategy;
- output mode: labels, probabilities, default;
- target encoder inverse transform;
- forecasting horizon slicing.

### Tests

- labels/probabilities choose the expected solver method;
- Pipeline solver path remains supported;
- custom industrial strategy solver path remains supported;
- encoder inverse transform is stable;
- forecasting prediction is sliced to forecast horizon.

### Acceptance Criteria

- Prediction branching is named and testable.
- Mutations to prediction data target are explicit or removed if safe.

## PR 10. Metrics And Finetune Cleanup

### Goal

Move metrics evaluation and finetune flow into dedicated services.

### Suggested Locations

- `fedot_ind/api/services/metrics.py`
- `fedot_ind/api/services/finetune.py`

### Changes

Extract:

- `_metric_evaluation_loop`;
- `finetune` model-to-tune assembly;
- finetune tuning data preparation.

Introduce records such as:

- `MetricEvaluationRequest`;
- `MetricEvaluationResult`;
- `FinetuneRequest`.

### Tests

- encoded target path is preserved;
- dict predictions for ensembles still produce per-model metrics;
- classification/regression/forecasting metrics behave as before;
- `return_only_fitted` behavior is unchanged.

### Acceptance Criteria

- Metrics and finetune code no longer rely on large inline branching inside `FedotIndustrial`.
- Expected inputs and outputs are explicit.

## PR 11. Style Cleanup And Migration Guardrails

### Goal

Finish migration away from mixed anonymous Pymonad chains in the API layer.

### Changes

- Replace remaining anonymous `.then(lambda ...)` chains in `FedotIndustrial` and nearby API modules with named helpers
  or explicit shell steps.
- Add a short checklist to the developer guide.
- Optionally add Codex skill guidance for this repository:
    - when to use Pymonad;
    - when to extract a service;
    - when to introduce a typed plan.

### Tests

- no new behavior tests unless code changes require them;
- keep facade smoke tests green.

### Acceptance Criteria

- `FedotIndustrial` reads as an orchestrator.
- Monadic helpers are used only in small transformation or decision flows.
- Future code has a documented pattern to follow.

## Recommended Execution Order

1. PR 1-3: establish rules, helper layer, and typed domain vocabulary.
2. PR 4: extract initial assumption planning because it is already relevant to kernel warm-start.
3. PR 5-7: extract input processing, repository/Dask runtime, and solver factory.
4. PR 8: rewrite `FedotIndustrial.fit` over the extracted services.
5. PR 9-10: extract prediction, metrics, and finetune.
6. PR 11: cleanup remaining style inconsistencies and document guardrails.

## Final Target

Pymonad remains part of the project style, but it becomes a disciplined layer for explicit value flows:

```text
raw input -> validated input -> normalized plan -> explicit result
```

`FedotIndustrial` becomes the place where runtime effects are visible:

```text
prepare data
activate repository
start runtime
build solver
fit
predict
evaluate
```

This preserves the mathematical feel of functional composition while making debugging, testing, and future
kernel-learning integration safer.
