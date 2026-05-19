# FedotIndustrial Functional Flow Rules

This document is the implementation guide for the staged refactor described in
`docs/dev_guide/fedot_industrial_functional_refactor_plan.md`.

The goal is not to remove Pymonad from the API layer. The goal is to use it
where it makes data flow clearer, and to keep runtime effects visible.

## Allowed Patterns

- Use small pure functions for validation, normalization, routing and plan
  construction.
- Use `Either` for expected alternatives or domain failures.
- Use `Maybe` only for ordinary optional values.
- Use `pipe(...)` when a sequence of transformations is easier to read as a
  value flow.
- Use `tap(effect_fn)` only when the effect is explicitly named and belongs to
  the shell.
- Convert raw dictionaries into typed plans near the API boundary.
- Keep repository activation, Dask startup, solver construction, fitting and
  prediction as visible orchestration steps.

## Discouraged Patterns

- Do not hide mutation of `self`, `self.manager`, repositories or solvers in
  `.then(lambda ...)` chains.
- Do not use `Either` as a replacement for every `if`.
- Do not mix validation, runtime startup and model fitting in one anonymous
  monadic chain.
- Do not encode closed domain states as scattered strings when an enum or typed
  record is practical.
- Do not catch expected domain failures with broad exceptions when a structured
  result can describe them.

## Effect Boundaries

The facade may perform effects, but each effect should have a named boundary:

- `IndustrialRepositoryInitializer.activate(...)`
- `DaskRuntimeInitializer.start(...)`
- `SolverFactory.create(...)`
- `IndustrialInputProcessor.process(...)`
- `PredictionService.predict(...)`

Those services can call pure helpers internally. The public facade should read
as a lifecycle script rather than as a long monadic expression.

## Pymonad Helper Layer

New code in the API layer should prefer `fedot_ind.api.flow` helpers over raw
anonymous chains:

```python
from fedot_ind.api.flow import pipe, tap, unwrap_or_raise

result = pipe(
    raw_config,
    normalize_config,
    validate_plan,
    tap(logger.debug),
)
plan = unwrap_or_raise(result)
```

This keeps the functional style while making named steps easy to test and easy
to debug.

## Test Expectations

- Pure planning helpers need direct unit tests.
- Facade or service wrappers need boundary tests that preserve public behavior.
- Normalization and routing helpers should have invariant-style tests for
  deterministic ordering, idempotence and fallback behavior.
- Runtime services should be tested with fakes or monkeypatching before full
  integration tests are added.

## Migration Checklist

Before refactoring a `FedotIndustrial` method:

1. Name the effectful operations in the method.
2. Extract pure decisions into helpers or typed plans.
3. Keep side effects in services or explicit facade steps.
4. Replace anonymous `.then(lambda ...)` mutation with named helpers or a direct
   imperative line.
5. Add tests for the extracted helper before rewiring the facade.
