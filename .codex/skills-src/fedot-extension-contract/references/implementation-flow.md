# Implementation Flow

## Adding a New Adapter, Baseline, or Optional Backend

Follow this path:

1. Define or extend a typed `ModelSpec`, preset, or manifest payload.
2. Declare supported task type, data assumptions, and runtime capabilities explicitly.
3. Provide a callable adapter or builder with a narrow signature.
4. Validate or materialize the config before registration or runtime use.
5. Smoke test the adapter path or mark it unavailable with a structured reason.
6. Let runtime lookup and parameter rules operate only from the validated spec.

## Refactoring Rules

### Keep deterministic rules pure

Prefer small deterministic helpers for:

- checking duplicate model ids or aliases
- checking capability presence
- checking adapter signature shape
- interpreting hyperparameter defaults
- mapping benchmark methods or task modes to adapter builders

### Keep effects at the boundary

Leave these at the shell:

- importing optional heavy dependencies
- mutating the registry
- instantiating real model objects
- touching benchmark artifacts or external runtime code

## Compatibility Guidance

When evolving a manifest, preset, or adapter contract:

- prefer additive fields over breaking replacements
- keep old authoring patterns working when possible
- add validation for newly required fields explicitly
- keep error codes stable enough for tests and callers to reason about them

## Failure Modeling Guidance

Prefer explicit contract errors for:

- invalid manifest or preset type
- empty adapter or model name
- duplicate model name
- invalid adapter builder
- empty task or data type capability
- smoke-test failure
- missing manifest, preset, or adapter mapping
