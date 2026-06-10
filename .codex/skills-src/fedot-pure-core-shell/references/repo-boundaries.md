# Repo Boundaries

## Purpose

Use this file when deciding what to preserve as the effectful shell and what to extract into a pure helper.

## Boundary Objects to Preserve

### `fedot_ind/core/models/kernel`

Preserve public forecasters and model-level entry points such as:

- `OKHSForecaster`
- `OKHSForecasterTorch`
- public kernel-facing estimators and thin wrappers
- compatibility bridges that preserve stable user-facing behavior

Treat these as shell objects that own lifecycle, compatibility, diagnostics exposure, and orchestration.

### `fedot_ind/core/operation/decomposition`

Preserve numerical boundary objects such as:

- `OKHSTransformer`
- `FractionalLiouvilleOperator`
- `FractionalDMD`
- DMD wrapper classes
- decomposition builders and runtime adapters

Move rule-heavy selection, validation, normalization, and planning behind these boundaries into smaller collaborators where possible.

### `fedot_ind/core/operation/transformation`

Preserve transform entry points and representation objects. Extract:

- trajectory sampling rules
- rank-selection heuristics
- policy normalization
- deterministic feature or representation planning

### `benchmark/v2`

Preserve benchmark runners, adapters, and publication-pack orchestration. Extract:

- manifest parsing
- metric aggregation rules
- split logic
- artifact planning
- adapter selection and validation

### `examples` and `docs`

Preserve example scripts and docs as shell-facing surfaces. Extract:

- data generation rules
- split and evaluation logic
- plotting decisions
- report rendering helpers

## What Belongs in the Pure Core

Prefer extraction when logic:

- derives a result from explicit inputs
- can be expressed as a deterministic function
- only needs data, not live dependencies
- is easy to validate independently
- represents business rules more than orchestration

Typical pure outputs:

- plans
- normalized parameter objects
- filtered candidate lists
- routing decisions
- capability sets
- typed validation results

## What Should Stay in the Shell

Keep these in the shell:

- file access
- network calls
- logging
- environment inspection
- runtime setup
- dependency construction
- compatibility glue
- command sequencing across effectful dependencies

## Python-Oriented Typing Guidance

Translate FP-informed patterns into Python carefully:

- Use `dataclass` for stable records.
- Use `Enum` for closed state sets.
- Use `Protocol` or narrow abstract base classes for dependency boundaries when helpful.
- Use `TypedDict` sparingly for raw external payloads.
- Avoid carrying `dict[str, Any]` across multiple layers when a shape is stable enough to name.
