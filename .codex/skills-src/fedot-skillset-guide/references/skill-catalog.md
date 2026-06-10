# Skill Catalog

## `$fedot-pure-core-shell`

Use for:

- extracting planning, validation, filtering, and normalization rules from OOP shells
- preserving public facades while simplifying internal logic
- splitting shell effects from deterministic helpers

Typical packages:

- `fedot_ind/core/models`
- `fedot_ind/core/operation`
- `examples/rkhs_okhs`
- `docs/okhs_dmd`

## `$fedot-invariant-tests-review`

Use for:

- choosing the right test layer after refactors
- PR review against the FP checklist
- adding invariants for normalization, routing, batching, merging, and serialization

Typical packages:

- `tests/*`
- any mirrored refactor touching public boundaries and pure helpers

## `$fedot-safe-configs`

Use for:

- replacing unsafe parsing
- removing sentinel strings and hidden null semantics
- redesigning parse, validate, normalize, and default flows

Typical packages:

- `benchmark/v2/manifests.py`
- `benchmark/v2/presets.py`
- parser-heavy config or policy flows
- benchmark or model metadata parsing paths

## `$fedot-extension-contract`

Use for:

- benchmark adapter contract design
- manifest or preset materialization
- registry flow
- smoke testing optional backends or factories
- runtime adapter behavior downstream of validated configs

Typical packages:

- `benchmark/v2`
- optional baseline adapters
- runtime integration of external or heavyweight models

## `$fedot-typed-domain-errors`

Use for:

- replacing stringly typed orchestration and nullable state
- introducing named domain records, enums, and structured failures
- clarifying requests, results, plans, and error surfaces

Typical packages:

- `fedot_ind/core/models`
- `fedot_ind/core/operation`
- `benchmark/v2`
- adapter and policy layers

## `$fedot-refactor-router`

Use for:

- ambiguous tasks
- mixed refactors that span several concerns
- early planning before implementation

Typical packages:

- any combination of the above
