# Test Placement Map

## Mirror the Source Tree

Prefer tests under directories that mirror the IndustrialTS source tree:

- `fedot_ind/core/models/...` -> `tests/unit/core/models/...`
- `fedot_ind/core/operation/...` -> `tests/unit/core/operation/...`
- `benchmark/v2/...` -> `tests/unit/models/...`
- `examples/...` helpers -> `tests/unit/examples/...`

Keep this mirrored layout even when a refactor introduces new helper modules.

## Pick the Test Layer

### Boundary test

Place a boundary test when the code under change is:

- a facade
- a builder
- a service coordinator
- an adapter
- a runtime bridge
- a public repository entry point

Boundary tests should exercise the public path and verify visible behavior.

### Pure helper test

Place a helper test when the code under change is:

- a planner
- a selector
- a parser
- a validator
- a normalizer
- a filter

Helper tests should focus on explicit inputs and outputs.

### Mixed change

When a refactor extracts pure logic from a public boundary, add both:

- a shell-level regression or smoke test
- pure-helper tests for the new extracted logic

## Marker Guidance

Express test type through pytest markers instead of inventing directory conventions. Keep markers aligned with the repository's existing pytest practices.
