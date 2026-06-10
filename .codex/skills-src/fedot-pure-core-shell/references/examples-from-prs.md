# Examples From IndustrialTS Refactoring Slices

## Benchmark Adapter and Runtime Bridge

Signal:

- several internal configuration points must be edited to add or interpret a benchmark adapter, optional baseline, or manifest-driven runtime integration

Extraction target:

- typed adapter parameter resolution
- capability interpretation
- registry query logic

Shell to preserve:

- registration and runtime adapter entry points

Tests to expect:

- mirrored tests under `tests/unit/models`
- smoke coverage for the entry path
- focused tests for typed resolution rules

## Manifest and Preset Safety Changes

Signal:

- manifest or preset parsing relies on raw dicts, sentinel strings, or unchecked exceptions

Extraction target:

- parse, validate, normalize, and choose-default stages

Shell to preserve:

- manifest-loading entry points, preset builders, and registered-run entry surfaces

Tests to expect:

- mirrored tests under `tests/unit/models`
- success and failure cases

## Kernel, Decomposition, and Benchmark Rule Extractions

Signal:

- large methods mix orchestration with assumptions, presets, defaults, or preprocessing choices

Extraction target:

- planning rules
- parameter normalization
- filter and recommendation logic

Shell to preserve:

- public forecasters and benchmark facades
- decomposition-facing entry points
- preprocessing orchestrators

Tests to expect:

- service or facade tests in mirrored `tests/unit/core/...` or `tests/unit/models/...`
- unit tests for new helper modules
- invariants for normalization or routing when meaningful
