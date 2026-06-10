# Candidate Hotspots

## `fedot_ind/core/models/kernel`

Good candidates include:

- method resolution
- q-policy and window-policy records
- trajectory preprocessing summaries
- prediction diagnostics and failure records

Examples already moving in the right direction:

- enums and policies in `fedot_ind/core/models/kernel/okhs_common.py`
- structured optimization info in `fedot_ind/core/models/kernel/okhs_forecasting.py`

## `fedot_ind/core/operation/transformation`

Good candidates include:

- trajectory sampling plans
- rank-regularization decisions
- representation policy choices
- normalization of kernel or decomposition settings

Use named decision objects when several related booleans, modes, or thresholds travel together.

## `fedot_ind/core/models/automl` and `benchmark/v2`

Good candidates include:

- benchmark configs
- dataset and model specs
- rolling forecast policies
- adapter execution results
- registry entry records

This area benefits from separating parse failures, absent values, and environment failures.

## Integration and adapter flows

Good candidates include:

- optional external baseline availability
- manifest or preset validation errors
- adapter capability summaries
- parameter resolution results

Prefer explicit dataclasses and enums instead of falling back to raw metadata handling.

## Review Clues

A hotspot likely needs this skill when:

- several functions pass the same raw dictionary onward
- branch conditions inspect many string keys
- `None` values accumulate and later require interpretation
- the code returns tuples whose meaning is hard to remember without reading the implementation
