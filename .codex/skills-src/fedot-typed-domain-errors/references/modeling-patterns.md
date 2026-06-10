# Modeling Patterns

## Prefer Small Named Records

Use a named record such as a `dataclass` when several values move together as one concept:

- run request
- planning result
- filter decision
- capability set
- validation result

Good signs:

- callers repeatedly pass the same group of fields together
- field names matter for readability
- tests need to assert on several related outputs

## Prefer Enums for Closed Sets

Use enums when the valid set is finite and meaningful:

- task mode
- execution mode
- backend type
- preprocessing orientation
- validation status

This is especially useful when the current code branches on strings.

## Prefer Narrow Typed Dictionaries Only at the Edge

Use `TypedDict` or raw dictionaries only when the payload is inherently dictionary-shaped because it comes from:

- JSON
- YAML
- user params
- repository metadata

Convert to named internal objects once the payload crosses the boundary.

## Prefer Value Objects Over Boolean Clusters

If a function needs several booleans that collectively describe one decision, replace them with one named result object. This often makes tests and branching much clearer.

## Current Positive Examples in the Repo

Patterns already appearing in IndustrialTS that are worth extending:

- policy and diagnostics records in `fedot_ind/core/models/kernel/okhs_common.py`
- benchmark config and result dataclasses in `benchmark/v2/core.py`
- manifest, preset, and registry records in `benchmark/v2/manifests.py`, `benchmark/v2/presets.py`, and `benchmark/v2/registry.py`
- typed preprocessing decisions and rules in OKHS trajectory or representation flows
