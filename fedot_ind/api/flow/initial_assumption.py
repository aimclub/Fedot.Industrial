"""Initial-assumption planning and normalization helpers."""

from typing import Any, Mapping, Optional

from fedot_ind.api.flow.domain import InitialAssumptionPlan, plan_initial_assumption


def resolve_initial_assumption_plan(
        automl_config: Optional[Mapping[str, Any]] = None,
        industrial_config: Optional[Mapping[str, Any]] = None,
) -> InitialAssumptionPlan:
    """Resolve the initial-assumption source from API config dictionaries."""
    automl_initial = dict(automl_config or {}).get("initial_assumption")
    industrial_initial = dict(industrial_config or {}).get("initial_assumption")
    return plan_initial_assumption(
        automl_initial=automl_initial,
        industrial_initial=industrial_initial,
    )


def normalize_initial_assumption(initial_assumption: Any) -> Any:
    """Normalize initial assumptions before they are passed to FEDOT.

    ``PipelineBuilder``-like objects are materialized through their ``build``
    method.  Sequences are normalized recursively while preserving order.
    """
    if initial_assumption is None:
        return None
    if isinstance(initial_assumption, (list, tuple)):
        return [normalize_initial_assumption(item) for item in initial_assumption]
    build = getattr(initial_assumption, "build", None)
    if callable(build):
        return build()
    return initial_assumption


def normalize_initial_assumption_from_configs(
        automl_config: Optional[Mapping[str, Any]] = None,
        industrial_config: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Resolve and normalize the initial assumption from API configs."""
    plan = resolve_initial_assumption_plan(
        automl_config=automl_config,
        industrial_config=industrial_config,
    )
    return normalize_initial_assumption(plan.assumption)
