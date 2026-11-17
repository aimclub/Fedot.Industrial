"""Supply the Industrial parameter catalogue to standalone stage plans."""

from typing import Any

from industrial_forecasting.model_names import canonical_forecasting_model_name
from industrial_forecasting.tuning.plan import (
    FALLBACK_STAGE_SEARCH_SPACE_PARAMETERS,
    ForecastingStageName,
    ForecastingStageSearchSpace,
    ForecastingStageTuningPlan,
    StageTuningGroup,
    build_forecasting_stage_search_spaces as _build_search_spaces,
    build_forecasting_stage_tuning_plan,
)

__all__ = [
    'FALLBACK_STAGE_SEARCH_SPACE_PARAMETERS', 'ForecastingStageName', 'ForecastingStageSearchSpace',
    'ForecastingStageTuningPlan', 'StageTuningGroup',
    'build_forecasting_stage_search_spaces', 'build_forecasting_stage_tuning_plan',
]


def _operation_parameter_space(model_name: str) -> dict | None:
    try:
        from fedot_ind.core.tuning.search_space import industrial_search_space
    except ModuleNotFoundError:
        return None
    return industrial_search_space.get(canonical_forecasting_model_name(model_name), {})


def build_forecasting_stage_search_spaces(
        model_name: str, params: dict[str, Any] | None = None,
) -> tuple[ForecastingStageSearchSpace, ...]:
    return _build_search_spaces(model_name, params, parameter_space=_operation_parameter_space(model_name))
