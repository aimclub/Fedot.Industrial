"""A scoped assertion that a declared merge used the explicit node interpreter."""

from contextvars import ContextVar

from fedot_ind.core.models.ts_forecasting.branching.contracts import ForecastBranchError, merge_kind


active_forecast_merge = ContextVar('industrial_forecast_merge', default=None)


def require_forecast_merge_context(declaration):
    kind = merge_kind(declaration.tags)
    if kind is not None and active_forecast_merge.get() is not kind:
        raise ForecastBranchError('Use ForecastPipelineAdapter: the ordinary FEDOT merger loses forecast predictions')
