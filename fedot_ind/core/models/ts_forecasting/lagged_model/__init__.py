"""Public index for lag-based forecasting models.

Implementations are imported on demand so one optional forecasting family does
not become a runtime dependency of every other family.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    'LaggedRidgeForecaster': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.lagged_ridge_forecaster',
        'LaggedRidgeForecaster',
    ),
    'LaggedRidgeForecasterImplementation': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.lagged_ridge_forecaster',
        'LaggedRidgeForecasterImplementation',
    ),
    'LowRankLaggedRidgeForecaster': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.low_rank_lagged_ridge_forecaster',
        'LowRankLaggedRidgeForecaster',
    ),
    'LowRankLaggedRidgeForecasterImplementation': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.low_rank_lagged_ridge_forecaster',
        'LowRankLaggedRidgeForecasterImplementation',
    ),
    'MSSAForecaster': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.mssa_forecaster',
        'MSSAForecaster',
    ),
    'MSSAForecasterImplementation': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.mssa_forecaster',
        'MSSAForecasterImplementation',
    ),
    'SSAForecasterImplementation': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.ssa_forecaster',
        'SSAForecasterImplementation',
    ),
    'TopologicalAR': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.topo_forecaster',
        'TopologicalAR',
    ),
    'TopologicalRidgeForecaster': (
        'fedot_ind.core.models.ts_forecasting.lagged_model.topo_forecaster',
        'TopologicalRidgeForecaster',
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
