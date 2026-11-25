"""Industrial model factories and compatibility entrypoints for stage tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from industrial_forecasting.neural.models import SeriesModelFactory, build_series_model, instantiate_forecaster
from industrial_forecasting.neural.runtime import ForecastingSplitSpec
from industrial_forecasting.progress import ForecastingProgressPolicy
from industrial_forecasting.tuning.series import (
    ForecastMetricEvaluator, ForecastingSeriesEvaluation,
    ForecastingSeriesEvaluator as _ForecastingSeriesEvaluator,
    ForecastingSeriesStageTuningResult,
    ForecastingSeriesStageTuningRunner as _ForecastingSeriesStageTuningRunner,
)

from .stage_tuning import _operation_parameter_space

try:
    from fedot_ind.core.models.ts_forecasting.neural_models.neural_forecast_head import build_neural_forecast_head
except Exception:  # pragma: no cover - lightweight envs may miss neural runtime deps
    build_neural_forecast_head = None

try:
    from fedot_ind.core.models.ts_forecasting.dmd_models.okhs_fdmd_forecaster import build_okhs_fdmd_forecaster
except Exception:  # pragma: no cover - lightweight envs may miss operator runtime deps
    build_okhs_fdmd_forecaster = None


__all__ = [
    'ForecastMetricEvaluator', 'ForecastingSeriesEvaluation', 'ForecastingSeriesEvaluator',
    'ForecastingSeriesStageTuningResult', 'ForecastingSeriesStageTuningRunner',
    'build_forecasting_stage_objective_from_series', 'evaluate_forecasting_model_on_series',
    'run_forecasting_stage_tuning_on_series',
]


def _instantiate_runtime_model(
        canonical_model_name: str, *, forecast_horizon: int, params: dict[str, Any],
        series_length: int | None = None,
):
    if canonical_model_name == 'topo_forecaster':
        from fedot_ind.core.models.ts_forecasting.lagged_model.topo_forecaster import TopologicalRidgeForecaster

        return instantiate_forecaster(TopologicalRidgeForecaster, forecast_horizon=forecast_horizon, params=params)
    if canonical_model_name == 'hybrid_ensemble_forecaster':
        from fedot_ind.core.models.ts_forecasting.ensemble_models.hybrid_ensemble_forecaster import HybridEnsembleForecaster

        return instantiate_forecaster(HybridEnsembleForecaster, forecast_horizon=forecast_horizon, params=params)
    if canonical_model_name in {'okhs', 'okhs_fdmd_forecaster'}:
        if build_okhs_fdmd_forecaster is None:
            raise ValueError('OKHS FDMD runtime is unavailable in the current environment.')
        return build_okhs_fdmd_forecaster(
            forecast_horizon=int(forecast_horizon), params=dict(params), series_length=series_length,
        )
    if canonical_model_name in {'patch_tst_model', 'tst_model', 'tcn_model', 'deepar_model', 'nbeats_model'}:
        if build_neural_forecast_head is None:
            raise ValueError('Neural forecast head runtime is unavailable in the current environment.')
        return build_neural_forecast_head(
            canonical_model_name, forecast_horizon=int(forecast_horizon), params=dict(params),
        )
    return build_series_model(
        canonical_model_name, forecast_horizon=forecast_horizon, params=params, series_length=series_length,
    )


@dataclass
class ForecastingSeriesEvaluator(_ForecastingSeriesEvaluator):
    model_factory: SeriesModelFactory = field(default=_instantiate_runtime_model, kw_only=True)


@dataclass
class ForecastingSeriesStageTuningRunner(_ForecastingSeriesStageTuningRunner):
    model_factory: SeriesModelFactory = field(default=_instantiate_runtime_model, kw_only=True)

    def __post_init__(self):
        if self.parameter_space is None:
            self.parameter_space = _operation_parameter_space(self.model_name)
        super().__post_init__()


def evaluate_forecasting_model_on_series(
        model_name: str,
        *,
        time_series: np.ndarray,
        forecast_horizon: int,
        params: dict[str, Any] | None = None,
        metric_name: str = 'rmse',
        split_spec: ForecastingSplitSpec | None = None,
        seasonal_period: int = 1,
        show_progress: bool | None = None,
        progress_policy: ForecastingProgressPolicy | dict[str, Any] | bool | None = None,
) -> ForecastingSeriesEvaluation:
    """Evaluate a forecasting model on a single series with temporal splits."""
    return ForecastingSeriesEvaluator(
        model_name,
        time_series=time_series,
        forecast_horizon=forecast_horizon,
        params=params,
        metric_name=metric_name,
        split_spec=split_spec,
        seasonal_period=seasonal_period,
        show_progress=show_progress,
        progress_policy=progress_policy,
    ).run()


def build_forecasting_stage_objective_from_series(
        model_name: str,
        *,
        time_series: np.ndarray,
        forecast_horizon: int,
        metric_name: str = 'rmse',
        split_spec: ForecastingSplitSpec | None = None,
        seasonal_period: int = 1,
        show_progress: bool | None = None,
        progress_policy: ForecastingProgressPolicy | dict[str, Any] | bool | None = None,
) -> Callable[[dict[str, Any]], float]:
    """Create a stage-tuning objective that scores candidates on a series."""
    runner = ForecastingSeriesStageTuningRunner(
        model_name,
        time_series=time_series,
        forecast_horizon=forecast_horizon,
        metric_name=metric_name,
        split_spec=split_spec,
        seasonal_period=seasonal_period,
        show_progress=show_progress,
        progress_policy=progress_policy,
    )
    return runner._build_objective()


def run_forecasting_stage_tuning_on_series(
        model_name: str,
        *,
        time_series: np.ndarray,
        forecast_horizon: int,
        base_params: dict[str, Any] | None = None,
        stage_updates: dict[str, Any] | None = None,
        metric_name: str = 'rmse',
        split_spec: ForecastingSplitSpec | None = None,
        seasonal_period: int = 1,
        max_values_per_parameter: int = 3,
        max_stage_candidates: int = 16,
        show_progress: bool | None = None,
        progress_policy: ForecastingProgressPolicy | dict[str, Any] | bool | None = None,
) -> ForecastingSeriesStageTuningResult:
    """Run full stage tuning for one model and one time series."""
    return ForecastingSeriesStageTuningRunner(
        model_name,
        time_series=time_series,
        forecast_horizon=forecast_horizon,
        base_params=base_params,
        stage_updates=stage_updates,
        metric_name=metric_name,
        split_spec=split_spec,
        seasonal_period=seasonal_period,
        max_values_per_parameter=max_values_per_parameter,
        max_stage_candidates=max_stage_candidates,
        show_progress=show_progress,
        progress_policy=progress_policy,
    ).run()
