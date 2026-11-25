"""Native model factories and a typed entrypoint for local forecasts."""

import inspect
from typing import Any, Protocol

import numpy as np

from ..contracts import ForecastScenario, ModelScope, PointForecast
from ..model_names import canonical_forecasting_model_name
from .havok import HAVOKForecaster
from .hybrid import HybridEnsembleForecaster
from .lagged import LaggedRidgeForecaster
from .low_rank import LowRankLaggedRidgeForecaster
from .mssa import MSSAForecaster


class SeriesForecaster(Protocol):
    def fit(self, time_series: np.ndarray): ...

    def predict(self, time_series: np.ndarray, *, forecast_horizon: int) -> np.ndarray: ...


class SeriesModelFactory(Protocol):
    def __call__(
        self, model_name: str, *, forecast_horizon: int, params: dict[str, Any], series_length: int | None,
    ) -> SeriesForecaster: ...


_MODELS = {
    "lagged_ridge_forecaster": LaggedRidgeForecaster,
    "low_rank_lagged_ridge_forecaster": LowRankLaggedRidgeForecaster,
    "mssa_forecaster": MSSAForecaster,
    "ssa_forecaster": MSSAForecaster,
    "havok_forecaster": HAVOKForecaster,
    "hybrid_ensemble_forecaster": HybridEnsembleForecaster,
    "lagged_forecaster": LaggedRidgeForecaster,
}


def instantiate_forecaster(model_class, *, forecast_horizon: int, params: dict[str, Any]) -> SeriesForecaster:
    accepted = inspect.signature(model_class).parameters
    return model_class(
        forecast_horizon=int(forecast_horizon),
        **{key: value for key, value in params.items() if key in accepted and key != "forecast_horizon"},
    )


def build_series_model(
    model_name: str, *, forecast_horizon: int, params: dict[str, Any], series_length: int | None = None,
) -> SeriesForecaster:
    """Build an unfitted native model; unsupported families need a caller-supplied factory."""
    name = canonical_forecasting_model_name(model_name)
    if name not in _MODELS:
        raise ValueError(f"Unsupported runtime stage tuning model: {name}")
    if name == "lagged_forecaster" and str(params.get("channel_model", "ridge")).lower() != "ridge":
        raise ValueError("Runtime stage tuning supports 'lagged_forecaster' only with channel_model='ridge'.")
    effective_params = {**params, "coupled": False} if name == "ssa_forecaster" else params
    return instantiate_forecaster(_MODELS[name], forecast_horizon=forecast_horizon, params=effective_params)


def forecast_series(
    scenario: ForecastScenario, model_name: str, *, params: dict[str, Any] | None = None,
) -> PointForecast:
    """Fit one model per series on its history and retain the supplied forecast coordinates."""
    if scenario.scope is not ModelScope.LOCAL or scenario.covariates:
        raise ValueError("Native series models support local scenarios without covariates.")
    forecasts = []
    for row in scenario.history:
        history = row.copy()
        model = build_series_model(
            model_name, forecast_horizon=scenario.horizon, params=dict(params or {}), series_length=len(history),
        )
        model.fit(history)
        forecasts.append(model.predict(history, forecast_horizon=scenario.horizon))
    return PointForecast.for_scenario(scenario, np.stack(forecasts))
