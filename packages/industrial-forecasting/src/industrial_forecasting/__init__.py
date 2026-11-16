"""Forecasting contracts and NumPy models; tensor support is opt-in."""

from .contracts import Covariate, CovariateAvailability, ForecastScenario, ModelScope, PointForecast, QuantileForecast
from .grid import GridError, GridErrorCode, TimeGrid
from .metrics import ForecastMetric, ForecastScore, OwaStepScaling, pinball_loss, score_forecast

__all__ = [
    "Covariate", "CovariateAvailability", "ForecastScenario", "GridError", "GridErrorCode",
    "ForecastMetric", "ForecastScore", "ModelScope", "OwaStepScaling", "PointForecast",
    "QuantileForecast", "TimeGrid", "pinball_loss", "score_forecast",
]
