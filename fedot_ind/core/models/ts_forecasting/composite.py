"""Opt-in terminal forecasts whose internal operations are not search nodes."""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from industrial_forecasting.dwt import ArinoSpec, predict_arino
from industrial_forecasting.wavelet_hfcm import WaveletHFCMSpec, fit_wavelet_hfcm
from industrial_forecasting.wavelets import _finite_series


@dataclass(frozen=True)
class ForecastTemplate:
    operation: str
    version: int
    edges: tuple[tuple[str, str], ...]
    parameters: tuple[str, ...]

    def describe(self, config):
        config = self.flatten(config)
        return {"template": self.operation, "version": self.version, "edges": self.edges,
                "parameters": {key: config[key] for key in self.parameters if key in config}}

    @property
    def roles(self):
        return tuple(sorted({key.split(".")[0] for key in self.parameters}))

    def flatten(self, params):
        result = {}
        for key, value in dict(params or {}).items():
            if key in self.roles:
                if not isinstance(value, Mapping):
                    raise ValueError(f"Composite role {key!r} requires a parameter mapping.")
                entries = ((f"{key}.{leaf}", item) for leaf, item in value.items())
            else:
                entries = ((key, value),)
            for name, item in entries:
                if name in result:
                    raise ValueError(f"Repeated composite parameter: {name}")
                result[name] = item
        return result


HFCM = ForecastTemplate(
    "wavelet_hfcm_forecaster", 1,
    (("history", "haar"), ("haar", "lags"), ("lags", "hfcm"), ("hfcm", "sum")),
    ("haar.levels", "hfcm.order", "hfcm.alpha"),
)
ARINO = ForecastTemplate(
    "arino_forecaster", 1,
    (("history", "pad"), ("pad", "dwt"), ("dwt", "low"), ("dwt", "high"),
     ("low", "trend"), ("high", "seasonal"), ("trend", "sum"), ("seasonal", "sum")),
    ("pad.length", "dwt.wavelet", "dwt.levels", "dwt.slow_bands", "trend.lookback", "seasonal.period"),
)
TEMPLATES = {item.operation: item for item in (ARINO, HFCM)}


def template_spec(operation, params):
    """Validate the configured roles before starting any fit."""
    template = TEMPLATES[operation]
    config = template.flatten(params)
    unknown = config.keys() - set(template.parameters) - {"forecast_length"}
    if unknown:
        raise ValueError(f"Unknown composite parameters: {sorted(unknown)}")
    if operation == HFCM.operation:
        return WaveletHFCMSpec(config.get("haar.levels", 2), config.get("hfcm.order", 3),
                               config.get("hfcm.alpha", 0.1))
    return ArinoSpec(wavelet=config.get("dwt.wavelet", "db3"), levels=config.get("dwt.levels", 3),
                     slow_bands=config.get("dwt.slow_bands", 1), padding=config.get("pad.length", 0),
                     seasonal_period=config.get("seasonal.period", 12),
                     trend_lookback=config.get("trend.lookback"))


class _CompositeForecast:
    template: ForecastTemplate

    def __init__(self, params=None):
        self.params = dict(params or {})
        self.spec = template_spec(self.template.operation, self.params)
        self.model_ = None

    def fit(self, data):
        self.horizon_ = data.task.task_params.forecast_length
        features = np.asarray(data.features)
        if features.ndim == 2 and features.shape[0] == 1:
            features = features[0]
        self.model_ = self._fit(_finite_series(features))
        return self

    def predict(self, data):
        if self.model_ is None:
            raise ValueError("Fit the composite forecast before prediction.")
        if data.task.task_params.forecast_length != self.horizon_:
            raise ValueError("The forecast horizon does not match the fitted operation.")
        return self._forecast(self.model_, self.horizon_)

    def get_diagnostics(self):
        config = ({'haar.levels': self.spec.levels, 'hfcm.order': self.spec.order, 'hfcm.alpha': self.spec.alpha}
                  if self.template is HFCM else
                  {'pad.length': self.spec.padding, 'dwt.wavelet': self.spec.wavelet,
                   'dwt.levels': self.spec.levels, 'dwt.slow_bands': self.spec.slow_bands,
                   'trend.lookback': self.spec.trend_lookback, 'seasonal.period': self.spec.seasonal_period})
        return self.template.describe(config)


class WaveletHFCMImplementation(_CompositeForecast):
    template = HFCM

    def _fit(self, history):
        return fit_wavelet_hfcm(history, self.spec)

    @staticmethod
    def _forecast(model, horizon):
        return model.predict(horizon)


class ArinoImplementation(_CompositeForecast):
    template = ARINO

    def _fit(self, history):
        return predict_arino(history, self.horizon_, self.spec)

    @staticmethod
    def _forecast(model, horizon):
        return model.copy()
