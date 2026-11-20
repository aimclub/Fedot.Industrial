"""Opt-in terminal forecasts whose internal operations are not search nodes."""

from dataclasses import dataclass

import numpy as np

from industrial_forecasting.dwt import predict_arino
from industrial_forecasting.grid import positive_count
from industrial_forecasting.wavelet_hfcm import fit_wavelet_hfcm
from industrial_forecasting.wavelets import _finite_series

from fedot_ind.core.models.composite_templates import ARINO, CWT, HFCM, PAGE, template_spec


def _history(data):
    features = np.asarray(data.features)
    return _finite_series(features[0] if features.ndim == 2 and features.shape[0] == 1 else features)


class _CompositeForecast:
    def __init__(self, params=None):
        self.params = dict(params or {})
        self.spec = template_spec(self.template.operation, self.params)
        self.model_ = None

    def fit(self, data):
        horizon = positive_count(data.task.task_params.forecast_length, "Task forecast horizon")
        self.template.forecast_horizon.resolve(self.params, horizon)
        model = self._fit(_history(data), horizon)
        self.model_, self.horizon_ = model, horizon
        return self

    def predict(self, data):
        if self.model_ is None:
            raise ValueError("Fit the composite forecast before prediction.")
        if positive_count(data.task.task_params.forecast_length, "Task forecast horizon") != self.horizon_:
            raise ValueError("The forecast horizon does not match the fitted operation.")
        return self._forecast(_history(data))

    def get_diagnostics(self):
        return self.template.describe(self.params)


@dataclass(frozen=True, eq=False)
class _OriginPrediction:
    history: np.ndarray
    values: np.ndarray


class _FixedOriginForecast(_CompositeForecast):
    def _fit(self, history, horizon):
        snapshot = history.copy()
        snapshot.setflags(write=False)
        return _OriginPrediction(snapshot, self._fit_predict(history, horizon))

    def _forecast(self, history):
        if not np.array_equal(self.model_.history, history):
            raise ValueError("This forecast is tied to the fitted history. Refit with the changed history.")
        return self.model_.values.copy()


class WaveletHFCMImplementation(_FixedOriginForecast):
    template = HFCM

    def _fit_predict(self, history, horizon):
        return fit_wavelet_hfcm(history, self.spec).predict(horizon)


class ArinoImplementation(_FixedOriginForecast):
    template = ARINO

    def _fit_predict(self, history, horizon):
        return predict_arino(history, horizon, self.spec)


class WaveletAttentionImplementation(_CompositeForecast):
    template = CWT

    def _fit(self, history, horizon):
        from industrial_forecasting.neural.wavelet_attention import WaveletAttentionSpec, fit_wavelet_attention
        settings = self.spec
        spec = WaveletAttentionSpec(window=settings.window, horizon=horizon,
                                    width=settings.width, attention=settings.attention,
                                    cnn_width=settings.cnn_width, lstm_width=settings.lstm_width)
        return fit_wavelet_attention((history,), spec, epochs=settings.epochs, lr=settings.learning_rate)

    def _forecast(self, history):
        return self.model_.predict(history)


class PageProjectionImplementation(_CompositeForecast):
    template = PAGE

    def _fit(self, history, _horizon):
        from industrial_forecasting.neural.page_projection import fit_page_projection
        settings = self.spec
        return fit_page_projection(history, length=settings.length, rank=settings.rank,
                                   learn_projection=settings.learn_projection, baseline_lag=settings.baseline_lag,
                                   epochs=settings.epochs, learning_rate=settings.learning_rate)

    def _forecast(self, history):
        import torch
        known = history[-(self.spec.length - 1):]
        with torch.inference_mode():
            return self.model_(torch.tensor(known[None], dtype=torch.float32)).numpy()
