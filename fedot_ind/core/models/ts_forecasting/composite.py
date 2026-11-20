"""Opt-in terminal forecasts whose internal operations are not search nodes."""

import numpy as np

from industrial_forecasting.dwt import predict_arino
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
        self.horizon_ = data.task.task_params.forecast_length
        self.model_ = self._fit(_history(data))
        return self

    def predict(self, data):
        if self.model_ is None:
            raise ValueError("Fit the composite forecast before prediction.")
        if data.task.task_params.forecast_length != self.horizon_:
            raise ValueError("The forecast horizon does not match the fitted operation.")
        return self._forecast(_history(data))

    def get_diagnostics(self):
        return self.template.describe(self.params)


class WaveletHFCMImplementation(_CompositeForecast):
    template = HFCM

    def _fit(self, history):
        return fit_wavelet_hfcm(history, self.spec)

    def _forecast(self, _history):
        return self.model_.predict(self.horizon_)


class ArinoImplementation(_CompositeForecast):
    template = ARINO

    def _fit(self, history):
        return predict_arino(history, self.horizon_, self.spec)

    def _forecast(self, _history):
        return self.model_.copy()


class WaveletAttentionImplementation(_CompositeForecast):
    template = CWT

    def _fit(self, history):
        from industrial_forecasting.neural.wavelet_attention import WaveletAttentionSpec, fit_wavelet_attention
        settings = self.spec
        spec = WaveletAttentionSpec(window=settings.window, horizon=self.horizon_,
                                    width=settings.width, attention=settings.attention,
                                    cnn_width=settings.cnn_width, lstm_width=settings.lstm_width)
        return fit_wavelet_attention((history,), spec, epochs=settings.epochs, lr=settings.learning_rate)

    def _forecast(self, history):
        return self.model_.predict(history)


class PageProjectionImplementation(_CompositeForecast):
    template = PAGE

    def _fit(self, history):
        from industrial_forecasting.neural.page_projection import fit_page_projection
        if self.horizon_ != 1:
            raise ValueError("The Page projection prototype forecasts one observation at a block end.")
        settings = self.spec
        return fit_page_projection(history, length=settings.length, rank=settings.rank,
                                   learn_projection=settings.learn_projection, baseline_lag=settings.baseline_lag,
                                   epochs=settings.epochs, learning_rate=settings.learning_rate)

    def _forecast(self, history):
        import torch
        known = history[-(self.spec.length - 1):]
        with torch.inference_mode():
            return self.model_(torch.tensor(known[None], dtype=torch.float32)).numpy()
