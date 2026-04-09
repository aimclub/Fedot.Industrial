# models/our_models.py
import numpy as np

from fedot_ind.core.models.kernel.okhs_forecasting_torch import OKHSForecasterTorch
from ..core.base_benchmark import BaseForecaster


class OurOKHSForecaster(BaseForecaster):
    """Наша модель OKHS для benchmarking"""

    def __init__(self, q=0.7, forecast_horizon=10, method='dmd', **kwargs):
        self.q = q
        self.forecast_horizon = forecast_horizon
        self.method = method
        self.kwargs = kwargs
        self.model = None

    def fit(self, time_series: np.ndarray, **kwargs):
        self.model = OKHSForecasterTorch(
            q=self.q,
            forecast_horizon=self.forecast_horizon,
            method=self.method,
            **self.kwargs
        )
        self.model.fit(time_series)

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(horizon=horizon)

    def get_model_info(self) -> dict:
        return {
            'name': 'OurOKHS',
            'q': self.q,
            'method': self.method,
            'forecast_horizon': self.forecast_horizon,
            **self.kwargs
        }
