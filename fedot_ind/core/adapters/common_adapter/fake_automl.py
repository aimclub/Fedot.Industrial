import time
import random
from typing import Dict, List, Optional
import pandas as pd

from fedot_ind.core.adapters.common_adapter.automl_ts_common import AutoMLForecastingAdapter
from fedot_ind.core.adapters.common_adapter.contracts import AutoMLBudget, AutoMLResult


class FakeAutoMLBackend(AutoMLForecastingAdapter):
    def __init__(self, **params):
        super().__init__(**params)
        self.supports_quantiles = params.get('supports_quantiles', True)
        self.fail_on_fit = params.get('fail_on_fit', False)
        self.fail_on_predict = params.get('fail_on_predict', False)
        self.last_observation = None
        self.fitted_mode = None

    def _supports_quantiles(self) -> bool:
        return self.supports_quantiles

    def _fit_univariate(self, data: pd.DataFrame, budget: AutoMLBudget, **kwargs):
        if self.fail_on_fit:
            raise RuntimeError("Forced fit failure for testing")

        time.sleep(0.01)

        if 'value' in data.columns:
            self.last_observation = data['value'].iloc[-1]
        elif data.shape[1] == 1:
            self.last_observation = data.iloc[-1, 0]
        else:
            self.last_observation = data.iloc[-1, 0]

        self.fitted_mode = 'univariate'
        self._fitted_model_count += 1


    def _fit_panel(self, data: pd.DataFrame, budget: AutoMLBudget, **kwargs):
        if self.fail_on_fit:
            self._failure_count += 1
            raise RuntimeError("Forced fit failure for testing")

        time.sleep(0.01)

        # For panel data, store last observation for each series
        if 'value' in data.columns:
            self.last_observation = data.groupby('series_id')['value'].last().to_dict()
        else:
            self.last_observation = data.iloc[-1, 0]

        self.fitted_mode = 'panel'
        self._fitted_model_count += 1


    def _fit_multivariate(self, data: pd.DataFrame, budget: AutoMLBudget, **kwargs):
        if self.fail_on_fit:
            self._failure_count += 1
            raise RuntimeError("Forced fit failure for testing")

        time.sleep(0.01)

        target_column = kwargs.get('target_column', data.columns[0])
        self.last_observation = data[target_column].iloc[-1]

        self.fitted_mode = 'multivariate'
        self._fitted_model_count += 1


    def predict(self, data: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
        if self.fail_on_predict:
            raise RuntimeError("Forced predict failure for testing")

        if self.last_observation is None:
            raise ValueError("Model not fitted yet")

        if isinstance(self.last_observation, dict):
            predictions = []
            for series_id, last_val in self.last_observation.items():
                for h in range(horizon):
                    predictions.append({
                        'series_id': series_id,
                        'horizon': h + 1,
                        'prediction': last_val
                    })
            return pd.DataFrame(predictions)
        else:
            predictions = pd.DataFrame({
                'horizon': range(1, horizon + 1),
                'prediction': [self.last_observation] * horizon
            })
            return predictions


    def predict_quantiles(self, data: pd.DataFrame, horizon: int,
                          quantiles: List[float], **kwargs) -> pd.DataFrame:
        if not self._supports_quantiles():
            raise NotImplementedError("This backend does not support quantile forecasting")

        if self.fail_on_predict:
            raise RuntimeError("Forced predict failure for testing")

        if self.last_observation is None:
            raise ValueError("Model not fitted yet")

        results = []
        for h in range(1, horizon + 1):
            for q in quantiles:
                results.append({
                    'horizon': h,
                    'quantile': q,
                    'prediction': self.last_observation
                })

        return pd.DataFrame(results)


    def availability(self) -> Dict[str, bool]:
        return {
            'univariate': True,
            'panel': True,
            'multivariate': True,
            'quantiles': self.supports_quantiles
        }


    def resource_report(self) -> Dict:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return {
            "memory_mb": process.memory_info().rss // (1024 * 1024),
            "cpu_cores": psutil.cpu_count()
        }