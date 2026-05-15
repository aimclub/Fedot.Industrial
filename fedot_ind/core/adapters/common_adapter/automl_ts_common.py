from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import time

from fedot_ind.core.adapters.common_adapter.contracts import AutoMLBudget, AutoMLResult


class AutoMLForecastingAdapter(ABC):
    def __init__(self, **params):
        self.params = params
        self.model = None
        self._fitted_model_count = 0
        self._failure_count = 0

    def fit(self, data: pd.DataFrame, metadata: Dict,
            budget: AutoMLBudget, **kwargs) -> 'AutoMLForecastingAdapter':
        mode = metadata.get('mode', 'univariate')
        start_time = time.time()

        try:
            if mode == 'univariate':
                self._fit_univariate(data, budget, **kwargs)
            elif mode == 'panel':
                self._fit_panel(data, budget, **kwargs)
            elif mode == 'multivariate':
                self._fit_multivariate(data, budget, **kwargs)
            else:
                raise ValueError(f"{mode} is not supported")
        except Exception as e:
            self._failure_count += 1
            raise

        self._result = AutoMLResult(
            wall_clock=time.time() - start_time,
            fitted_model_count=self._fitted_model_count,
            failure_count=self._failure_count,
            quantile_support=self._supports_quantiles()
        )

        return self

    @abstractmethod
    def _fit_univariate(self, data: pd.DataFrame, budget: AutoMLBudget, **kwargs):
        pass

    @abstractmethod
    def _fit_panel(self, data: pd.DataFrame, budget: AutoMLBudget, **kwargs):
        pass

    @abstractmethod
    def _fit_multivariate(self, data: pd.DataFrame, budget: AutoMLBudget, **kwargs):
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict_quantiles(self, data: pd.DataFrame, horizon: int,
                          quantiles: List[float], **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def availability(self) -> Dict[str, bool]:
        pass

    @abstractmethod
    def resource_report(self) -> Dict:
        pass
    @property
    def result(self) -> Optional[AutoMLResult]:
        return self._result