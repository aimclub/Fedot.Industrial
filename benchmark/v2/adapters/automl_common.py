from __future__ import annotations

import importlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from benchmark.v2.core import ForecastingModelAdapter, ModelFamily, RunStatus, ForecastingDatasetRecord

class ModelExecutionError(RuntimeError):
    def __init__(self, status: RunStatus, message: str,
                 skip_reason: str | None = None,
                 backend_error: str | None = None,
                 dependency_status: dict[str, str] | None = None,
                 api_status: dict[str, str] | None = None,
                 budget_info: dict[str, Any] | None = None):
        self.status = status
        self.message = message
        self.skip_reason = skip_reason
        self.backend_error = backend_error
        self.dependency_status = dependency_status
        self.api_status = api_status
        self.budget_info = budget_info
        super().__init__(message)


class AutoMLBudget:
    def __init__(self, time_limit_sec: Optional[float] = None,
                 trial_limit: Optional[int] = None,
                 memory_limit_mb: Optional[int] = None,
                 random_seed: Optional[int] = 42,
                 early_stopping: bool = True,
                 max_models: Optional[int] = None):
        self.time_limit_sec = time_limit_sec
        self.trial_limit = trial_limit
        self.memory_limit_mb = memory_limit_mb
        self.random_seed = random_seed
        self.early_stopping = early_stopping
        self.max_models = max_models

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutoMLBudget':
        return cls(**{k: v for k, v in data.items() if k in cls.__init__.__code__.co_varnames})


class AutoMLResourceReport:
    def __init__(self, wall_clock_sec: float,
                 cpu_time_sec: float = 0.0,
                 memory_usage_mb: float = 0.0,
                 models_fitted: int = 0,
                 failed_trials: int = 0,
                 best_trial: Optional[Dict[str, Any]] = None,
                 quantile_support: bool = False,
                 gpu_memory_usage_mb: float = 0.0,
                 total_trials: int = 0,
                 early_stopped: bool = False):
        self.wall_clock_sec = wall_clock_sec
        self.cpu_time_sec = cpu_time_sec
        self.memory_usage_mb = memory_usage_mb
        self.models_fitted = models_fitted
        self.failed_trials = failed_trials
        self.best_trial = best_trial
        self.quantile_support = quantile_support
        self.gpu_memory_usage_mb = gpu_memory_usage_mb
        self.total_trials = total_trials
        self.early_stopped = early_stopped

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class AutoMLDataFormat(Enum):
    UNIVARIATE = "univariate"
    PANEL = "panel"
    MULTIVARIATE = "multivariate"


@dataclass
class CommonAutoMLAdapter(ForecastingModelAdapter, ABC):
    backend_name: str = "automl_backend"
    data_format: AutoMLDataFormat = AutoMLDataFormat.UNIVARIATE
    budget: Optional[AutoMLBudget] = None

    name: str = "CommonAutoML"
    family: ModelFamily = ModelFamily.AUTOML
    tags: tuple[str, ...] = ('automl', 'forecasting', 'external')
    optional: bool = True

    _fitted_model: Any = None
    _resource_report: Optional[AutoMLResourceReport] = None
    _backend_imported: bool = False
    _backend_module: Any = None

    _quantile_supported: bool = False

    def __post_init__(self):
        if self.budget is None:
            self.budget = AutoMLBudget()

    @abstractmethod
    def _import_backend(self) -> Any:
        pass

    @abstractmethod
    def _prepare_data(self, dataset_record: ForecastingDatasetRecord) -> Any:
        pass

    @abstractmethod
    def _fit(self, data: Any, budget: AutoMLBudget, kwargs: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def _predict(self, model: Any, data: Any, forecast_horizon: int) -> np.ndarray:
        pass

    @abstractmethod
    def _predict_quantiles(self, model: Any, data: Any,
                           forecast_horizon: int,
                           quantiles: List[float]) -> np.ndarray:
        pass

    @abstractmethod
    def _extract_report(self, model: Any, start_time: float, end_time: float,
                        kwargs: Dict[str, Any]) -> AutoMLResourceReport:
        pass

    def availability(self) -> Tuple[RunStatus, str]:
        try:
            self._backend_module = self._import_backend()
            self._backend_imported = True
            if self.budget and self.budget.time_limit_sec is not None and self.budget.time_limit_sec <= 0:
                return RunStatus.NOT_AVAILABLE, "time_limit_sec must be positive"
            if self.budget and self.budget.trial_limit is not None and self.budget.trial_limit <= 0:
                return RunStatus.NOT_AVAILABLE, "trial_limit must be positive"
            return RunStatus.SUCCESS, "Backend available"
        except ImportError as e:
            return RunStatus.NOT_AVAILABLE, f"Backend {self.backend_name} not installed: {e}"
        except Exception as e:
            return RunStatus.BACKEND_UNAVAILABLE, f"Backend initialization failed: {e}"

    def forecast(self, dataset_record: ForecastingDatasetRecord) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self._backend_imported:
            status, msg = self.availability()
            if status != RunStatus.SUCCESS:
                raise ModelExecutionError(status, msg)

        start_time = time.time()
        start_cpu = time.process_time()

        try:
            data = self._prepare_data(dataset_record)
            kwargs = self._get_kwargs(dataset_record)

            try:
                fitted_model = self._fit(data, self.budget, kwargs)
                self._fitted_model = fitted_model
            except TimeoutError as e:
                raise ModelExecutionError(RunStatus.TIMEOUT, f"Budget time limit exceeded: {e}")
            except MemoryError as e:
                raise ModelExecutionError(RunStatus.MEMORY_ERROR, f"Budget memory limit exceeded: {e}")

            forecast = self._predict(fitted_model, data, dataset_record.forecast_horizon)
            forecast = np.asarray(forecast, dtype=float).reshape(-1)

            quantiles = kwargs.get('quantiles', [0.1, 0.5, 0.9])
            quantile_forecasts = None
            if self.supports_quantiles():
                try:
                    quantile_forecasts = self._predict_quantiles(
                        fitted_model, data, dataset_record.forecast_horizon, quantiles
                    )
                except NotImplementedError:
                    pass

            end_time = time.time()
            end_cpu = time.process_time()

            report = self._extract_report(fitted_model, start_time, end_time, kwargs)
            report.cpu_time_sec = end_cpu - start_cpu
            self._resource_report = report

            metadata = {
                'backend': self.backend_name,
                'data_format': self.data_format.value,
                'budget': self.budget.to_dict() if self.budget else None,
                'resource_report': report.to_dict(),
                'models_fitted': report.models_fitted,
                'failed_trials': report.failed_trials,
                'quantile_support': report.quantile_support,
                'forecast_length': len(forecast),
                'kwargs': kwargs,
            }
            if quantile_forecasts is not None:
                metadata['quantiles'] = {
                    f'q{q:.2f}': quantile_forecasts[:, i].tolist()
                    for i, q in enumerate(quantiles) if i < quantile_forecasts.shape[1]
                }
            if len(forecast) > 1:
                metadata['prediction_range'] = float(forecast.max() - forecast.min())
                metadata['prediction_std'] = float(forecast.std())

            return forecast, metadata

        except Exception as e:
            end_time = time.time()
            if self._resource_report is None:
                self._resource_report = AutoMLResourceReport(
                    wall_clock_sec=end_time - start_time,
                    models_fitted=0,
                    failed_trials=1,
                )
            raise ModelExecutionError(
                RunStatus.FAILED,
                f"AutoML forecast failed: {e}",
                budget_info=self._resource_report.to_dict() if self._resource_report else None
            )

    def supports_quantiles(self) -> bool:
        return self._quantile_supported

    def get_dependency_status(self) -> Dict[str, str]:
        status = {}
        try:
            self._import_backend()
            status[self.backend_name] = 'present'
            status['import_success'] = 'true'
        except ImportError:
            status[self.backend_name] = 'missing'
            status['import_success'] = 'false'
        except Exception as e:
            status[self.backend_name] = 'error'
            status['import_error'] = str(e)

        for dep in self._get_optional_dependencies():
            try:
                importlib.import_module(dep)
                status[dep] = 'present'
            except ImportError:
                status[dep] = 'missing'
        return status

    def _get_optional_dependencies(self) -> List[str]:
        return []

    def _get_kwargs(self, dataset_record):
        if isinstance(dataset_record, list):
            rec = dataset_record[0]
        else:
            rec = dataset_record
        return {
            'forecast_horizon': rec.forecast_horizon,
            'seasonal_period': rec.seasonal_period,
            'frequency': rec.frequency,
            'series_id': rec.series_id,
            'static_covariates': rec.static_covariates,
            'known_future_covariates': rec.known_future_covariates,
            'observed_past_covariates': rec.observed_past_covariates,
            'panel_ids': rec.panel_ids,
            'hierarchy': rec.hierarchy,
            'data_format': self.data_format.value if hasattr(self.data_format, 'value') else self.data_format,
        }
