from __future__ import annotations

import time
import importlib
import warnings
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from benchmark.v2.core import RunStatus, ModelFamily, ForecastingDatasetRecord
from benchmark.v2.adapters.automl_common import (
    AutoMLBudget,
    AutoMLResourceReport,
    AutoMLDataFormat,
    CommonAutoMLAdapter,
    ModelExecutionError,
)

@dataclass
class FLAMLAdapter(CommonAutoMLAdapter):

    backend_name: str = "flaml"
    name: str = "FLAML"
    data_format: AutoMLDataFormat = AutoMLDataFormat.UNIVARIATE

    lag_window: int = 10
    use_scaling: bool = False
    add_rolling_features: bool = False

    max_iter: Optional[int] = None
    model_list: Optional[List[str]] = None
    quantile_levels: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    random_seed: int = 42
    verbose: int = 0

    _scaler: Any = None
    _lag: int = 0
    _fitted_model: Any = None
    _resource_report: Optional[AutoMLResourceReport] = None
    _backend_imported: bool = False
    _backend_module: Any = None
    _early_stopped: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.budget is None:
            self.budget = AutoMLBudget(time_limit_sec=60, random_seed=self.random_seed)
        if self.max_iter is None and self.budget.trial_limit is not None:
            self.max_iter = self.budget.trial_limit
        self._quantile_supported = len(self.quantile_levels) > 0

    def _import_backend(self) -> Any:
        try:
            from flaml import AutoML
            return AutoML
        except ImportError as e:
            raise ImportError("FLAML is not installed. Install with: pip install flaml") from e

    def _prepare_data(self, dataset_record: Union[ForecastingDatasetRecord, List[ForecastingDatasetRecord]]) -> Dict[str, Any]:

        if isinstance(dataset_record, list):
            all_X = []
            all_y = []
            all_series_ids = []
            for rec in dataset_record:
                X, y, meta = self._build_features_from_record(rec)
                series_id_encoded = hash(rec.series_id) % 1000
                X = np.column_stack([X, np.full((X.shape[0], 1), series_id_encoded)])
                all_X.append(X)
                all_y.append(y)
                all_series_ids.extend([rec.series_id] * len(y))
            X = np.vstack(all_X)
            y = np.concatenate(all_y)
            lag = meta['lag']
            tail = meta['tail']
            tail = dataset_record[-1].train_values[-lag:]
            return {
                'X_train': X,
                'y_train': y,
                'meta': {'lag': lag, 'tail': tail, 'horizon': dataset_record[0].forecast_horizon, 'is_panel': True},
                'series_ids': all_series_ids,
            }
        else:
            X, y, meta = self._build_features_from_record(dataset_record)
            return {
                'X_train': X,
                'y_train': y,
                'meta': {**meta, 'is_panel': False},
            }

    def _build_features_from_record(self, record: ForecastingDatasetRecord) -> Tuple[np.ndarray, np.ndarray, Dict]:
        series = np.asarray(record.train_values, dtype=float)
        lag = min(self.lag_window, len(series) - 1)
        if lag < 1:
            lag = 1
        self._lag = lag

        X = []
        for i in range(lag, len(series)):
            row = list(series[i-lag:i])
            if self.add_rolling_features:
                window = series[i-lag:i]
                row.append(np.mean(window))
                row.append(np.std(window))
                row.append(np.max(window) - np.min(window))
            X.append(row)
        X = np.array(X)
        y = series[lag:]

        if len(X) == 0:
            raise RuntimeError("Not enough data to create lag features.")

        return X, y, {'lag': lag, 'tail': series[-lag:], 'horizon': record.forecast_horizon}

    def _fit(self, data: Dict[str, Any], budget: AutoMLBudget, kwargs: Dict[str, Any]) -> Any:
        AutoML = self._backend_module
        X_train = data['X_train']
        y_train = data['y_train']

        if data['meta'].get('is_panel', False):
            raise NotImplementedError("Panel data is not fully supported for recursive forecasting in FLAML adapter. Use univariate mode.")

        if self.use_scaling:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X_train = self._scaler.fit_transform(X_train)

        settings = {
            'task': 'regression',
            'time_budget': budget.time_limit_sec or 60,
            'metric': 'mae',
            'seed': budget.random_seed or self.random_seed,
            'verbose': self.verbose,
        }
        if self.max_iter is not None:
            settings['max_iter'] = self.max_iter
        if self.model_list is not None:
            settings['model_list'] = self.model_list

        automl = AutoML()
        automl.fit(X_train, y_train, **settings)

        self._early_stopped = False
        if hasattr(automl, 'best_iteration') and self.max_iter is not None:
            if automl.best_iteration < self.max_iter - 1:
                self._early_stopped = True

        self._fitted_model = automl
        return automl

    def _predict(self, model: Any, data: Dict[str, Any], forecast_horizon: int) -> np.ndarray:
        if data['meta'].get('is_panel', False):
            raise NotImplementedError("Panel prediction not supported in FLAML adapter.")
        lag = data['meta']['lag']
        tail = data['meta']['tail'].copy()
        preds = []

        for _ in range(forecast_horizon):
            row = list(tail[-lag:])
            if self.add_rolling_features:
                window = tail[-lag:]
                row.append(np.mean(window))
                row.append(np.std(window))
                row.append(np.max(window) - np.min(window))
            X_pred = np.array(row).reshape(1, -1)
            if self.use_scaling and self._scaler is not None:
                X_pred = self._scaler.transform(X_pred)
            y_pred = model.predict(X_pred)[0]
            preds.append(y_pred)
            tail = np.append(tail[1:], y_pred)

        return np.array(preds)

    def _predict_quantiles(self, model: Any, data: Dict[str, Any],
                           forecast_horizon: int, quantiles: List[float]) -> np.ndarray:
        if data['meta'].get('is_panel', False):
            raise NotImplementedError("Panel quantile prediction not supported.")
        try:
            lag = data['meta']['lag']
            tail = data['meta']['tail'].copy()
            quantile_preds = []

            for _ in range(forecast_horizon):
                row = list(tail[-lag:])
                if self.add_rolling_features:
                    window = tail[-lag:]
                    row.append(np.mean(window))
                    row.append(np.std(window))
                    row.append(np.max(window) - np.min(window))
                X_pred = np.array(row).reshape(1, -1)
                if self.use_scaling and self._scaler is not None:
                    X_pred = self._scaler.transform(X_pred)

                q_values = model.predict(X_pred, quantiles=quantiles)[0]
                quantile_preds.append(q_values)

                if 0.5 in quantiles:
                    mean_pred = q_values[quantiles.index(0.5)]
                else:
                    mean_pred = model.predict(X_pred)[0]
                tail = np.append(tail[1:], mean_pred)

            return np.array(quantile_preds)
        except Exception:
            point = self._predict(model, data, forecast_horizon)
            return np.column_stack([point] * len(quantiles))

    def forecast(
        self,
        dataset_record: Union[ForecastingDatasetRecord, List[ForecastingDatasetRecord]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self._backend_imported:
            status, msg = self.availability()
            if status != RunStatus.SUCCESS:
                raise ModelExecutionError(status, msg)

        start_time = time.time()
        start_cpu = time.process_time()

        try:
            data = self._prepare_data(dataset_record)

            if isinstance(dataset_record, list):
                horizon = max(rec.forecast_horizon for rec in dataset_record)
                kwargs = self._get_kwargs(dataset_record[0])
                kwargs['forecast_horizon'] = horizon
            else:
                horizon = dataset_record.forecast_horizon
                kwargs = self._get_kwargs(dataset_record)
            try:
                fitted_model = self._fit(data, self.budget, kwargs)
                self._fitted_model = fitted_model
            except TimeoutError as e:
                raise ModelExecutionError(RunStatus.TIMEOUT, f"Budget time limit exceeded: {e}")
            except MemoryError as e:
                raise ModelExecutionError(RunStatus.MEMORY_ERROR, f"Budget memory limit exceeded: {e}")
            forecast = self._predict(fitted_model, data, horizon)
            forecast = np.asarray(forecast, dtype=float).reshape(-1)
            quantiles = kwargs.get('quantiles', self.quantile_levels)
            quantile_forecasts = None
            if self.supports_quantiles():
                try:
                    quantile_forecasts = self._predict_quantiles(
                        fitted_model, data, horizon, quantiles
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
                f"FLAML forecast failed: {e}",
                budget_info=self._resource_report.to_dict() if self._resource_report else None
            )

    def _extract_report(self, model: Any, start_time: float, end_time: float,
                        kwargs: Dict[str, Any]) -> AutoMLResourceReport:
        wall_clock = end_time - start_time
        total_trials = 1
        if hasattr(model, 'best_iteration'):
            total_trials = model.best_iteration + 1
        elif hasattr(model, 'n_iter_'):
            total_trials = model.n_iter_

        memory_usage_mb = 0.0
        try:
            import psutil
            memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            pass

        return AutoMLResourceReport(
            wall_clock_sec=wall_clock,
            memory_usage_mb=memory_usage_mb,
            models_fitted=total_trials,
            quantile_support=self.supports_quantiles(),
            total_trials=total_trials,
            early_stopped=self._early_stopped,
        )

    def supports_quantiles(self) -> bool:
        return len(self.quantile_levels) > 0

    def _get_optional_dependencies(self) -> List[str]:
        return ['psutil', 'scikit-learn']

    def availability(self) -> Tuple[RunStatus, str]:
        try:
            self._backend_module = self._import_backend()
            self._backend_imported = True
            return RunStatus.SUCCESS, "FLAML available"
        except ImportError as e:
            return RunStatus.NOT_AVAILABLE, f"FLAML not installed: {e}"
        except Exception as e:
            return RunStatus.BACKEND_UNAVAILABLE, f"FLAML initialization failed: {e}"


def create_flaml_adapter(
    name: str = "FLAML",
    tags: Tuple[str, ...] = ('automl', 'forecasting', 'external', 'flaml'),
    **kwargs
) -> FLAMLAdapter:
    budget_kwargs = {k: kwargs.pop(k) for k in ['time_limit_sec', 'trial_limit', 'memory_limit_mb',
                                                'random_seed', 'early_stopping', 'max_models'] if k in kwargs}
    budget = AutoMLBudget(**budget_kwargs) if budget_kwargs else None

    adapter_kwargs = {
        'name': name,
        'tags': tags,
        'budget': budget,
        'lag_window': kwargs.pop('lag_window', 10),
        'use_scaling': kwargs.pop('use_scaling', False),
        'add_rolling_features': kwargs.pop('add_rolling_features', False),
        'max_iter': kwargs.pop('max_iter', None),
        'model_list': kwargs.pop('model_list', None),
        'quantile_levels': kwargs.pop('quantile_levels', [0.1, 0.5, 0.9]),
        'random_seed': kwargs.pop('random_seed', 42),
        'verbose': kwargs.pop('verbose', 0),
        'data_format': kwargs.pop('data_format', AutoMLDataFormat.UNIVARIATE),
    }
    return FLAMLAdapter(**adapter_kwargs)
