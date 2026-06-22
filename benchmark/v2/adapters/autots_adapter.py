from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from benchmark.v2.core import RunStatus, ModelFamily, ForecastingDatasetRecord
from benchmark.v2.adapters.automl_common import (
    AutoMLBudget,
    AutoMLResourceReport,
    AutoMLDataFormat,
    CommonAutoMLAdapter,
    ModelExecutionError,
)


@dataclass
class AutoTSAdapter(CommonAutoMLAdapter):

    backend_name: str = "autots"
    name: str = "AutoTS"
    data_format: AutoMLDataFormat = AutoMLDataFormat.UNIVARIATE

    forecast_length: Optional[int] = None
    frequency: Optional[str] = None
    prediction_interval: float = 0.9
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    model_list: Optional[List[str]] = None
    transformer_list: Optional[List[str]] = None
    models_mode: str = "default"
    max_generations: Optional[int] = None
    random_seed: int = 42
    verbose: int = 0

    _fitted_predictor: Any = None
    _resource_report: Optional[AutoMLResourceReport] = None
    _backend_imported: bool = False
    _backend_module: Any = None
    _early_stopped: bool = False

    def __post_init__(self):
        if isinstance(self.data_format, str):
            self.data_format = AutoMLDataFormat(self.data_format.lower())
        super().__post_init__()
        if self.budget is None:
            self.budget = AutoMLBudget(time_limit_sec=3600, random_seed=self.random_seed)
        if self.frequency is None:
            self.frequency = "D"
        self._quantile_supported = bool(self.quantiles)

    def _import_backend(self) -> Any:
        try:
            from autots import AutoTS
            return AutoTS
        except ImportError as e:
            raise ImportError("AutoTS not installed. Install with: pip install autots") from e

    def _prepare_data(self, dataset_record: Union[ForecastingDatasetRecord, List[ForecastingDatasetRecord]]) -> pd.DataFrame:
        if isinstance(dataset_record, list):
            dfs = []
            for rec in dataset_record:
                df = self._single_record_to_df(rec, include_series_id=True)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        else:
            return self._single_record_to_df(dataset_record, include_series_id=False)

    def _single_record_to_df(self, record: ForecastingDatasetRecord, include_series_id: bool = False) -> pd.DataFrame:
        train = np.asarray(record.train_values, dtype=float)
        freq = self.frequency or record.frequency
        start_date = pd.Timestamp('2000-01-01')
        timestamps = pd.date_range(start=start_date, periods=len(train), freq=freq)
        df = pd.DataFrame({'datetime': timestamps, 'value': train})
        if include_series_id:
            df['series_id'] = record.series_id
        return df

    def _fit(self, data: pd.DataFrame, budget: AutoMLBudget, kwargs: Dict[str, Any]) -> Any:
        AutoTS = self._backend_module
        forecast_len = kwargs.get('forecast_horizon', 1)
        if self.forecast_length is None:
            self.forecast_length = forecast_len

        ts_kwargs = {
            'forecast_length': self.forecast_length,
            'frequency': self.frequency,
            'prediction_interval': self.prediction_interval,
            'model_list': self.model_list if self.model_list is not None else "scalable",
            'transformer_list': self.transformer_list,
            'models_mode': self.models_mode,
            'random_seed': budget.random_seed or self.random_seed,
            'verbose': self.verbose,
        }
        if self.max_generations is not None:
            ts_kwargs['max_generations'] = self.max_generations
        elif budget.trial_limit is not None:
            ts_kwargs['max_generations'] = budget.trial_limit

        if budget.memory_limit_mb is not None:
            warnings.warn("memory_limit_mb not supported by AutoTS.")
        if budget.max_models is not None:
            warnings.warn("max_models not directly supported. Use model_list instead.")

        model = AutoTS(**ts_kwargs)
        model.fit(data, date_col='datetime', value_col='value')

        self._fitted_predictor = model
        self._early_stopped = getattr(model, '_stopped_early', False)
        return model

    def _predict(self, model: Any, data: pd.DataFrame, forecast_horizon: int) -> np.ndarray:
        try:
            predictions = model.predict(forecast_length=forecast_horizon)
        except TypeError:
            predictions = model.predict()

        if isinstance(predictions, tuple):
            forecast_df = predictions[0]
        else:
            forecast_df = predictions

        if 'series_id' in forecast_df.columns:
            first_series = forecast_df['series_id'].unique()[0]
            df_series = forecast_df[forecast_df['series_id'] == first_series]
        else:
            df_series = forecast_df

        if 'mean' in df_series.columns:
            point = df_series['mean'].values
        elif 'yhat' in df_series.columns:
            point = df_series['yhat'].values
        else:
            point = df_series.iloc[:, 0].values

        return np.asarray(point, dtype=float).reshape(-1)[:forecast_horizon]

    def _predict_quantiles(self, model: Any, data: pd.DataFrame,
                           forecast_horizon: int, quantiles: List[float]) -> np.ndarray:
        try:
            predictions = model.predict(forecast_length=forecast_horizon)
        except TypeError:
            predictions = model.predict()

        if isinstance(predictions, tuple):
            forecast_df = predictions[0]
        else:
            forecast_df = predictions

        if 'series_id' in forecast_df.columns:
            first_series = forecast_df['series_id'].unique()[0]
            df_series = forecast_df[forecast_df['series_id'] == first_series]
        else:
            df_series = forecast_df

        quantile_data = []
        for q in quantiles:
            col_candidates = [f'{q:.2f}', f'{q:.1f}', f'q{q:.2f}', f'q{q:.1f}', str(q)]
            found = False
            for col in col_candidates:
                if col in df_series.columns:
                    quantile_data.append(df_series[col].values)
                    found = True
                    break
            if not found:
                if 'lower_CI' in df_series.columns and 'upper_CI' in df_series.columns:
                    lower = df_series['lower_CI'].values
                    upper = df_series['upper_CI'].values
                    mean = df_series['mean'].values if 'mean' in df_series.columns else np.full_like(lower, np.nan)
                    if q <= 0.5:
                        alpha = q / 0.5
                        values = lower + alpha * (mean - lower)
                    else:
                        alpha = (q - 0.5) / 0.5
                        values = mean + alpha * (upper - mean)
                    quantile_data.append(values)
                else:
                    mean = df_series['mean'].values if 'mean' in df_series.columns else df_series.iloc[:, 0].values
                    quantile_data.append(mean)

        result = np.column_stack(quantile_data)
        return result[:forecast_horizon, :]

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
                horizon = dataset_record[0].forecast_horizon
                for rec in dataset_record:
                    if rec.forecast_horizon != horizon:
                        warnings.warn(
                            "Different forecast horizons in panel data. "
                            f"Using max horizon: {max(rec.forecast_horizon for rec in dataset_record)}"
                        )
                        horizon = max(rec.forecast_horizon for rec in dataset_record)
                kwargs = self._get_kwargs(dataset_record[0])
                kwargs['forecast_horizon'] = horizon
            else:
                horizon = dataset_record.forecast_horizon
                kwargs = self._get_kwargs(dataset_record)

            try:
                fitted_model = self._fit(data, self.budget, kwargs)
                self._fitted_predictor = fitted_model
            except TimeoutError as e:
                raise ModelExecutionError(RunStatus.TIMEOUT, f"Budget time limit exceeded: {e}")
            except MemoryError as e:
                raise ModelExecutionError(RunStatus.MEMORY_ERROR, f"Budget memory limit exceeded: {e}")

            forecast = self._predict(fitted_model, data, horizon)
            forecast = np.asarray(forecast, dtype=float).reshape(-1)

            quantiles = kwargs.get('quantiles', self.quantiles)
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
                f"AutoTS forecast failed: {e}",
                budget_info=self._resource_report.to_dict() if self._resource_report else None
            )

    def _extract_report(self, model: Any, start_time: float, end_time: float,
                        kwargs: Dict[str, Any]) -> AutoMLResourceReport:
        wall_clock = end_time - start_time
        model_count = 0
        if hasattr(model, 'model_list'):
            model_count = len(model.model_list) if model.model_list else 0
        elif hasattr(model, 'best_model'):
            model_count = 1
        else:
            model_count = getattr(model, '_model_count', 0)

        memory_usage_mb = 0.0
        try:
            import psutil
            memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            pass

        total_trials = model_count

        return AutoMLResourceReport(
            wall_clock_sec=wall_clock,
            memory_usage_mb=memory_usage_mb,
            models_fitted=model_count,
            quantile_support=self.supports_quantiles(),
            total_trials=total_trials,
            early_stopped=self._early_stopped,
        )

    def supports_quantiles(self) -> bool:
        return bool(self.quantiles)

    def _get_optional_dependencies(self) -> List[str]:
        return ['psutil', 'pandas']

    def availability(self) -> Tuple[RunStatus, str]:
        try:
            self._backend_module = self._import_backend()
            self._backend_imported = True
            return RunStatus.SUCCESS, "AutoTS available"
        except ImportError as e:
            return RunStatus.NOT_AVAILABLE, f"AutoTS not installed: {e}"
        except Exception as e:
            return RunStatus.BACKEND_UNAVAILABLE, f"AutoTS initialization failed: {e}"


def create_autots_adapter(
    name: str = "AutoTS",
    tags: Tuple[str, ...] = ('automl', 'forecasting', 'external', 'autots'),
    **kwargs
) -> AutoTSAdapter:
    budget_kwargs = {k: kwargs.pop(k) for k in ['time_limit_sec', 'trial_limit', 'memory_limit_mb',
                                                'random_seed', 'early_stopping', 'max_models'] if k in kwargs}
    budget = AutoMLBudget(**budget_kwargs) if budget_kwargs else None

    adapter_kwargs = {
        'name': name,
        'tags': tags,
        'budget': budget,
        'forecast_length': kwargs.pop('forecast_length', None),
        'frequency': kwargs.pop('frequency', 'D'),
        'prediction_interval': kwargs.pop('prediction_interval', 0.9),
        'quantiles': kwargs.pop('quantiles', [0.1, 0.5, 0.9]),
        'model_list': kwargs.pop('model_list', None),
        'transformer_list': kwargs.pop('transformer_list', None),
        'models_mode': kwargs.pop('models_mode', 'default'),
        'max_generations': kwargs.pop('max_generations', None),
        'random_seed': kwargs.pop('random_seed', 42),
        'verbose': kwargs.pop('verbose', 0),
        'data_format': kwargs.pop('data_format', AutoMLDataFormat.UNIVARIATE),
    }
    return AutoTSAdapter(**adapter_kwargs)
