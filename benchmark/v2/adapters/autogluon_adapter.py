from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import time

try:
    import psutil
except ImportError:
    psutil = None

try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    TimeSeriesPredictor = None
    TimeSeriesDataFrame = None

from benchmark.v2.core import RunStatus, ModelFamily, ForecastingDatasetRecord
from benchmark.v2.adapters.automl_common import (
    AutoMLBudget,
    AutoMLResourceReport,
    AutoMLDataFormat,
    CommonAutoMLAdapter,
    ModelExecutionError,
)
from benchmark.v2.covariates import CovariateCapabilities, CovariateType, CovariateSupport


@dataclass
class AutoGluonAdapter(CommonAutoMLAdapter):

    backend_name: str = "autogluon"
    name: str = "AutoGluon"
    data_format: AutoMLDataFormat = AutoMLDataFormat.UNIVARIATE

    presets: str = "best_quality"
    eval_metric: str = "sMAPE"
    hyperparameters: Optional[Dict[str, Any]] = None
    hyperparameter_tune_kwargs: Optional[Dict[str, Any]] = None
    excluded_model_types: Optional[List[str]] = None
    quantile_levels: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    _fitted_predictor: Optional[TimeSeriesPredictor] = None
    _backend_imported: bool = False
    _backend_module: Optional[Tuple] = None
    _resource_report: Optional[AutoMLResourceReport] = None
    _early_stopped: bool = False
    _time_limit: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()

        if self.budget is None:
            self.budget = AutoMLBudget(time_limit_sec=3600, random_seed=42)

        valid_presets = ["best_quality", "medium_quality", "low_quality"]
        if self.presets not in valid_presets:
            warnings.warn(
                f"Invalid preset '{self.presets}'. Using 'best_quality'. "
                f"Valid options: {valid_presets}"
            )
            self.presets = "best_quality"

        if self.budget:
            if self.budget.trial_limit is not None:
                warnings.warn(
                    "trial_limit is not supported by AutoGluon adapter. "
                    "Use time_limit_sec instead."
                )
            if self.budget.memory_limit_mb is not None:
                warnings.warn(
                    "memory_limit_mb is not supported by AutoGluon adapter."
                )
            if self.budget.max_models is not None:
                warnings.warn(
                    "max_models is not supported by AutoGluon adapter. "
                    "Use hyperparameters to control model selection."
                )

        self._quantile_supported = len(self.quantile_levels) > 0

        if isinstance(self.data_format, str):
            self.data_format = AutoMLDataFormat(self.data_format.lower())
        super().__post_init__()


    def _import_backend(self) -> Tuple:
        if not AUTOGLUON_AVAILABLE:
            raise ImportError(
                "AutoGluon is not installed. "
                "Install with: pip install autogluon.timeseries"
            )
        try:
            from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
            return (TimeSeriesPredictor, TimeSeriesDataFrame)
        except ImportError as e:
            raise ImportError(f"Failed to import AutoGluon TimeSeries: {e}")

    def _frequency_to_pandas_freq(self, frequency: str) -> str:
        freq_map = {
            'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q', 'Y': 'Y',
            'H': 'H', 'T': 'T', 'S': 'S',
            'MS': 'MS', 'QS': 'QS', 'YS': 'YS',
        }
        freq_upper = frequency.upper()
        if freq_upper in freq_map:
            return freq_map[freq_upper]
        if freq_upper.startswith('W'):
            return 'W'
        if freq_upper.startswith('M'):
            return 'M'
        if freq_upper.startswith('Q'):
            return 'Q'
        if freq_upper.startswith('Y'):
            return 'Y'
        if freq_upper.startswith('H'):
            return 'H'
        if freq_upper.startswith('T') or freq_upper.startswith('MIN'):
            return 'T'
        warnings.warn(f"Unknown frequency '{frequency}', defaulting to 'D'")
        return 'D'

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
            if isinstance(dataset_record, list):
                if self.data_format != AutoMLDataFormat.PANEL:
                    warnings.warn(
                        "Received list of records but data_format is not PANEL. "
                        "Treating as panel data."
                    )
                data = self._prepare_panel_data(dataset_record)
                forecast_horizon = dataset_record[0].forecast_horizon
                framework_kwargs = self._get_kwargs(dataset_record[0])
                for rec in dataset_record:
                    if rec.forecast_horizon != forecast_horizon:
                        warnings.warn(
                            "Different forecast horizons in panel data. "
                            f"Using max horizon: {max(rec.forecast_horizon for rec in dataset_record)}"
                        )
                        forecast_horizon = max(rec.forecast_horizon for rec in dataset_record)
                framework_kwargs['forecast_horizon'] = forecast_horizon
            else:
                data = self._prepare_data(dataset_record)
                forecast_horizon = dataset_record.forecast_horizon
                framework_kwargs = self._get_kwargs(dataset_record)

            known_covariates_data = self._prepare_known_covariates(dataset_record, forecast_horizon)

            try:
                fitted_model = self._fit(data, self.budget, framework_kwargs)
                self._fitted_predictor = fitted_model
            except TimeoutError as e:
                raise ModelExecutionError(RunStatus.TIMEOUT, f"Budget time limit exceeded: {e}")
            except MemoryError as e:
                raise ModelExecutionError(RunStatus.MEMORY_ERROR, f"Budget memory limit exceeded: {e}")

            forecast = self._predict(
                fitted_model,
                data,
                forecast_horizon,
                known_covariates=known_covariates_data
            )
            forecast = np.asarray(forecast, dtype=float).reshape(-1)

            quantiles = framework_kwargs.get('quantiles', self.quantile_levels)
            quantile_forecasts = None
            if self.supports_quantiles():
                try:
                    quantile_forecasts = self._predict_quantiles(
                        fitted_model,
                        data,
                        forecast_horizon,
                        quantiles,
                        known_covariates=known_covariates_data
                    )
                except NotImplementedError:
                    pass

            end_time = time.time()
            end_cpu = time.process_time()

            report = self._extract_report(fitted_model, start_time, end_time, framework_kwargs)
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
                'kwargs': framework_kwargs,
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

    def _prepare_panel_data(self, records: List[ForecastingDatasetRecord]) -> TimeSeriesDataFrame:
        if not AUTOGLUON_AVAILABLE:
            raise RuntimeError("AutoGluon is not available")

        all_dfs = []
        static_features_dict = {}

        for rec in records:
            freq = self._frequency_to_pandas_freq(rec.frequency)
            train_y = np.asarray(rec.train_values, dtype=float)
            start_date = pd.Timestamp('2000-01-01')
            timestamps = pd.date_range(start=start_date, periods=len(train_y), freq=freq)

            df = pd.DataFrame({
                'target': train_y,
                'timestamp': timestamps,
                'item_id': rec.series_id,
            })

            # Ковариаты
            if rec.observed_past_covariates:
                for name, values in rec.observed_past_covariates.items():
                    df[name] = np.asarray(values)[:len(train_y)]

            if rec.known_future_covariates:
                for name, values in rec.known_future_covariates.items():
                    df[name] = np.asarray(values)[:len(train_y)]

            all_dfs.append(df)

            static_dict = {}
            if rec.static_covariates:
                static_dict.update(rec.static_covariates)
            if rec.panel_ids:
                static_dict.update(rec.panel_ids)
            if rec.hierarchy:
                static_dict.update(rec.hierarchy)
            if static_dict:
                static_features_dict[rec.series_id] = static_dict

        combined_df = pd.concat(all_dfs, ignore_index=True)
        ts_df = TimeSeriesDataFrame.from_data_frame(
            combined_df,
            id_column='item_id',
            timestamp_column='timestamp'
        )

        if static_features_dict:
            static_df = pd.DataFrame.from_dict(static_features_dict, orient='index')
            static_df.index.name = 'item_id'
            ts_df.static_features = static_df

        return ts_df

    def _prepare_data(self, dataset_record: ForecastingDatasetRecord) -> TimeSeriesDataFrame:
        if not AUTOGLUON_AVAILABLE:
            raise RuntimeError("AutoGluon is not available")

        item_id = dataset_record.series_id
        freq = self._frequency_to_pandas_freq(dataset_record.frequency)
        train_y = np.asarray(dataset_record.train_values, dtype=float)

        start_date = pd.Timestamp('2000-01-01')
        timestamps = pd.date_range(start=start_date, periods=len(train_y), freq=freq)

        df = pd.DataFrame({
            'target': train_y,
            'timestamp': timestamps,
            'item_id': item_id,
        })

        if dataset_record.observed_past_covariates:
            for name, values in dataset_record.observed_past_covariates.items():
                df[name] = np.asarray(values)[:len(train_y)]

        if dataset_record.known_future_covariates:
            for name, values in dataset_record.known_future_covariates.items():
                df[name] = np.asarray(values)[:len(train_y)]

        ts_df = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column='item_id',
            timestamp_column='timestamp'
        )

        static_dict = {}
        if dataset_record.static_covariates:
            static_dict.update(dataset_record.static_covariates)
        if dataset_record.panel_ids:
            static_dict.update(dataset_record.panel_ids)
        if dataset_record.hierarchy:
            static_dict.update(dataset_record.hierarchy)

        if static_dict:
            static_df = pd.DataFrame([static_dict], index=[item_id])
            static_df.index.name = 'item_id'
            ts_df.static_features = static_df

        return ts_df

    def _prepare_known_covariates(
        self,
        dataset_record: Union[ForecastingDatasetRecord, List[ForecastingDatasetRecord]],
        forecast_horizon: int
    ) -> Optional[TimeSeriesDataFrame]:
        if not AUTOGLUON_AVAILABLE:
            raise RuntimeError("AutoGluon is not available")

        records = [dataset_record] if not isinstance(dataset_record, list) else dataset_record

        has_future = any(rec.known_future_covariates for rec in records)
        if not has_future:
            return None

        all_dfs = []
        for rec in records:
            freq = self._frequency_to_pandas_freq(rec.frequency)
            train_len = len(rec.train_values)
            future_timestamps = pd.date_range(
                start=pd.Timestamp('2000-01-01') + pd.Timedelta(days=train_len),
                periods=forecast_horizon,
                freq=freq
            )
            df_future = pd.DataFrame({
                'timestamp': future_timestamps,
                'item_id': rec.series_id,
            })
            if rec.known_future_covariates:
                for name, values in rec.known_future_covariates.items():
                    future_values = np.asarray(values)[train_len:train_len + forecast_horizon]
                    df_future[name] = future_values
            all_dfs.append(df_future)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        return TimeSeriesDataFrame.from_data_frame(
            combined_df,
            id_column='item_id',
            timestamp_column='timestamp'
        )

    def _fit(
        self,
        data: TimeSeriesDataFrame,
        budget: AutoMLBudget,
        kwargs: Dict[str, Any],
    ) -> TimeSeriesPredictor:
        if not AUTOGLUON_AVAILABLE:
            raise RuntimeError("AutoGluon is not available")

        prediction_length = kwargs.get('forecast_horizon', 1)
        quantile_levels = kwargs.get('quantiles', self.quantile_levels)

        predictor = TimeSeriesPredictor(
            prediction_length=prediction_length,
            target='target',
            eval_metric=self.eval_metric,
            path=kwargs.get('path', './autogluon_benchmark'),
            verbosity=kwargs.get('verbosity', 0),
            quantile_levels=quantile_levels,
        )

        fit_kwargs = {'presets': self.presets}

        if budget.time_limit_sec is not None and budget.time_limit_sec > 0:
            fit_kwargs['time_limit'] = budget.time_limit_sec

        if budget.random_seed is not None:
            fit_kwargs['random_seed'] = budget.random_seed

        if self.hyperparameters is not None:
            fit_kwargs['hyperparameters'] = self.hyperparameters

        if self.hyperparameter_tune_kwargs is not None:
            fit_kwargs['hyperparameter_tune_kwargs'] = self.hyperparameter_tune_kwargs

        if self.excluded_model_types is not None:
            fit_kwargs['excluded_model_types'] = self.excluded_model_types

        self._early_stopped = False

        try:
            self._time_limit = fit_kwargs.get('time_limit')
            predictor.fit(data, **fit_kwargs)

            # Проверяем early stopping
            if hasattr(predictor, '_trainer'):
                trainer = predictor._trainer
                if hasattr(trainer, 'model_results'):
                    for model_name, result in trainer.model_results.items():
                        if hasattr(result, 'stopped_early') and result.stopped_early:
                            self._early_stopped = True
                            break
                if hasattr(trainer, 'time_limit_exceeded'):
                    self._early_stopped = getattr(trainer, 'time_limit_exceeded', False)

        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "time limit" in error_msg:
                self._early_stopped = True
                raise TimeoutError(f"AutoGluon fit exceeded time limit: {e}")
            if "memory" in error_msg:
                raise MemoryError(f"AutoGluon fit exceeded memory limit: {e}")
            raise

        self._fitted_predictor = predictor
        return predictor

    def _predict(
        self,
        model: TimeSeriesPredictor,
        data: TimeSeriesDataFrame,
        forecast_horizon: int,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
    ) -> np.ndarray:
        if not AUTOGLUON_AVAILABLE:
            raise RuntimeError("AutoGluon is not available")

        try:
            predictions = model.predict(data, known_covariates=known_covariates)
            if 'mean' in predictions.columns:
                forecast = predictions['mean'].values
            else:
                forecast = predictions.iloc[:, 0].values
            return np.asarray(forecast, dtype=float).reshape(-1)
        except Exception as e:
            raise RuntimeError(f"AutoGluon prediction failed: {e}")

    def _predict_quantiles(
        self,
        model: TimeSeriesPredictor,
        data: TimeSeriesDataFrame,
        forecast_horizon: int,
        quantiles: List[float],
        known_covariates: Optional[TimeSeriesDataFrame] = None,
    ) -> np.ndarray:
        if not AUTOGLUON_AVAILABLE:
            raise RuntimeError("AutoGluon is not available")

        try:
            predictions = model.predict(data, known_covariates=known_covariates)
            quantile_data = []

            for q in quantiles:
                col_candidates = [
                    f'q{q:.2f}', f'{q:.2f}', f'{q:.1f}', f'q{q:.1f}', str(q)
                ]
                found = False
                for col in col_candidates:
                    if col in predictions.columns:
                        quantile_data.append(predictions[col].values)
                        found = True
                        break

                if not found:
                    for col in predictions.columns:
                        try:
                            col_float = float(col)
                            if abs(col_float - q) < 0.001:
                                quantile_data.append(predictions[col].values)
                                found = True
                                break
                        except (ValueError, TypeError):
                            continue

                if not found:
                    quantile_data.append(predictions['mean'].values)

            return np.column_stack(quantile_data)

        except Exception as e:
            raise RuntimeError(f"AutoGluon quantile prediction failed: {e}")

    def _extract_report(
        self,
        model: TimeSeriesPredictor,
        start_time: float,
        end_time: float,
        kwargs: Dict[str, Any],
    ) -> AutoMLResourceReport:
        wall_clock = end_time - start_time

        models_fitted = 0
        if hasattr(model, 'model_names'):
            try:
                models_fitted = len(model.model_names())
            except Exception:
                pass

        if models_fitted == 0 and hasattr(model, '_trainer'):
            try:
                trainer = model._trainer
                if hasattr(trainer, 'model_names'):
                    models_fitted = len(trainer.model_names())
                elif hasattr(trainer, 'model_results'):
                    models_fitted = len(trainer.model_results)
            except Exception:
                pass

        memory_usage_mb = 0.0
        if psutil is not None:
            try:
                memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            except Exception:
                pass

        total_trials = 0
        if hasattr(model, '_trainer') and hasattr(model._trainer, 'model_results'):
            try:
                total_trials = len(model._trainer.model_results)
            except Exception:
                pass

        if total_trials == 0:
            total_trials = models_fitted

        return AutoMLResourceReport(
            wall_clock_sec=wall_clock,
            memory_usage_mb=memory_usage_mb,
            models_fitted=models_fitted,
            failed_trials=0,
            quantile_support=self.supports_quantiles(),
            total_trials=total_trials,
            early_stopped=self._early_stopped,
        )

    def supports_quantiles(self) -> bool:
        return self._quantile_supported

    def get_covariate_capabilities(self) -> CovariateCapabilities:
        from benchmark.v2.covariates import CovariateCapabilities
        return CovariateCapabilities(
            supported_types={CovariateType.KNOWN_FUTURE, CovariateType.STATIC},
            support_levels={
                CovariateType.KNOWN_FUTURE: CovariateSupport.OPTIONAL,
                CovariateType.STATIC: CovariateSupport.OPTIONAL,
            },
            max_known_future=10,
            max_static=5,
        )

    def availability(self) -> Tuple[RunStatus, str]:
        try:
            self._import_backend()
            self._backend_imported = True
            if self.budget and self.budget.time_limit_sec is not None and self.budget.time_limit_sec <= 0:
                return RunStatus.NOT_AVAILABLE, "time_limit_sec must be positive"
            if self.quantile_levels:
                invalid = [q for q in self.quantile_levels if not (0 < q < 1)]
                if invalid:
                    return RunStatus.NOT_AVAILABLE, f"Invalid quantile levels: {invalid}"
            return RunStatus.SUCCESS, "AutoGluon TimeSeries is available"
        except ImportError as e:
            return RunStatus.NOT_AVAILABLE, f"AutoGluon import failed: {e}"
        except Exception as e:
            return RunStatus.BACKEND_UNAVAILABLE, f"AutoGluon initialization failed: {e}"


def create_autogluon_adapter(
    name: str = "AutoGluon",
    tags: Tuple[str, ...] = ('automl', 'forecasting', 'external', 'autogluon'),
    **kwargs
) -> AutoGluonAdapter:
    budget_kwargs = {}
    for key in ['time_limit_sec', 'trial_limit', 'memory_limit_mb',
                'random_seed', 'early_stopping', 'max_models']:
        if key in kwargs:
            budget_kwargs[key] = kwargs.pop(key)

    budget = AutoMLBudget(**budget_kwargs) if budget_kwargs else None
    quantile_levels = kwargs.pop('quantile_levels', [0.1, 0.5, 0.9])

    adapter_kwargs = {
        'name': name,
        'tags': tags,
        'budget': budget,
        'quantile_levels': quantile_levels,
        'presets': kwargs.pop('presets', 'best_quality'),
        'eval_metric': kwargs.pop('eval_metric', 'sMAPE'),
        'hyperparameters': kwargs.pop('hyperparameters', None),
        'hyperparameter_tune_kwargs': kwargs.pop('hyperparameter_tune_kwargs', None),
        'excluded_model_types': kwargs.pop('excluded_model_types', None),
        'data_format': kwargs.pop('data_format', AutoMLDataFormat.UNIVARIATE),
    }
    return AutoGluonAdapter(**adapter_kwargs)
