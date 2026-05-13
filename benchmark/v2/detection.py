from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional in lightweight environments
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    ConvAutoEncoderDetectionModel = None
    TCN_AutoEncoderDetectionModel = None

from fedot_ind.core.repository.detection_registry import canonical_detection_model_name

from benchmark.v2.core import (
    BenchmarkSuiteConfig,
    DetectionSeriesRecord,
    DetectionModelAdapter,
    DatasetSpec,
    MetricRecord,
    ModelSpec,
    PredictionRecord,
    RunStatus,
    TaskType,
    DetectionBenchmarkResult,
    DetectionSeriesRecord,
    BenchmarkRunRecord,
    BenchmarkAggregateReport,
    new_run_id,
)

from benchmark.v2.progress import BenchmarkProgressMonitor


SUPPORTED_ANOMALY_DETECTION_METRICS = ('accuracy', 'balanced_accuracy', 'f1_macro')

# PROJECT_ROOT
# DEFAULT_LOCAL_M4_DIR
# DEFAULT_LOCAL_MONASH_DIR

class BenchmarkConfigurationError(ValueError):
    pass

class ModelExecutionError(RuntimeError):
    def __init__(self, status: RunStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message

def validate_detection_suite_config(config: BenchmarkSuiteConfig) -> None:
    """проверка, что BenchmarkSuiteConfig подходит для задачи anomaly_detection"""
    if config.task_type is not TaskType.ANOMALY_DETECTION:
        raise BenchmarkConfigurationError('Anomaly detection suite expects task_type=anomaly_detection.')
    if not config.datasets:
        raise BenchmarkConfigurationError('Benchmark suite must contain at least one dataset spec.')
    if not config.models:
        raise BenchmarkConfigurationError('Benchmark suite must contain at least one model spec.')
    unsupported = set(config.metrics) - set(SUPPORTED_ANOMALY_DETECTION_METRICS)
    if unsupported:
        raise BenchmarkConfigurationError(f'Unsupported anomaly detection metrics: {sorted(unsupported)}')

# функции хелперы для работы с датасет адаптерами и _resolve_stage_tuning_split_spec()
class SKABAdapter:
    """чтение файла датасета skab и приведение к единому формату.
    Адапрер возвращает record (один временной ряд, у которого есть train/test и horizon).
    После adapter-а runner работает одинаково с любым датасетом.
    """
    benchmark_name = 'skab'
    def __init__():
        """Инициализирует self.loader входным в __init__ или _default_loader()
        """
        pass

    def load_series(self, spec: DatasetSpec) -> tuple[DetectionSeriesRecord, ...]:
        ...
    
    def _load_local_records():
        pass
    
    @staticmethod
    def _default_loader():
        pass

class MPSIAdapter:
    """чтение файла датасета mpsi и приведение к единому формату.
    Адапрер возвращает record (один временной ряд, у которого есть train/test и horizon).
    После adapter-а runner работает одинаково с любым датасетом.
    """
    benchmark_name = 'mpsi'

    def load_series(self, spec: DatasetSpec) -> tuple[DetectionSeriesRecord, ...]:
        ...

# class InMemoryForecastingAdapter:
#     """не реализован, надо разобраться, зачем это и куда дальше нужно
#     """
#     raise NotImplementedError

# def _safe_import(module_name: str) -> bool:
#     try:
#         importlib.import_module(module_name)
#         return True
#     except Exception:
#         return False

# ХЗ, нужен ли нам аналог _build_fedot_forecasting_input()

@dataclass
class NaiveModel(DetectionModelAdapter):
    def availability(self) -> tuple[RunStatus, str]:
        raise NotImplementedError
        # return RunStatus.SUCCESS, 'ready'
    def detect(self, series_record: DetectionSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        # pass
        raise NotImplementedError

@dataclass
class IsolationForestDetectionModel(DetectionModelAdapter):
    # поля c типом и дефолтными значениями
    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'
    def detect(self, series_record: DetectionSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        pass

@dataclass
class OneClassDetectionModel(DetectionModelAdapter):
    # поля c типом и дефолтными значениями
    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'
    def detect(self, series_record: DetectionSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        pass

@dataclass
class ConvAutoEncoderDetectionModel(DetectionModelAdapter):
    # поля c типом и дефолтными значениями
    def availability(self) -> tuple[RunStatus, str]:
        if ConvAutoEncoderDetectionModel is None:
            return RunStatus.NOT_AVAILABLE, 'torch is required for neural anomaly detectors.'
        return RunStatus.SUCCESS, 'ready'
    def detect(self, series_record: DetectionSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        pass

@dataclass
class TCN_AutoEncoderDetectionModel(DetectionModelAdapter):
    # поля c типом и дефолтными значениями
    def availability(self) -> tuple[RunStatus, str]:
        if TCN_AutoEncoderDetectionModel is None:
            return RunStatus.NOT_AVAILABLE, 'torch is required for neural anomaly detectors.'
        return RunStatus.SUCCESS, 'ready'
    def detect(self, series_record: DetectionSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        pass

def build_dataset_adapter(spec: DatasetSpec):
    """выбор адаптера"""
    benchmark = spec.benchmark.lower()
    if benchmark == 'skab':
        return SKABAdapter()
    if benchmark == 'mpsi':
        return MPSIAdapter()
    # if benchmark == 'in_memory':
    #     return InMemoryDetectionAdapter() TODO: разобраться что такое InMemoryDetectionAdapter в forecasting
    raise BenchmarkConfigurationError(f'Unsupported anomaly detection benchmark adapter: {spec.benchmark}')

def build_model_adapter(spec: ModelSpec) -> DetectionModelAdapter: # TODO: реализовать классы моделей, наследуемые от DetectionModelAdapter
    """выбор модели. создаёт объект класса модели с 
    аргументами для работы модели и функциями availability() и forecast()"""
    raw_adapter_name = spec.adapter_name.lower()
    adapter_name = canonical_detection_model_name(raw_adapter_name)
    params = dict(spec.params)

    # if adapter_name in CANONICAL_STAGE_DETECTION_MODELS:
    #     return RuntimeDetectionModelAdapter(...)

    # raise BenchmarkConfigurationError(...)
    if adapter_name == 'feature_iforest_detector':
        return IsolationForestDetectionModel()
    if adapter_name == 'feature_oneclass_detector':
        return OneClassDetectionModel()
    if adapter_name == 'conv_autoencoder_detector':
        return ConvAutoEncoderDetectionModel()
    if adapter_name == 'tcn_autoencoder_detector':
        return TCN_AutoEncoderDetectionModel()
    
    # наивные модели
    if adapter_name == 'naive':
        return NaiveModel()

    raise BenchmarkConfigurationError(f'Unsupported forecasting model adapter: {spec.adapter_name}')

def build_leaderboard(
        run_records: tuple[BenchmarkRunRecord, ...],
        primary_metric: str,
) -> BenchmarkAggregateReport:
    successful = [record for record in run_records if record.status is RunStatus.SUCCESS]
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for record in successful:
        metric_value = record.metrics_summary.get(primary_metric)
        if metric_value is None:
            continue
        grouped.setdefault((record.benchmark, record.dataset_name, record.model_name), []).append(metric_value)

    leaderboard_rows = []
    for (benchmark, dataset_name, model_name), values in grouped.items():
        leaderboard_rows.append(
            {
                'benchmark': benchmark,
                'dataset_name': dataset_name,
                'model_name': model_name,
                primary_metric: float(np.mean(values)),
                'n_series': len(values),
            }
        )

    leaderboard_rows = sorted(leaderboard_rows, key=lambda row: row[primary_metric])
    for rank, row in enumerate(leaderboard_rows, start=1):
        row['rank'] = rank

    status_counts: dict[str, int] = {}
    for record in run_records:
        status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1

    run_id = run_records[0].run_id if run_records else new_run_id('empty')
    return BenchmarkAggregateReport(
        run_id=run_id,
        task_type=TaskType.ANOMALY_DETECTION,
        primary_metric=primary_metric,
        leaderboard_rows=tuple(leaderboard_rows),
        status_counts=status_counts,
    )

class DetectionSuiteRunner:
    def __init__(self, config: BenchmarkSuiteConfig):
        validate_detection_suite_config(config)
        self.config = config
        self.run_id = new_run_id(config.run_spec.run_name)
        self.series_records: list[DetectionSeriesRecord] = []
        self.run_records: list[BenchmarkRunRecord] = []
        self.prediction_records: list[PredictionRecord] = []
        self.metric_records: list[MetricRecord] = []
        self.progress = BenchmarkProgressMonitor(
            enabled=config.run_spec.show_progress,
            task_type=config.task_type.value,
            run_name=config.run_spec.run_name,
            leave=config.run_spec.progress_leave,
            log_errors=config.run_spec.progress_log_errors,
            log_summaries=config.run_spec.progress_log_summaries,
        )

    def run_suite(self) -> DetectionBenchmarkResult:
        try:
            self._iter_over_datasets()
        finally:
            self.progress.close()

        aggregate_report = build_leaderboard(
            tuple(self.run_records),
            primary_metric=self.config.run_spec.primary_metric,
        )
        return DetectionBenchmarkResult(
            run_id=self.run_id,
            config=self.config,
            series_records=tuple(self.series_records),
            run_records=tuple(self.run_records),
            prediction_records=tuple(self.prediction_records),
            metric_records=tuple(self.metric_records),
            aggregate_report=aggregate_report,
        )

    def _iter_over_datasets(self) -> None:
        for dataset_spec in self.config.datasets:
            dataset_series = self._load_dataset_series(dataset_spec)
            self._iter_over_models(dataset_spec, dataset_series)
            self.progress.dataset_finished()

    def _load_dataset_series(self, dataset_spec: DatasetSpec) -> tuple[DetectionSeriesRecord, ...]:
        dataset_adapter = build_dataset_adapter(dataset_spec)
        dataset_series = dataset_adapter.load_series(dataset_spec)
        self.series_records.extend(dataset_series)
        self.progress.extend_total(len(dataset_series) * len(self.config.models))
        self.progress.dataset_loaded(dataset_spec.dataset_name, len(dataset_series))
        return dataset_series

    def _iter_over_models(
            self,
            dataset_spec: DatasetSpec,
            dataset_series: tuple[DetectionSeriesRecord, ...],
    ) -> None:
        for model_spec in self.config.models:
            model = build_model_adapter(model_spec)
            self.progress.model_started(dataset_spec.dataset_name, model.name)
            try:
                availability_status, availability_message = model.availability()
                if availability_status is not RunStatus.SUCCESS:
                    self._handle_unavailable_model(
                        model_spec=model_spec,
                        model=model,
                        dataset_series=dataset_series,
                        availability_status=availability_status,
                        availability_message=availability_message,
                    )
                    continue
                self._iter_over_series(model_spec, model, dataset_series)
            finally:
                self.progress.model_finished()

    # def _iter_over_series(
    #         self,
    #         model_spec: ModelSpec,
    #         model: ForecastingModelAdapter,
    #         dataset_series: tuple[ForecastingSeriesRecord, ...],
    # ) -> None:
    #     for series_record in dataset_series:
    #         self.progress.item_started(series_record.dataset_name, model.name, series_record.series_id)
    #         regime_diagnostics, routing_recommendation = self._build_series_context(series_record)
    #         try:
    #             self._evaluate_series(model_spec, model, series_record, regime_diagnostics, routing_recommendation)
    #             self.progress.advance(RunStatus.SUCCESS.value)
    #         except ModelExecutionError as exc:
    #             self._append_failed_run_record(
    #                 model_spec=model_spec,
    #                 model=model,
    #                 series_record=series_record,
    #                 status=exc.status,
    #                 message=exc.message,
    #                 regime_diagnostics=regime_diagnostics,
    #                 routing_recommendation=routing_recommendation,
    #             )
    #             self.progress.advance(exc.status.value, exc.message)
    #         except Exception as exc:
    #             self._append_failed_run_record(
    #                 model_spec=model_spec,
    #                 model=model,
    #                 series_record=series_record,
    #                 status=RunStatus.FAILED,
    #                 message=str(exc),
    #                 regime_diagnostics=regime_diagnostics,
    #                 routing_recommendation=routing_recommendation,
    #             )
    #             self.progress.advance(RunStatus.FAILED.value, str(exc))

    # def _handle_unavailable_model(
    #         self,
    #         *,
    #         model_spec: ModelSpec,
    #         model: ForecastingModelAdapter,
    #         dataset_series: tuple[ForecastingSeriesRecord, ...],
    #         availability_status: RunStatus,
    #         availability_message: str,
    # ) -> None:
    #     for series_record in dataset_series:
    #         self.progress.item_started(series_record.dataset_name, model.name, series_record.series_id)
    #         regime_diagnostics, routing_recommendation = self._build_series_context(series_record)
    #         self.run_records.append(
    #             BenchmarkRunRecord(
    #                 run_id=self.run_id,
    #                 benchmark=series_record.benchmark,
    #                 dataset_name=series_record.dataset_name,
    #                 subset=series_record.subset,
    #                 series_id=series_record.series_id,
    #                 model_name=model.name,
    #                 status=availability_status,
    #                 tags=model.tags,
    #                 message=availability_message,
    #                 metadata=self._build_common_metadata(
    #                     model_spec=model_spec,
    #                     model=model,
    #                     regime_diagnostics=regime_diagnostics,
    #                     routing_recommendation=routing_recommendation,
    #                     extra={'optional': model.optional},
    #                 ),
    #             )
    #         )
    #         self.progress.advance(availability_status.value, availability_message)

    # def _build_series_context(self, series_record: ForecastingSeriesRecord):
    #     regime_diagnostics = analyze_regime_diagnostics(np.asarray(series_record.train_values, dtype=float))
    #     routing_recommendation = recommend_forecasting_model(regime_diagnostics)
    #     return regime_diagnostics, routing_recommendation

    # def _build_common_metadata(
    #         self,
    #         *,
    #         model_spec: ModelSpec,
    #         model: ForecastingModelAdapter,
    #         regime_diagnostics,
    #         routing_recommendation,
    #         extra: dict[str, Any] | None = None,
    # ) -> dict[str, Any]:
    #     return {
    #         'adapter_name': model_spec.adapter_name,
    #         'model_adapter_family': adapter_name_to_family(model_spec.adapter_name),
    #         'regime_diagnostics': regime_diagnostics.to_dict(),
    #         'routing_recommendation': routing_recommendation.to_dict(),
    #         'routing_recommendation_family': adapter_name_to_family(routing_recommendation.primary_adapter),
    #         **dict(extra or {}),
    #     }

    # def _validate_forecast_length(
    #         self,
    #         series_record: ForecastingSeriesRecord,
    #         prediction: np.ndarray,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     actual = np.asarray(series_record.test_values, dtype=float)
    #     forecast = np.asarray(prediction, dtype=float).reshape(-1)[: len(actual)]
    #     if len(forecast) != len(actual):
    #         raise ModelExecutionError(
    #             RunStatus.FAILED,
    #             f'Model returned {len(forecast)} predictions for horizon {len(actual)}.',
    #         )
    #     return actual, forecast

    # def _record_metric_bundle(
    #         self,
    #         *,
    #         series_record: ForecastingSeriesRecord,
    #         model_name: str,
    #         actual: np.ndarray,
    #         forecast: np.ndarray,
    #         metadata: dict[str, Any],
    #         metric_name_suffix: str = '',
    # ) -> tuple[dict[str, float], dict[str, float]]:
    #     metrics_summary: dict[str, float] = {}
    #     train = np.asarray(series_record.train_values, dtype=float)
    #     for metric_name in self.config.metrics:
    #         metric_value = compute_forecasting_metric(
    #             metric_name,
    #             actual,
    #             forecast,
    #             train,
    #             series_record.seasonal_period,
    #         )
    #         metrics_summary[metric_name] = metric_value
    #         self.metric_records.append(
    #             MetricRecord(
    #                 run_id=self.run_id,
    #                 benchmark=series_record.benchmark,
    #                 dataset_name=series_record.dataset_name,
    #                 subset=series_record.subset,
    #                 series_id=series_record.series_id,
    #                 model_name=model_name,
    #                 metric_name=f'{metric_name}{metric_name_suffix}',
    #                 metric_value=metric_value,
    #                 status=RunStatus.SUCCESS,
    #             )
    #         )
    #         pointwise = compute_pointwise_metric(
    #             metric_name,
    #             actual,
    #             forecast,
    #             train,
    #             series_record.seasonal_period,
    #         )
    #         for horizon_index, pointwise_value in enumerate(pointwise, start=1):
    #             self.metric_records.append(
    #                 MetricRecord(
    #                     run_id=self.run_id,
    #                     benchmark=series_record.benchmark,
    #                     dataset_name=series_record.dataset_name,
    #                     subset=series_record.subset,
    #                     series_id=series_record.series_id,
    #                     model_name=model_name,
    #                     metric_name=f'{metric_name}{metric_name_suffix}',
    #                     metric_value=float(pointwise_value),
    #                     status=RunStatus.SUCCESS,
    #                     horizon_index=horizon_index,
    #                 )
    #             )

    #     event_metrics = _append_event_interval_metrics(
    #         self.metric_records,
    #         run_id=self.run_id,
    #         series_record=series_record,
    #         model_name=model_name,
    #         actual=actual,
    #         forecast=forecast,
    #         metadata=metadata,
    #         metric_name_suffix=metric_name_suffix,
    #     )
    #     metrics_summary.update({key: value for key, value in event_metrics.items() if key.startswith('mae_')})
    #     return metrics_summary, event_metrics

    # def _record_predictions(
    #         self,
    #         *,
    #         series_record: ForecastingSeriesRecord,
    #         model_name: str,
    #         actual: np.ndarray,
    #         forecast: np.ndarray,
    # ) -> None:
    #     for horizon_index, (actual_value, forecast_value) in enumerate(zip(actual, forecast), start=1):
    #         self.prediction_records.append(
    #             PredictionRecord(
    #                 run_id=self.run_id,
    #                 benchmark=series_record.benchmark,
    #                 dataset_name=series_record.dataset_name,
    #                 subset=series_record.subset,
    #                 series_id=series_record.series_id,
    #                 model_name=model_name,
    #                 horizon_index=horizon_index,
    #                 y_true=float(actual_value),
    #                 y_pred=float(forecast_value),
    #                 status=RunStatus.SUCCESS,
    #             )
    #         )

    # def _build_tuned_model_spec(self, model_spec: ModelSpec, best_parameters: dict[str, Any]) -> ModelSpec:
    #     tuned_params = {
    #         **dict(model_spec.params),
    #         **dict(best_parameters),
    #     }
    #     tuned_params.pop('stage_tuning_runtime', None)
    #     return ModelSpec(
    #         adapter_name=model_spec.adapter_name,
    #         display_name=model_spec.display_name,
    #         tags=model_spec.tags,
    #         optional=model_spec.optional,
    #         params=tuned_params,
    #     )

    # def _resolve_post_fit_tuning_runtime_config(
    #         self,
    #         model_spec: ModelSpec,
    #         baseline_metadata: dict[str, Any],
    # ) -> dict[str, Any]:
    #     raw_config = dict(model_spec.params.get('stage_tuning_runtime') or {})
    #     metadata_runtime = dict(baseline_metadata.get('stage_tuning_runtime') or {})
    #     return {
    #         'metric_name': str(raw_config.get('metric_name', metadata_runtime.get('metric_name',
    #                                                                               self.config.run_spec.primary_metric))),
    #         'stage_updates': raw_config.get('stage_updates'),
    #         'max_values_per_parameter': int(raw_config.get('max_values_per_parameter', 3)),
    #         'max_stage_candidates': int(raw_config.get('max_stage_candidates', 16)),
    #         'split_spec': _resolve_stage_tuning_split_spec(raw_config),
    #     }

    # def _maybe_run_post_fit_tuning_comparison(
    #         self,
    #         *,
    #         model_spec: ModelSpec,
    #         model: ForecastingModelAdapter,
    #         series_record: ForecastingSeriesRecord,
    #         baseline_metadata: dict[str, Any],
    #         baseline_metrics_summary: dict[str, float],
    #         regime_diagnostics,
    #         routing_recommendation,
    # ) -> dict[str, Any]:
    #     runtime_config = self._resolve_post_fit_tuning_runtime_config(model_spec, baseline_metadata)
    #     if run_forecasting_stage_tuning_on_series is None:
    #         return {
    #             **baseline_metadata,
    #             'stage_tuning_report_error': 'stage_tuning_runtime is unavailable in the current environment.',
    #         }

    #     try:
    #         report = run_forecasting_stage_tuning_on_series(
    #             model_spec.adapter_name,
    #             time_series=np.asarray(series_record.train_values, dtype=float),
    #             forecast_horizon=series_record.forecast_horizon,
    #             base_params={
    #                 key: value
    #                 for key, value in dict(model_spec.params).items()
    #                 if key != 'stage_tuning_runtime'
    #             },
    #             stage_updates=runtime_config.get('stage_updates'),
    #             metric_name=str(runtime_config['metric_name']),
    #             split_spec=runtime_config.get('split_spec'),
    #             seasonal_period=series_record.seasonal_period,
    #             max_values_per_parameter=int(runtime_config['max_values_per_parameter']),
    #             max_stage_candidates=int(runtime_config['max_stage_candidates']),
    #         )
    #         report_dict = report.to_dict()
    #         sequential_result = dict(report_dict.get('sequential_result') or {})
    #         best_parameters = dict(sequential_result.get('best_parameters') or {})
    #         enriched_baseline_metadata = {
    #             **baseline_metadata,
    #             'stage_tuning_report': report_dict,
    #             'stage_tuning_runtime': {
    #                 'enabled': True,
    #                 'metric_name': str(runtime_config['metric_name']),
    #                 'max_values_per_parameter': int(runtime_config['max_values_per_parameter']),
    #                 'max_stage_candidates': int(runtime_config['max_stage_candidates']),
    #                 'improved': report.metadata.get('improved'),
    #                 'baseline_score': report.metadata.get('baseline_score'),
    #                 'best_score': report.metadata.get('best_score'),
    #             },
    #         }
    #         if not best_parameters:
    #             return enriched_baseline_metadata

    #         tuned_model_spec = self._build_tuned_model_spec(model_spec, best_parameters)
    #         tuned_model = build_model_adapter(tuned_model_spec)
    #         tuned_status, tuned_message = tuned_model.availability()
    #         if tuned_status is not RunStatus.SUCCESS:
    #             return {
    #                 **enriched_baseline_metadata,
    #                 'stage_tuning_comparison_error': tuned_message,
    #             }

    #         tuned_prediction, tuned_metadata = tuned_model.forecast(series_record)
    #         actual, tuned_forecast = self._validate_forecast_length(series_record, tuned_prediction)
    #         tuned_metrics_summary, tuned_event_metrics = self._record_metric_bundle(
    #             series_record=series_record,
    #             model_name=model.name,
    #             actual=actual,
    #             forecast=tuned_forecast,
    #             metadata=tuned_metadata,
    #             metric_name_suffix='_tuned',
    #         )
    #         tuned_metrics_summary.update(
    #             {key: value for key, value in tuned_event_metrics.items() if key.startswith('mae_')}
    #         )
    #         return {
    #             **enriched_baseline_metadata,
    #             'stage_tuning_comparison': {
    #                 'best_parameters': best_parameters,
    #                 'baseline_metrics': baseline_metrics_summary,
    #                 'tuned_metrics': tuned_metrics_summary,
    #                 'improved_metrics': {
    #                     metric_name: bool(
    #                         tuned_metrics_summary.get(metric_name, math.inf)
    #                         <= baseline_metrics_summary.get(metric_name, math.inf)
    #                     )
    #                     for metric_name in baseline_metrics_summary
    #                     if metric_name in tuned_metrics_summary
    #                 },
    #                 'absolute_gain': {
    #                     metric_name: float(baseline_metrics_summary[metric_name] - tuned_metrics_summary[metric_name])
    #                     for metric_name in baseline_metrics_summary
    #                     if metric_name in tuned_metrics_summary
    #                 },
    #                 'tuned_metadata': dict(tuned_metadata),
    #                 'tuned_forecast': [float(value) for value in tuned_forecast.tolist()],
    #                 'tuned_adapter_family': adapter_name_to_family(tuned_model_spec.adapter_name),
    #                 'regime_diagnostics': regime_diagnostics.to_dict(),
    #                 'routing_recommendation': routing_recommendation.to_dict(),
    #             },
    #         }
    #     except Exception as exc:
    #         return {
    #             **baseline_metadata,
    #             'stage_tuning_comparison_error': str(exc),
    #         }

    # def _append_failed_run_record(
    #         self,
    #         *,
    #         model_spec: ModelSpec,
    #         model: ForecastingModelAdapter,
    #         series_record: ForecastingSeriesRecord,
    #         status: RunStatus,
    #         message: str,
    #         regime_diagnostics,
    #         routing_recommendation,
    # ) -> None:
    #     self.run_records.append(
    #         BenchmarkRunRecord(
    #             run_id=self.run_id,
    #             benchmark=series_record.benchmark,
    #             dataset_name=series_record.dataset_name,
    #             subset=series_record.subset,
    #             series_id=series_record.series_id,
    #             model_name=model.name,
    #             status=status,
    #             tags=model.tags,
    #             message=message,
    #             metadata=self._build_common_metadata(
    #                 model_spec=model_spec,
    #                 model=model,
    #                 regime_diagnostics=regime_diagnostics,
    #                 routing_recommendation=routing_recommendation,
    #                 extra={'optional': model.optional},
    #             ),
    #         )
    #     )

    # def _evaluate_series(
    #         self,
    #         model_spec: ModelSpec,
    #         model: ForecastingModelAdapter,
    #         series_record: ForecastingSeriesRecord,
    #         regime_diagnostics,
    #         routing_recommendation,
    # ) -> None:
    #     prediction, metadata = model.forecast(series_record)
    #     actual, forecast = self._validate_forecast_length(series_record, prediction)
    #     metrics_summary, event_metrics = self._record_metric_bundle(
    #         series_record=series_record,
    #         model_name=model.name,
    #         actual=actual,
    #         forecast=forecast,
    #         metadata=metadata,
    #     )
    #     self._record_predictions(
    #         series_record=series_record,
    #         model_name=model.name,
    #         actual=actual,
    #         forecast=forecast,
    #     )
    #     metadata = self._maybe_run_post_fit_tuning_comparison(
    #         model_spec=model_spec,
    #         model=model,
    #         series_record=series_record,
    #         baseline_metadata=metadata,
    #         baseline_metrics_summary=metrics_summary,
    #         regime_diagnostics=regime_diagnostics,
    #         routing_recommendation=routing_recommendation,
    #     )
    #     self.run_records.append(
    #         BenchmarkRunRecord(
    #             run_id=self.run_id,
    #             benchmark=series_record.benchmark,
    #             dataset_name=series_record.dataset_name,
    #             subset=series_record.subset,
    #             series_id=series_record.series_id,
    #             model_name=model.name,
    #             status=RunStatus.SUCCESS,
    #             tags=model.tags,
    #             metrics_summary=metrics_summary,
    #             metadata=self._build_common_metadata(
    #                 model_spec=model_spec,
    #                 model=model,
    #                 regime_diagnostics=regime_diagnostics,
    #                 routing_recommendation=routing_recommendation,
    #                 extra={
    #                     **metadata,
    #                     'active_forecast_steps': int(event_metrics.get('active_forecast_steps', 0)),
    #                     'calm_forecast_steps': int(event_metrics.get('calm_forecast_steps', 0)),
    #                 },
    #             ),
    #         )
    #     )


def run_anomaly_detection_suite(config: BenchmarkSuiteConfig) -> DetectionBenchmarkResult:  #надо реализовать в core.py
    return DetectionSuiteRunner(config).run_suite()
