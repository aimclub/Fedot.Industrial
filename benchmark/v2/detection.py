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

from fedot_ind.core.models.detection.progress_policy import (
    DetectionProgressPolicy,
    resolve_detection_progress_policy,
)

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
from benchmark.v2.verbosity import (
    DetectionVerbosityPolicy,
    resolve_detection_verbosity_policy,
)
from benchmark.v2.progress import BenchmarkProgressMonitor
from benchmark.v2.incremental_persistence import DetectionIncrementalPersistenceCoordinator


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

# Надо разобраться для чего это нужно и в задаче обнаружения аномалий
# def _parse_sequence_records(
#         payload: list[dict[str, Any]],
#         *,
#         benchmark: str,
#         dataset_name: str,
#         subset: str,
#         default_frequency: str,
#         default_horizon: int,
#         default_seasonal_period: int,
# ) -> list[ForecastingSeriesRecord]:
#     records: list[ForecastingSeriesRecord] = []
#     for index, item in enumerate(payload):
#         series_id = str(item.get('series_id', item.get('unique_id', f'{dataset_name}_{index}')))
#         frequency = str(item.get('frequency', default_frequency))
#         horizon = int(item.get('horizon', default_horizon))
#         seasonal_period = int(item.get('seasonal_period', default_seasonal_period))
#         values = item.get('values')
#         train = item.get('train_values')
#         test = item.get('test_values')
#         if values is not None:
#             train_values, test_values = _series_split_from_full_values(np.asarray(values, dtype=float), horizon)
#         elif train is not None and test is not None:
#             train_values = tuple(float(value) for value in np.asarray(train, dtype=float).reshape(-1))
#             test_values = tuple(float(value) for value in np.asarray(test, dtype=float).reshape(-1))
#         else:
#             raise BenchmarkConfigurationError('Sequence payload must include values or train_values/test_values.')

#         records.append(
#             ForecastingSeriesRecord(
#                 benchmark=benchmark,
#                 dataset_name=str(item.get('dataset_name', dataset_name)),
#                 subset=subset,
#                 series_id=series_id,
#                 frequency=frequency,
#                 forecast_horizon=horizon,
#                 seasonal_period=seasonal_period,
#                 train_values=train_values,
#                 test_values=test_values,
#                 metadata={'split_provenance': item.get('split_provenance', 'adapter_provided')},
#             )
#         )
#     return records

# def _sample_records(
#         records: list[DetectionSeriesRecord],
#         spec: DatasetSpec,
# ) -> tuple[DetectionSeriesRecord, ...]:
#     filtered = records
#     if spec.series_ids:
#         requested = set(spec.series_ids)
#         filtered = [record for record in filtered if record.series_id in requested]
#     if spec.sample_size is not None and len(filtered) > spec.sample_size:
#         rng = np.random.default_rng(spec.random_seed)
#         indices = rng.choice(len(filtered), size=spec.sample_size, replace=False)
#         filtered = [filtered[index] for index in sorted(indices)]
#     return tuple(filtered)

# функции хелперы для работы с датасет адаптерами и _resolve_stage_tuning_split_spec()
class SKABAdapter:
    """чтение файла датасета skab и приведение к единому формату.
    Адапрер возвращает record (один временной ряд, у которого есть train/test и horizon).
    После adapter-а runner работает одинаково с любым датасетом.
    """
    benchmark_name = 'skab'
    def __init__():
        """Инициализирует self.loader входным в __init__ или _default_loader()
        Также делает нормализацию и дополнительное семплирование (вообще разные доп штуки)
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

class InMemoryDetectionAdapter:
    """Dataset adapter backed by records provided directly in DatasetSpec."""

    benchmark_name = 'in_memory'

    def load_series(self, spec: DatasetSpec) -> tuple[DetectionSeriesRecord, ...]:
        """Convert in-memory payload records into DetectionSeriesRecord values."""
        payload = list(spec.adapter_options.get('records', ()))
        if not payload:
            raise BenchmarkConfigurationError('InMemory adapter requires adapter_options["records"].')
        records = _parse_sequence_records(
            payload=payload,
            benchmark=self.benchmark_name,
            dataset_name=spec.dataset_name,
            subset=spec.subset,
            default_frequency=str(spec.adapter_options.get('frequency', spec.subset)),
            default_horizon=int(spec.adapter_options.get('forecast_horizon', 1)),
            default_seasonal_period=int(spec.adapter_options.get('seasonal_period', 1)),
        )
        return _sample_records(records, spec)

# Используется только в OptionalExternalModel() и всё
def _safe_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False

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

class OptionalExternalModel(DetectionModelAdapter):
    """Placeholder adapter for optional external detection backends."""

    dependency_name: str
    name: str
    tags: tuple[str, ...] = ('baseline', 'detection', 'external')
    optional: bool = True
    scaffold_reason: str = 'Adapter scaffold is registered but backend training is not wired yet.'

    def availability(self) -> tuple[RunStatus, str]:
        """Check whether the optional external dependency can be imported."""
        if not _safe_import(self.dependency_name):
            return RunStatus.NOT_AVAILABLE, f'{self.dependency_name} is not installed.'
        return RunStatus.SUCCESS, 'dependency is available'

    def detect(self, series_record: DetectionSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Raise a skipped execution status until the external backend is wired."""
        del series_record
        raise ModelExecutionError(RunStatus.SKIPPED, self.scaffold_reason)

def build_dataset_adapter(spec: DatasetSpec):
    """выбор адаптера"""
    benchmark = spec.benchmark.lower()
    custom_loader = spec.adapter_options.get('loader')
    if benchmark == 'skab':
        return SKABAdapter(loader=custom_loader)
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

@dataclass
class DetectionSeriesArtifactsRecorder:
    """Collect per-series metric and prediction artifacts during a run."""

    run_id: str
    metric_names: tuple[str, ...]
    metric_records: list[MetricRecord]
    prediction_records: list[PredictionRecord]

    def validate_forecast_length(
            self,
            series_record: ForecastingSeriesRecord,
            prediction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate model forecast length and return aligned actual/forecast arrays."""
        actual = np.asarray(series_record.test_values, dtype=float)
        forecast = np.asarray(prediction, dtype=float).reshape(-1)[: len(actual)]
        if len(forecast) != len(actual):
            raise ModelExecutionError(
                RunStatus.FAILED,
                f'Model returned {len(forecast)} predictions for horizon {len(actual)}.',
            )
        return actual, forecast

    def record_metric_bundle(
            self,
            *,
            series_record: ForecastingSeriesRecord,
            model_name: str,
            actual: np.ndarray,
            forecast: np.ndarray,
            metadata: dict[str, Any],
            metric_name_suffix: str = '',
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Record aggregate, horizon-wise and event-aware metric rows."""
        metrics_summary: dict[str, float] = {}
        train = np.asarray(series_record.train_values, dtype=float)
        for metric_name in self.metric_names:
            metric_value = compute_forecasting_metric(
                metric_name,
                actual,
                forecast,
                train,
                series_record.seasonal_period,
            )
            metrics_summary[metric_name] = metric_value
            self.metric_records.append(
                MetricRecord(
                    run_id=self.run_id,
                    benchmark=series_record.benchmark,
                    dataset_name=series_record.dataset_name,
                    subset=series_record.subset,
                    series_id=series_record.series_id,
                    model_name=model_name,
                    metric_name=f'{metric_name}{metric_name_suffix}',
                    metric_value=metric_value,
                    status=RunStatus.SUCCESS,
                )
            )
            pointwise = compute_pointwise_metric(
                metric_name,
                actual,
                forecast,
                train,
                series_record.seasonal_period,
            )
            for horizon_index, pointwise_value in enumerate(pointwise, start=1):
                self.metric_records.append(
                    MetricRecord(
                        run_id=self.run_id,
                        benchmark=series_record.benchmark,
                        dataset_name=series_record.dataset_name,
                        subset=series_record.subset,
                        series_id=series_record.series_id,
                        model_name=model_name,
                        metric_name=f'{metric_name}{metric_name_suffix}',
                        metric_value=float(pointwise_value),
                        status=RunStatus.SUCCESS,
                        horizon_index=horizon_index,
                    )
                )

        event_metrics = _append_event_interval_metrics(
            self.metric_records,
            run_id=self.run_id,
            series_record=series_record,
            model_name=model_name,
            actual=actual,
            forecast=forecast,
            metadata=metadata,
            metric_name_suffix=metric_name_suffix,
        )
        metrics_summary.update({key: value for key, value in event_metrics.items() if key.startswith('mae_')})
        return metrics_summary, event_metrics

    def record_predictions(
            self,
            *,
            series_record: ForecastingSeriesRecord,
            model_name: str,
            actual: np.ndarray,
            forecast: np.ndarray,
    ) -> None:
        """Append per-horizon prediction records for one series/model item."""
        for horizon_index, (actual_value, forecast_value) in enumerate(zip(actual, forecast), start=1):
            self.prediction_records.append(
                PredictionRecord(
                    run_id=self.run_id,
                    benchmark=series_record.benchmark,
                    dataset_name=series_record.dataset_name,
                    subset=series_record.subset,
                    series_id=series_record.series_id,
                    model_name=model_name,
                    horizon_index=horizon_index,
                    y_true=float(actual_value),
                    y_pred=float(forecast_value),
                    status=RunStatus.SUCCESS,
                )
            )


@dataclass
class DetectionPostFitTuningCoordinator:
    """Run mandatory post-fit stage tuning and compare tuned vs baseline metrics."""

    config: BenchmarkSuiteConfig
    verbosity_policy: DetectionVerbosityPolicy
    artifacts_recorder: DetectionSeriesArtifactsRecorder

    def _build_tuned_model_spec(self, model_spec: ModelSpec, best_parameters: dict[str, Any]) -> ModelSpec:
        tuned_params = {
            **dict(model_spec.params),
            **dict(best_parameters),
        }
        tuned_params.pop('stage_tuning_runtime', None)
        return ModelSpec(
            adapter_name=model_spec.adapter_name,
            display_name=model_spec.display_name,
            tags=model_spec.tags,
            optional=model_spec.optional,
            params=tuned_params,
        )

    def _resolve_runtime_config(
            self,
            model_spec: ModelSpec,
            baseline_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        raw_config = dict(model_spec.params.get('stage_tuning_runtime') or {})
        metadata_runtime = dict(baseline_metadata.get('stage_tuning_runtime') or {})
        raw_progress_policy = raw_config.get('progress_policy', metadata_runtime.get('progress_policy'))
        raw_verbosity_policy = raw_config.get('verbosity_policy', self.verbosity_policy.to_dict())
        if raw_progress_policy is None:
            resolved_progress_policy = resolve_forecasting_progress_policy(
                None,
                show_progress=self.config.run_spec.show_progress,
            )
        else:
            resolved_progress_policy = resolve_forecasting_progress_policy(raw_progress_policy)
        resolved_verbosity_policy = (
            raw_verbosity_policy
            if isinstance(raw_verbosity_policy, ForecastingVerbosityPolicy)
            else resolve_forecasting_verbosity_policy(
                (raw_verbosity_policy or {}).get('level') if isinstance(raw_verbosity_policy, dict)
                else raw_verbosity_policy,
                options=raw_verbosity_policy if isinstance(raw_verbosity_policy, dict) else None,
            )
        )
        return {
            'metric_name': str(
                raw_config.get(
                    'metric_name',
                    metadata_runtime.get(
                        'metric_name',
                        self.config.run_spec.primary_metric))),
            'stage_updates': raw_config.get('stage_updates'),
            'max_values_per_parameter': int(
                raw_config.get(
                    'max_values_per_parameter',
                    3)),
            'max_stage_candidates': int(
                raw_config.get(
                    'max_stage_candidates',
                    16)),
            'split_spec': _resolve_stage_tuning_split_spec(raw_config),
            'progress_policy': resolved_progress_policy,
            'verbosity_policy': resolved_verbosity_policy,
        }

    def run(
            self,
            *,
            model_spec: ModelSpec,
            model: ForecastingModelAdapter,
            series_record: ForecastingSeriesRecord,
            baseline_metadata: dict[str, Any],
            baseline_metrics_summary: dict[str, float],
            regime_diagnostics,
            routing_recommendation,
    ) -> dict[str, Any]:
        """Execute stage tuning after baseline validation and return enriched metadata."""
        runtime_config = self._resolve_runtime_config(model_spec, baseline_metadata)
        if run_forecasting_stage_tuning_on_series is None:
            return {
                **baseline_metadata,
                'stage_tuning_report_error': 'stage_tuning_runtime is unavailable in the current environment.',
            }

        try:
            report = run_forecasting_stage_tuning_on_series(
                model_spec.adapter_name,
                time_series=np.asarray(series_record.train_values, dtype=float),
                forecast_horizon=series_record.forecast_horizon,
                base_params={
                    key: value
                    for key, value in dict(model_spec.params).items()
                    if key != 'stage_tuning_runtime'
                },
                stage_updates=runtime_config.get('stage_updates'),
                metric_name=str(runtime_config['metric_name']),
                split_spec=runtime_config.get('split_spec'),
                seasonal_period=series_record.seasonal_period,
                max_values_per_parameter=int(runtime_config['max_values_per_parameter']),
                max_stage_candidates=int(runtime_config['max_stage_candidates']),
                progress_policy=runtime_config.get('progress_policy'),
            )
            verbosity_policy = runtime_config.get('verbosity_policy', self.verbosity_policy)
            report_dict = verbosity_policy.prune_stage_tuning_report(report.to_dict()) or {}
            sequential_result = dict(report_dict.get('sequential_result') or {})
            best_parameters = dict(sequential_result.get('best_parameters') or {})
            enriched_baseline_metadata = {
                **baseline_metadata,
                'stage_tuning_report': report_dict,
                'stage_tuning_runtime': verbosity_policy.prune_stage_tuning_runtime({
                    'enabled': True,
                    'metric_name': str(runtime_config['metric_name']),
                    'max_values_per_parameter': int(runtime_config['max_values_per_parameter']),
                    'max_stage_candidates': int(runtime_config['max_stage_candidates']),
                    'improved': report.metadata.get('improved'),
                    'baseline_score': report.metadata.get('baseline_score'),
                    'best_score': report.metadata.get('best_score'),
                    'progress_policy': resolve_forecasting_progress_policy(
                        runtime_config.get('progress_policy'),
                        show_progress=self.config.run_spec.show_progress,
                    ).to_dict(),
                }) or {},
            }
            if not best_parameters:
                return enriched_baseline_metadata

            tuned_model_spec = self._build_tuned_model_spec(model_spec, best_parameters)
            tuned_model = build_model_adapter(tuned_model_spec)
            tuned_status, tuned_message = tuned_model.availability()
            if tuned_status is not RunStatus.SUCCESS:
                return {
                    **enriched_baseline_metadata,
                    'stage_tuning_comparison_error': tuned_message,
                }

            tuned_prediction, tuned_metadata = tuned_model.forecast(series_record)
            actual, tuned_forecast = self.artifacts_recorder.validate_forecast_length(series_record, tuned_prediction)
            tuned_metrics_summary, tuned_event_metrics = self.artifacts_recorder.record_metric_bundle(
                series_record=series_record,
                model_name=model.name,
                actual=actual,
                forecast=tuned_forecast,
                metadata=tuned_metadata,
                metric_name_suffix='_tuned',
            )
            tuned_metrics_summary.update(
                {key: value for key, value in tuned_event_metrics.items() if key.startswith('mae_')}
            )
            comparison_payload = verbosity_policy.prune_stage_tuning_comparison(
                {
                    'best_parameters': best_parameters,
                    'baseline_metrics': baseline_metrics_summary,
                    'tuned_metrics': tuned_metrics_summary,
                    'improved_metrics': {
                        metric_name: bool(
                            tuned_metrics_summary.get(metric_name, math.inf)
                            <= baseline_metrics_summary.get(metric_name, math.inf)
                        )
                        for metric_name in baseline_metrics_summary
                        if metric_name in tuned_metrics_summary
                    },
                    'absolute_gain': {
                        metric_name: float(baseline_metrics_summary[metric_name] - tuned_metrics_summary[metric_name])
                        for metric_name in baseline_metrics_summary
                        if metric_name in tuned_metrics_summary
                    },
                    'tuned_metadata': dict(tuned_metadata),
                    'tuned_forecast': [float(value) for value in tuned_forecast.tolist()],
                    'tuned_adapter_family': adapter_name_to_family(tuned_model_spec.adapter_name),
                    'regime_diagnostics': regime_diagnostics.to_dict(),
                    'routing_recommendation': routing_recommendation.to_dict(),
                }
            )
            return {
                **enriched_baseline_metadata,
                **({'stage_tuning_comparison': comparison_payload} if comparison_payload is not None else {}),
            }
        except Exception as exc:
            return {
                **baseline_metadata,
                'stage_tuning_comparison_error': str(exc),
            }


def build_leaderboard(
        run_records: tuple[BenchmarkRunRecord, ...],
        primary_metric: str,
) -> BenchmarkAggregateReport:
    """Aggregate successful run records into a benchmark leaderboard."""
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
    """Orchestrate detection benchmark datasets, models, series and artifacts."""

    def __init__(self, config: BenchmarkSuiteConfig):
        """Initialize benchmark state, progress, verbosity and resume coordinators."""
        validate_detection_suite_config(config)
        self.config = config
        self.run_id = (
            DetectionIncrementalPersistenceCoordinator.resolve_run_id(config)
            or new_run_id(config.run_spec.run_name)
        )
        self.series_records: list[DetectionSeriesRecord] = []
        self.run_records: list[BenchmarkRunRecord] = []
        self.prediction_records: list[PredictionRecord] = []
        self.metric_records: list[MetricRecord] = []
        self.known_series_keys: set[tuple[str, str, str, str]] = set()
        self.resumed_item_keys: set[str] = set()
        self.progress_policy = resolve_detection_progress_policy(
            DetectionProgressPolicy(
                enabled=bool(config.run_spec.show_progress),
                leave=bool(config.run_spec.progress_leave),
                stage_tuning_enabled=bool(config.run_spec.show_progress),
                head_training_enabled=bool(config.run_spec.show_progress),
            ),
            show_progress=config.run_spec.show_progress,
        )
        self.verbosity_policy = resolve_detection_progress_policy(
            config.run_spec.verbosity,
            options=config.run_spec.verbosity_options,
        )
        self.artifacts_recorder = DetectionSeriesArtifactsRecorder(
            run_id=self.run_id,
            metric_names=tuple(self.config.metrics),
            metric_records=self.metric_records,
            prediction_records=self.prediction_records,
        )
        self.post_fit_tuning = DetectionPostFitTuningCoordinator(
            config=self.config,
            verbosity_policy=self.verbosity_policy,
            artifacts_recorder=self.artifacts_recorder,
        )
        self.progress = BenchmarkProgressMonitor(
            enabled=config.run_spec.show_progress,
            task_type=config.task_type.value,
            run_name=config.run_spec.run_name,
            leave=config.run_spec.progress_leave,
            log_errors=config.run_spec.progress_log_errors,
            log_summaries=config.run_spec.progress_log_summaries,
        )
        self.incremental_persistence = DetectionIncrementalPersistenceCoordinator(
            config=self.config,
            run_id=self.run_id,
        )
        self._load_resume_state()

    def _load_resume_state(self) -> None:
        resume_state = self.incremental_persistence.load_resume_state()
        if resume_state is None:
            return
        self.series_records = list(resume_state.series_records)
        self.run_records = list(resume_state.run_records)
        self.metric_records = list(resume_state.metric_records)
        self.prediction_records = list(resume_state.prediction_records)
        self.known_series_keys = {self._series_key(record) for record in self.series_records}
        self.resumed_item_keys = set(resume_state.item_artifact_paths)
        self.progress.seed_resume_state(
            completed_items=resume_state.completed_items,
            status_counts=resume_state.status_counts,
        )

    def _series_key(self, series_record: DetectionSeriesRecord) -> tuple[str, str, str, str]:
        return (
            str(series_record.benchmark),
            str(series_record.dataset_name),
            str(series_record.subset),
            str(series_record.series_id),
        )

    # def _augment_model_spec_with_progress_policy(self, model_spec: ModelSpec) -> ModelSpec:
    #     params = dict(model_spec.params)
    #     params.setdefault('progress_policy', self.progress_policy.to_dict())
    #     if isinstance(params.get('stage_tuning_runtime'), dict):
    #         params['stage_tuning_runtime'] = {
    #             **dict(params['stage_tuning_runtime']),
    #             'verbosity_policy': dict(
    #                 params['stage_tuning_runtime'].get('verbosity_policy', self.verbosity_policy.to_dict())
    #             ),
    #         }
    #     return ModelSpec(
    #         adapter_name=model_spec.adapter_name,
    #         display_name=model_spec.display_name,
    #         tags=model_spec.tags,
    #         optional=model_spec.optional,
    #         params=params,
    #     )

    # def _runner_context_metadata(self) -> dict[str, Any]:
    #     if not self.verbosity_policy.include_runner_context:
    #         return {}
    #     return {
    #         'benchmark_runtime_context': {
    #             'progress_policy': self.progress_policy.to_dict(),
    #             'verbosity_policy': self.verbosity_policy.to_dict(),
    #         }
    #     }

    def run_suite(self) -> DetectionBenchmarkResult:
        """Run the configured detection suite and return all collected records."""
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
            artifact_manifest=self.incremental_persistence.build_artifact_manifest(),
        )

    def _iter_over_datasets(self) -> None:
        for dataset_spec in self.config.datasets:
            dataset_series = self._load_dataset_series(dataset_spec)
            self._iter_over_models(dataset_spec, dataset_series)
            self.progress.dataset_finished()

    def _load_dataset_series(self, dataset_spec: DatasetSpec) -> tuple[DetectionSeriesRecord, ...]:
        dataset_adapter = build_dataset_adapter(dataset_spec)
        dataset_series = dataset_adapter.load_series(dataset_spec)
        for record in dataset_series:
            record_key = self._series_key(record)
            if record_key not in self.known_series_keys:
                self.series_records.append(record)
                self.known_series_keys.add(record_key)
        # self.incremental_persistence.persist_series_catalog(self.series_records)
        self.progress.extend_total(len(dataset_series) * len(self.config.models))
        self.progress.dataset_loaded(dataset_spec.dataset_name, len(dataset_series))
        return dataset_series

    def _iter_over_models(
            self,
            dataset_spec: DatasetSpec,
            dataset_series: tuple[DetectionSeriesRecord, ...],
    ) -> None:
        for model_spec in self.config.models:
            resolved_model_spec = self._augment_model_spec_with_progress_policy(model_spec)
            model = build_model_adapter(resolved_model_spec)
            self.progress.model_started(dataset_spec.dataset_name, model.name)
            try:
                availability_status, availability_message = model.availability()
                if availability_status is not RunStatus.SUCCESS:
                    self._handle_unavailable_model(
                        model_spec=resolved_model_spec,
                        model=model,
                        dataset_series=dataset_series,
                        availability_status=availability_status,
                        availability_message=availability_message,
                    )
                    continue
                self._iter_over_series(resolved_model_spec, model, dataset_series)
            finally:
                self.progress.model_finished()

    # def _iter_over_series(
    #         self,
    #         model_spec: ModelSpec,
    #         model: ForecastingModelAdapter,
    #         dataset_series: tuple[ForecastingSeriesRecord, ...],
    # ) -> None:
    #     for series_record in dataset_series:
    #         item_key = self.incremental_persistence.item_key(series_record, model.name)
    #         if item_key in self.resumed_item_keys:
    #             existing_record = next(
    #                 (
    #                     record for record in self.run_records
    #                     if record.benchmark == series_record.benchmark
    #                     and record.dataset_name == series_record.dataset_name
    #                     and record.subset == series_record.subset
    #                     and record.series_id == series_record.series_id
    #                     and record.model_name == model.name
    #                 ),
    #                 None,
    #             )
    #             self.progress.item_resumed(
    #                 series_record.dataset_name,
    #                 model.name,
    #                 series_record.series_id,
    #                 existing_record.status.value if existing_record is not None else 'success',
    #             )
    #             continue
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

    def _handle_unavailable_model(
            self,
            *,
            model_spec: ModelSpec,
            model: DetectionModelAdapter,
            dataset_series: tuple[DetectionSeriesRecord, ...],
            availability_status: RunStatus,
            availability_message: str,
    ) -> None:
        for series_record in dataset_series:
            item_key = self.incremental_persistence.item_key(series_record, model.name)
            if item_key in self.resumed_item_keys:
                existing_record = next(
                    (
                        record for record in self.run_records
                        if record.benchmark == series_record.benchmark
                        and record.dataset_name == series_record.dataset_name
                        and record.subset == series_record.subset
                        and record.series_id == series_record.series_id
                        and record.model_name == model.name
                    ),
                    None,
                )
                self.progress.item_resumed(
                    series_record.dataset_name,
                    model.name,
                    series_record.series_id,
                    existing_record.status.value if existing_record is not None else availability_status.value,
                )
                continue
            self.progress.item_started(series_record.dataset_name, model.name, series_record.series_id)
            regime_diagnostics, routing_recommendation = self._build_series_context(series_record)
            self.run_records.append(
                BenchmarkRunRecord(
                    run_id=self.run_id,
                    benchmark=series_record.benchmark,
                    dataset_name=series_record.dataset_name,
                    subset=series_record.subset,
                    series_id=series_record.series_id,
                    model_name=model.name,
                    status=availability_status,
                    tags=model.tags,
                    message=availability_message,
                    metadata=self._build_common_metadata(
                        model_spec=model_spec,
                        model=model,
                        regime_diagnostics=regime_diagnostics,
                        routing_recommendation=routing_recommendation,
                        extra={'optional': model.optional},
                    ),
                )
            )
            self.incremental_persistence.persist_item_result(
                series_record=series_record,
                run_record=self.run_records[-1],
            )
            self.progress.advance(availability_status.value, availability_message)

    # def _build_series_context(self, series_record: ForecastingSeriesRecord):
    #     regime_diagnostics = analyze_regime_diagnostics(np.asarray(series_record.train_values, dtype=float))
    #     routing_recommendation = recommend_forecasting_model(regime_diagnostics)
    #     return regime_diagnostics, routing_recommendation

    def _build_common_metadata(
            self,
            *,
            model_spec: ModelSpec,
            model: DetectionModelAdapter,
            regime_diagnostics,
            routing_recommendation,
            extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            'adapter_name': model_spec.adapter_name,
            'model_adapter_family': adapter_name_to_family(model_spec.adapter_name),
            'regime_diagnostics': regime_diagnostics.to_dict(),
            'routing_recommendation': routing_recommendation.to_dict(),
            'routing_recommendation_family': adapter_name_to_family(routing_recommendation.primary_adapter),
            **self._runner_context_metadata(),
            **dict(extra or {}),
        }

    # def _validate_forecast_length(
    #         self,
    #         series_record: ForecastingSeriesRecord,
    #         prediction: np.ndarray,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     return self.artifacts_recorder.validate_forecast_length(series_record, prediction)

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
    #     return self.artifacts_recorder.record_metric_bundle(
    #         series_record=series_record,
    #         model_name=model_name,
    #         actual=actual,
    #         forecast=forecast,
    #         metadata=metadata,
    #         metric_name_suffix=metric_name_suffix,
    #     )

    # def _record_predictions(
    #         self,
    #         *,
    #         series_record: ForecastingSeriesRecord,
    #         model_name: str,
    #         actual: np.ndarray,
    #         forecast: np.ndarray,
    # ) -> None:
    #     self.artifacts_recorder.record_predictions(
    #         series_record=series_record,
    #         model_name=model_name,
    #         actual=actual,
    #         forecast=forecast,
    #     )

    # def _build_tuned_model_spec(self, model_spec: ModelSpec, best_parameters: dict[str, Any]) -> ModelSpec:
    #     return self.post_fit_tuning._build_tuned_model_spec(model_spec, best_parameters)

    # def _resolve_post_fit_tuning_runtime_config(
    #         self,
    #         model_spec: ModelSpec,
    #         baseline_metadata: dict[str, Any],
    # ) -> dict[str, Any]:
    #     return self.post_fit_tuning._resolve_runtime_config(model_spec, baseline_metadata)

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
    #     return self.post_fit_tuning.run(
    #         model_spec=model_spec,
    #         model=model,
    #         series_record=series_record,
    #         baseline_metadata=baseline_metadata,
    #         baseline_metrics_summary=baseline_metrics_summary,
    #         regime_diagnostics=regime_diagnostics,
    #         routing_recommendation=routing_recommendation,
    #     )

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
    # ) -> BenchmarkRunRecord:
    #     run_record = BenchmarkRunRecord(
    #         run_id=self.run_id,
    #         benchmark=series_record.benchmark,
    #         dataset_name=series_record.dataset_name,
    #         subset=series_record.subset,
    #         series_id=series_record.series_id,
    #         model_name=model.name,
    #         status=status,
    #         tags=model.tags,
    #         message=message,
    #         metadata=self._build_common_metadata(
    #             model_spec=model_spec,
    #             model=model,
    #             regime_diagnostics=regime_diagnostics,
    #             routing_recommendation=routing_recommendation,
    #             extra={'optional': model.optional},
    #         ),
    #     )
    #     self.run_records.append(run_record)
    #     self.incremental_persistence.persist_item_result(
    #         series_record=series_record,
    #         run_record=run_record,
    #     )
    #     return run_record

    # def _evaluate_series(
    #         self,
    #         model_spec: ModelSpec,
    #         model: ForecastingModelAdapter,
    #         series_record: ForecastingSeriesRecord,
    #         regime_diagnostics,
    #         routing_recommendation,
    # ) -> None:
    #     metric_count_before = len(self.metric_records)
    #     prediction_count_before = len(self.prediction_records)
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
    #     run_record = BenchmarkRunRecord(
    #         run_id=self.run_id,
    #         benchmark=series_record.benchmark,
    #         dataset_name=series_record.dataset_name,
    #         subset=series_record.subset,
    #         series_id=series_record.series_id,
    #         model_name=model.name,
    #         status=RunStatus.SUCCESS,
    #         tags=model.tags,
    #         metrics_summary=metrics_summary,
    #         metadata=self._build_common_metadata(
    #             model_spec=model_spec,
    #             model=model,
    #             regime_diagnostics=regime_diagnostics,
    #             routing_recommendation=routing_recommendation,
    #             extra={
    #                 **metadata,
    #                 'active_forecast_steps': int(event_metrics.get('active_forecast_steps', 0)),
    #                 'calm_forecast_steps': int(event_metrics.get('calm_forecast_steps', 0)),
    #             },
    #         ),
    #     )
    #     self.run_records.append(run_record)
    #     self.incremental_persistence.persist_item_result(
    #         series_record=series_record,
    #         run_record=run_record,
    #         metric_records=tuple(self.metric_records[metric_count_before:]),
    #         prediction_records=tuple(self.prediction_records[prediction_count_before:]),
    #     )


def run_anomaly_detection_suite(config: BenchmarkSuiteConfig) -> DetectionBenchmarkResult:
    return DetectionSuiteRunner(config).run_suite()
