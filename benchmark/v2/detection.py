from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.detection.progress_policy import (
    DetectionProgressPolicy,
    resolve_detection_progress_policy,
)
from fedot_ind.core.models.detection.runtime import (
    DetectionSplitKind,
    DetectionSplitSpec,
    ensure_detection_array,
    infer_regime_segments,
)
from fedot_ind.core.models.detection.stage_tuning_runtime import run_detection_stage_tuning_on_series
from fedot_ind.core.operation.interfaces.detection_runtime_strategy import DETECTION_RUNTIME_MODELS
from fedot_ind.core.repository.detection_registry import (
    CANONICAL_STAGE_DETECTION_MODELS,
    canonical_detection_model_name,
    detection_family_for,
)
from fedot_ind.tools.serialisation.path_lib import EXAMPLES_DATA_PATH

from benchmark.v2.core import (
    BenchmarkAggregateReport,
    BenchmarkRunRecord,
    BenchmarkSuiteConfig,
    DatasetSpec,
    DetectionBenchmarkResult,
    DetectionSeriesRecord,
    LabelPredictionRecord,
    MetricRecord,
    ModelSpec,
    RunStatus,
    TaskType,
    new_run_id,
)
from benchmark.v2.incremental_persistence import DetectionIncrementalPersistenceCoordinator
from benchmark.v2.progress import BenchmarkProgressMonitor
from benchmark.v2.verbosity import (
    DetectionVerbosityPolicy,
    resolve_detection_verbosity_policy,
)

SUPPORTED_ANOMALY_DETECTION_METRICS = ('accuracy', 'balanced_accuracy', 'f1_macro')
_HIGHER_IS_BETTER_METRICS = frozenset(SUPPORTED_ANOMALY_DETECTION_METRICS)

try:  # pragma: no cover
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False


class BenchmarkConfigurationError(ValueError):
    pass


class ModelExecutionError(RuntimeError):
    def __init__(self, status: RunStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class DetectionRegimeDiagnostics:
    n_segments: int
    segment_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            'n_segments': self.n_segments,
            'segment_labels': list(self.segment_labels),
        }


@dataclass(frozen=True)
class DetectionRoutingContext:
    primary_adapter: str

    def to_dict(self) -> dict[str, Any]:
        return {'primary_adapter': self.primary_adapter, 'source': 'detection_family'}


def validate_detection_suite_config(config: BenchmarkSuiteConfig) -> None:
    if config.task_type is not TaskType.ANOMALY_DETECTION:
        raise BenchmarkConfigurationError('Anomaly detection suite expects task_type=anomaly_detection.')
    if not config.datasets:
        raise BenchmarkConfigurationError('Benchmark suite must contain at least one dataset spec.')
    if not config.models:
        raise BenchmarkConfigurationError('Benchmark suite must contain at least one model spec.')
    unsupported = set(config.metrics) - set(SUPPORTED_ANOMALY_DETECTION_METRICS)
    if unsupported:
        raise BenchmarkConfigurationError(f'Unsupported anomaly detection metrics: {sorted(unsupported)}')


def _parse_detection_records(
        payload: list[dict[str, Any]],
        *,
        benchmark: str,
        dataset_name: str,
        subset: str,
) -> list[DetectionSeriesRecord]:
    records: list[DetectionSeriesRecord] = []
    for index, item in enumerate(payload):
        series_id = str(item.get('series_id', f'{dataset_name}_{index}'))
        values = item.get('values')
        target = item.get('target')
        train_values = item.get('train_values')
        test_values = item.get('test_values')
        test_target = item.get('test_target', item.get('target'))
        metadata = dict(item.get('metadata', {}))

        if values is not None:
            matrix = np.asarray(values, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            target_array = target
            if target_array is None:
                raise BenchmarkConfigurationError('Detection payload with values requires target labels.')
            record_target = tuple(int(value) for value in np.asarray(target_array, dtype=int).reshape(-1))
            record_values = tuple(tuple(float(value) for value in row) for row in matrix)
        elif train_values is not None and test_values is not None:
            train_matrix = np.asarray(train_values, dtype=float)
            test_matrix = np.asarray(test_values, dtype=float)
            if train_matrix.ndim == 1:
                train_matrix = train_matrix.reshape(-1, 1)
            if test_matrix.ndim == 1:
                test_matrix = test_matrix.reshape(-1, 1)
            if test_target is None:
                raise BenchmarkConfigurationError('Legacy detection payload requires test_target.')
            metadata.setdefault('train_values', train_matrix.tolist())
            metadata.setdefault(
                'train_labels',
                [0] * train_matrix.shape[0],
            )
            metadata['split_mode'] = metadata.get('split_mode', 'legacy_pair')
            record_values = tuple(tuple(float(value) for value in row) for row in test_matrix)
            record_target = tuple(int(value) for value in np.asarray(test_target, dtype=int).reshape(-1))
        else:
            raise BenchmarkConfigurationError('Detection payload must include values/target or train_values/test_values.')

        timestamps = tuple(str(value) for value in item.get('timestamps', ()))
        records.append(
            DetectionSeriesRecord(
                benchmark=benchmark,
                dataset_name=str(item.get('dataset_name', dataset_name)),
                subset=subset,
                series_id=series_id,
                values=record_values,
                target=record_target,
                timestamps=timestamps,
                metadata={**metadata, 'split_provenance': item.get('split_provenance', 'adapter_provided')},
            )
        )
    return records


def _sample_records(
        records: list[DetectionSeriesRecord],
        spec: DatasetSpec,
) -> tuple[DetectionSeriesRecord, ...]:
    filtered = records
    if spec.series_ids:
        requested = set(spec.series_ids)
        filtered = [record for record in filtered if record.series_id in requested]
    if spec.sample_size is not None and len(filtered) > spec.sample_size:
        rng = np.random.default_rng(spec.random_seed)
        indices = rng.choice(len(filtered), size=spec.sample_size, replace=False)
        filtered = [filtered[index] for index in sorted(indices)]
    return tuple(filtered)


def _record_values_array(series_record: DetectionSeriesRecord) -> np.ndarray:
    matrix = np.asarray(series_record.values, dtype=float)
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)
    return matrix


def _record_labels_array(series_record: DetectionSeriesRecord) -> np.ndarray:
    if series_record.target is None:
        raise BenchmarkConfigurationError(f'Series {series_record.series_id} is missing target labels.')
    return np.asarray(series_record.target, dtype=int).reshape(-1)


def _train_fit_values(series_record: DetectionSeriesRecord) -> np.ndarray:
    metadata = dict(series_record.metadata)
    train_values = metadata.get('train_values')
    if train_values is not None:
        matrix = np.asarray(train_values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        return matrix
    values = _record_values_array(series_record)
    train_fraction = float(metadata.get('train_fraction', 0.7))
    split_index = max(1, int(round(values.shape[0] * train_fraction)))
    return values[:split_index]


def _resolve_stage_tuning_split_spec(raw: dict[str, Any] | None) -> DetectionSplitSpec | None:
    if not raw:
        return None
    split_spec = raw.get('split_spec')
    if isinstance(split_spec, DetectionSplitSpec):
        return split_spec
    if isinstance(split_spec, dict):
        kind = split_spec.get('kind', DetectionSplitKind.TEMPORAL.value)
        return DetectionSplitSpec(
            kind=DetectionSplitKind(str(kind).lower()),
            train_fraction=float(split_spec.get('train_fraction', 0.7)),
            calibration_fraction=float(split_spec.get('calibration_fraction', 0.15)),
            random_seed=int(split_spec.get('random_seed', 0)),
            prevent_future_leakage=bool(split_spec.get('prevent_future_leakage', True)),
            target_domain=split_spec.get('target_domain'),
        )
    return DetectionSplitSpec(
        train_fraction=float(raw.get('train_fraction', 0.7)),
        calibration_fraction=float(raw.get('calibration_fraction', 0.15)),
        random_seed=int(raw.get('random_seed', 0)),
    )


# ADAPTED: SKAB dual-mode loader (legacy_pair + single_series).
class SKABAdapter:
    """Load SKAB CSV series as DetectionSeriesRecord values.

    split_mode:
    - legacy_pair (default): anomaly-free train CSV + labeled test CSV
    - single_series: one CSV with temporal train/test split via train_data_size
    """

    benchmark_name = 'skab'

    def __init__(self, loader: Callable[..., Any] | None = None):
        self._loader = loader or self._default_loader()

    def load_series(self, spec: DatasetSpec) -> tuple[DetectionSeriesRecord, ...]:
        split_mode = str(spec.adapter_options.get('split_mode', 'legacy_pair')).lower()
        folder = str(spec.adapter_options.get('folder', 'valve1'))
        train_data_size = spec.adapter_options.get('train_data_size', 'anomaly-free')
        data_root = Path(spec.adapter_options.get('data_root', EXAMPLES_DATA_PATH)) / 'benchmark' / 'detection' / 'data'
        series_ids = list(spec.series_ids) or [
            path.stem for path in sorted((data_root / folder).glob('*.csv'))
        ]
        records: list[DetectionSeriesRecord] = []
        for series_id in series_ids:
            if split_mode == 'single_series':
                records.append(
                    self._load_single_series_record(
                        data_root=data_root,
                        folder=folder,
                        series_id=series_id,
                        spec=spec,
                        train_data_size=train_data_size,
                    )
                )
            else:
                records.append(
                    self._load_legacy_pair_record(
                        folder=folder,
                        series_id=series_id,
                        spec=spec,
                        train_data_size=train_data_size,
                    )
                )
        return _sample_records(records, spec)

    def _load_legacy_pair_record(
            self,
            *,
            folder: str,
            series_id: str,
            spec: DatasetSpec,
            train_data_size: Any,
    ) -> DetectionSeriesRecord:
        train_pair, test_pair = self._loader({
            'benchmark': folder,
            'dataset': series_id,
            'train_data_size': train_data_size,
        })
        train_values, train_labels = train_pair
        test_values, test_labels = test_pair
        train_matrix = np.asarray(train_values, dtype=float)
        test_matrix = np.asarray(test_values, dtype=float)
        if train_matrix.ndim == 1:
            train_matrix = train_matrix.reshape(-1, 1)
        if test_matrix.ndim == 1:
            test_matrix = test_matrix.reshape(-1, 1)
        return DetectionSeriesRecord(
            benchmark=self.benchmark_name,
            dataset_name=spec.dataset_name,
            subset=spec.subset,
            series_id=series_id,
            values=tuple(tuple(float(value) for value in row) for row in test_matrix),
            target=tuple(int(value) for value in np.asarray(test_labels, dtype=int).reshape(-1)),
            metadata={
                'split_mode': 'legacy_pair',
                'folder': folder,
                'train_data_size': train_data_size,
                'train_values': train_matrix.tolist(),
                'train_labels': [int(value) for value in np.asarray(train_labels, dtype=int).reshape(-1)],
            },
        )

    def _load_single_series_record(
            self,
            *,
            data_root: Path,
            folder: str,
            series_id: str,
            spec: DatasetSpec,
            train_data_size: Any,
    ) -> DetectionSeriesRecord:
        csv_path = data_root / folder / f'{series_id}.csv'
        df = pd.read_csv(csv_path, index_col='datetime', sep=';', parse_dates=True)
        feature_columns = [column for column in df.columns if column not in {'anomaly', 'changepoint'}]
        values = df[feature_columns].to_numpy(dtype=float)
        labels = df['anomaly'].to_numpy(dtype=int)
        if isinstance(train_data_size, int):
            train_values = values[:train_data_size]
            test_values = values[train_data_size:]
            test_labels = labels[train_data_size:]
            metadata = {
                'split_mode': 'single_series',
                'folder': folder,
                'train_data_size': train_data_size,
                'train_values': train_values.tolist(),
                'train_labels': [0] * train_values.shape[0],
            }
            values = test_values
            labels = test_labels
        else:
            metadata = {'split_mode': 'single_series', 'folder': folder, 'train_fraction': 0.7}
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        return DetectionSeriesRecord(
            benchmark=self.benchmark_name,
            dataset_name=spec.dataset_name,
            subset=spec.subset,
            series_id=series_id,
            values=tuple(tuple(float(value) for value in row) for row in matrix),
            target=tuple(int(value) for value in labels.reshape(-1)),
            timestamps=tuple(str(value) for value in df.index.astype(str)),
            metadata=metadata,
        )

    @staticmethod
    def _default_loader() -> Callable[..., Any]:
        from fedot_ind.tools.loader import DataLoader

        return DataLoader(dataset_name={}).load_detection_data


class MPSIAdapter:
    """Stub adapter; use adapter_options['records'] until raw MPSI archives are wired."""

    benchmark_name = 'mpsi'

    def load_series(self, spec: DatasetSpec) -> tuple[DetectionSeriesRecord, ...]:
        payload = list(spec.adapter_options.get('records', ()))
        if not payload:
            raise BenchmarkConfigurationError(
                'MPSI adapter is not wired to archives yet; provide adapter_options["records"] for stub runs.',
            )
        records = _parse_detection_records(
            payload=payload,
            benchmark=self.benchmark_name,
            dataset_name=spec.dataset_name,
            subset=spec.subset,
        )
        return _sample_records(records, spec)


class InMemoryDetectionAdapter:
    """Dataset adapter backed by records provided directly in DatasetSpec.adapter_options."""

    benchmark_name = 'in_memory'

    def load_series(self, spec: DatasetSpec) -> tuple[DetectionSeriesRecord, ...]:
        payload = list(spec.adapter_options.get('records', ()))
        if not payload:
            raise BenchmarkConfigurationError('InMemory adapter requires adapter_options["records"].')
        records = _parse_detection_records(
            payload=payload,
            benchmark=self.benchmark_name,
            dataset_name=spec.dataset_name,
            subset=spec.subset,
        )
        return _sample_records(records, spec)


def _safe_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def compute_detection_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=int).reshape(-1)
    predicted = np.asarray(y_pred, dtype=int).reshape(-1)
    labels = sorted(set(actual.tolist()) | set(predicted.tolist()))
    if metric_name == 'accuracy':
        return float(np.mean(actual == predicted))
    recalls = []
    f1_scores = []
    for label in labels:
        true_positive = int(np.sum((actual == label) & (predicted == label)))
        false_positive = int(np.sum((actual != label) & (predicted == label)))
        false_negative = int(np.sum((actual == label) & (predicted != label)))
        label_support = int(np.sum(actual == label))
        recall = true_positive / label_support if label_support else 0.0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        recalls.append(recall)
        f1_scores.append(f1)
    if metric_name == 'balanced_accuracy':
        return float(np.mean(recalls)) if recalls else 0.0
    if metric_name == 'f1_macro':
        return float(np.mean(f1_scores)) if f1_scores else 0.0
    raise BenchmarkConfigurationError(f'Unsupported detection metric: {metric_name}')


@dataclass
class ConstantZeroDetectionModel:
    name: str = 'ConstantZero'
    tags: tuple[str, ...] = ('baseline', 'detection', 'naive')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        return RunStatus.SUCCESS, 'ready'

    def detect(self, series_record: DetectionSeriesRecord) -> tuple[dict[str, Any], dict[str, Any]]:
        labels = np.zeros(_record_labels_array(series_record).shape[0], dtype=int)
        return {'labels': labels}, {'model': self.name}


@dataclass
class RuntimeDetectionModelAdapter:
    adapter_name: str
    display_name: str
    params: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ('detection', 'runtime')
    optional: bool = False
    name: str = ''

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.display_name

    def availability(self) -> tuple[RunStatus, str]:
        canonical = canonical_detection_model_name(self.adapter_name)
        if canonical in {'conv_autoencoder_detector', 'tcn_autoencoder_detector'} and not _TORCH_AVAILABLE:
            return RunStatus.NOT_AVAILABLE, 'torch is required for neural anomaly detectors.'
        if canonical not in DETECTION_RUNTIME_MODELS:
            return RunStatus.NOT_AVAILABLE, f'Unsupported runtime detector: {self.adapter_name}'
        return RunStatus.SUCCESS, 'ready'

    def detect(self, series_record: DetectionSeriesRecord) -> tuple[dict[str, Any], dict[str, Any]]:
        canonical = canonical_detection_model_name(self.adapter_name)
        detector_cls = DETECTION_RUNTIME_MODELS[canonical]
        fit_values = _train_fit_values(series_record)
        eval_values = _record_values_array(series_record)
        detector = detector_cls(OperationParameters(**dict(self.params)))
        detector.fit(InputData(features=fit_values))
        score_series = detector.score_series_on_values(eval_values)
        labels = np.asarray(score_series.labels, dtype=int).reshape(-1)
        scores = np.asarray(score_series.scores, dtype=float).reshape(-1)
        metadata = {
            'canonical_name': canonical,
            'family': detection_family_for(canonical),
            'threshold': float(score_series.threshold),
            'stage_diagnostics': detector.get_stage_diagnostics(),
        }
        return {'labels': labels, 'scores': scores}, metadata


class OptionalExternalModel:
    dependency_name: str
    name: str
    tags: tuple[str, ...] = ('baseline', 'detection', 'external')
    optional: bool = True
    scaffold_reason: str = 'Adapter scaffold is registered but backend training is not wired yet.'

    def availability(self) -> tuple[RunStatus, str]:
        if not _safe_import(self.dependency_name):
            return RunStatus.NOT_AVAILABLE, f'{self.dependency_name} is not installed.'
        return RunStatus.SUCCESS, 'dependency is available'

    def detect(self, series_record: DetectionSeriesRecord) -> tuple[dict[str, Any], dict[str, Any]]:
        del series_record
        raise ModelExecutionError(RunStatus.SKIPPED, self.scaffold_reason)


def build_dataset_adapter(spec: DatasetSpec):
    benchmark = spec.benchmark.lower()
    custom_loader = spec.adapter_options.get('loader')
    if benchmark == 'skab':
        return SKABAdapter(loader=custom_loader)
    if benchmark == 'mpsi':
        return MPSIAdapter()
    if benchmark == 'in_memory':
        return InMemoryDetectionAdapter()
    raise BenchmarkConfigurationError(f'Unsupported anomaly detection benchmark adapter: {spec.benchmark}')


def build_model_adapter(spec: ModelSpec):
    raw_adapter_name = spec.adapter_name.lower()
    adapter_name = canonical_detection_model_name(raw_adapter_name)
    params = dict(spec.params)

    if adapter_name in CANONICAL_STAGE_DETECTION_MODELS:
        return RuntimeDetectionModelAdapter(
            adapter_name=adapter_name,
            display_name=spec.display_name,
            params=params,
            tags=spec.tags or ('detection', 'runtime'),
            optional=spec.optional,
        )
    if adapter_name in {'constant_zero', 'naive_zero', 'naive'}:
        return ConstantZeroDetectionModel(name=spec.display_name, tags=spec.tags or ('baseline', 'detection'))
    if adapter_name == 'pyod_iforest':
        return OptionalExternalModel(dependency_name='pyod', name=spec.display_name)

    raise BenchmarkConfigurationError(f'Unsupported detection model adapter: {spec.adapter_name}')


# ADAPTED from forecasting recorder: point labels and classification metrics only.
@dataclass
class DetectionSeriesArtifactsRecorder:
    run_id: str
    metric_names: tuple[str, ...]
    metric_records: list[MetricRecord]
    prediction_records: list[LabelPredictionRecord]

    def validate_detection_output(
            self,
            series_record: DetectionSeriesRecord,
            output: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        actual = _record_labels_array(series_record)
        labels = np.asarray(output.get('labels'), dtype=int).reshape(-1)
        if len(labels) != len(actual):
            raise ModelExecutionError(
                RunStatus.FAILED,
                f'Model returned {len(labels)} labels for {len(actual)} target points.',
            )
        scores_raw = output.get('scores')
        scores = None if scores_raw is None else np.asarray(scores_raw, dtype=float).reshape(-1)
        if scores is not None and len(scores) != len(actual):
            raise ModelExecutionError(
                RunStatus.FAILED,
                f'Model returned {len(scores)} scores for {len(actual)} target points.',
            )
        return actual, labels, scores

    def record_metric_bundle(
            self,
            *,
            series_record: DetectionSeriesRecord,
            model_name: str,
            actual: np.ndarray,
            labels: np.ndarray,
            metadata: dict[str, Any],
            metric_name_suffix: str = '',
    ) -> dict[str, float]:
        del metadata
        metrics_summary: dict[str, float] = {}
        for metric_name in self.metric_names:
            metric_value = compute_detection_metric(metric_name, actual, labels)
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
        return metrics_summary

    def record_predictions(
            self,
            *,
            series_record: DetectionSeriesRecord,
            model_name: str,
            actual: np.ndarray,
            labels: np.ndarray,
    ) -> None:
        for sample_index, (true_label, pred_label) in enumerate(zip(actual, labels)):
            self.prediction_records.append(
                LabelPredictionRecord(
                    run_id=self.run_id,
                    benchmark=series_record.benchmark,
                    dataset_name=series_record.dataset_name,
                    subset=series_record.subset,
                    model_name=model_name,
                    sample_index=sample_index,
                    y_true=str(int(true_label)),
                    y_pred=str(int(pred_label)),
                    status=RunStatus.SUCCESS,
                )
            )


@dataclass
class DetectionPostFitTuningCoordinator:
    config: BenchmarkSuiteConfig
    verbosity_policy: DetectionVerbosityPolicy
    artifacts_recorder: DetectionSeriesArtifactsRecorder

    def _build_tuned_model_spec(self, model_spec: ModelSpec, best_parameters: dict[str, Any]) -> ModelSpec:
        tuned_params = {**dict(model_spec.params), **dict(best_parameters)}
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
        resolved_progress_policy = resolve_detection_progress_policy(
            raw_progress_policy,
            show_progress=self.config.run_spec.show_progress,
        )
        resolved_verbosity_policy = (
            raw_verbosity_policy
            if isinstance(raw_verbosity_policy, DetectionVerbosityPolicy)
            else resolve_detection_verbosity_policy(
                (raw_verbosity_policy or {}).get('level') if isinstance(raw_verbosity_policy, dict)
                else raw_verbosity_policy,
                options=raw_verbosity_policy if isinstance(raw_verbosity_policy, dict) else None,
            )
        )
        return {
            'metric_name': str(
                raw_config.get('metric_name', metadata_runtime.get('metric_name', self.config.run_spec.primary_metric))
            ),
            'stage_updates': raw_config.get('stage_updates'),
            'max_values_per_parameter': int(raw_config.get('max_values_per_parameter', 3)),
            'max_stage_candidates': int(raw_config.get('max_stage_candidates', 16)),
            'split_spec': _resolve_stage_tuning_split_spec(raw_config),
            'progress_policy': resolved_progress_policy,
            'verbosity_policy': resolved_verbosity_policy,
        }

    def run(
            self,
            *,
            model_spec: ModelSpec,
            model,
            series_record: DetectionSeriesRecord,
            baseline_metadata: dict[str, Any],
            baseline_metrics_summary: dict[str, float],
            regime_diagnostics: DetectionRegimeDiagnostics,
            routing_recommendation: DetectionRoutingContext,
    ) -> dict[str, Any]:
        runtime_config = self._resolve_runtime_config(model_spec, baseline_metadata)
        try:
            values = ensure_detection_array(_record_values_array(series_record))
            labels = _record_labels_array(series_record)
            report = run_detection_stage_tuning_on_series(
                model_spec.adapter_name,
                values=values,
                labels=labels,
                base_params={
                    key: value
                    for key, value in dict(model_spec.params).items()
                    if key != 'stage_tuning_runtime'
                },
                stage_updates=runtime_config.get('stage_updates'),
                metric_name=str(runtime_config['metric_name']),
                split_spec=runtime_config.get('split_spec'),
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
                    'progress_policy': resolve_detection_progress_policy(
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
                return {**enriched_baseline_metadata, 'stage_tuning_comparison_error': tuned_message}

            tuned_output, tuned_metadata = tuned_model.detect(series_record)
            actual, tuned_labels, _ = self.artifacts_recorder.validate_detection_output(series_record, tuned_output)
            tuned_metrics_summary = self.artifacts_recorder.record_metric_bundle(
                series_record=series_record,
                model_name=model.name,
                actual=actual,
                labels=tuned_labels,
                metadata=tuned_metadata,
                metric_name_suffix='_tuned',
            )
            comparison_payload = verbosity_policy.prune_stage_tuning_comparison(
                {
                    'best_parameters': best_parameters,
                    'baseline_metrics': baseline_metrics_summary,
                    'tuned_metrics': tuned_metrics_summary,
                    'improved_metrics': {
                        metric_name: bool(
                            tuned_metrics_summary.get(metric_name, -math.inf)
                            >= baseline_metrics_summary.get(metric_name, -math.inf)
                        )
                        for metric_name in baseline_metrics_summary
                        if metric_name in tuned_metrics_summary
                    },
                    'absolute_gain': {
                        metric_name: float(
                            tuned_metrics_summary[metric_name] - baseline_metrics_summary[metric_name]
                        )
                        for metric_name in baseline_metrics_summary
                        if metric_name in tuned_metrics_summary
                    },
                    'tuned_metadata': dict(tuned_metadata),
                    'tuned_labels': [int(value) for value in tuned_labels.tolist()],
                    'tuned_adapter_family': detection_family_for(tuned_model_spec.adapter_name),
                    'regime_diagnostics': regime_diagnostics.to_dict(),
                    'routing_recommendation': routing_recommendation.to_dict(),
                }
            )
            return {
                **enriched_baseline_metadata,
                **({'stage_tuning_comparison': comparison_payload} if comparison_payload is not None else {}),
            }
        except Exception as exc:
            return {**baseline_metadata, 'stage_tuning_comparison_error': str(exc)}


# BORROWED from forecasting: leaderboard aggregation with detection metric sort direction.
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

    ascending = primary_metric not in _HIGHER_IS_BETTER_METRICS
    leaderboard_rows = sorted(leaderboard_rows, key=lambda row: row[primary_metric], reverse=not ascending)
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


# BORROWED from forecasting: dataset → model → series orchestration and resume hooks.
class DetectionSuiteRunner:
    def __init__(self, config: BenchmarkSuiteConfig):
        validate_detection_suite_config(config)
        self.config = config
        self.run_id = (
            DetectionIncrementalPersistenceCoordinator.resolve_run_id(config)
            or new_run_id(config.run_spec.run_name)
        )
        self.series_records: list[DetectionSeriesRecord] = []
        self.run_records: list[BenchmarkRunRecord] = []
        self.prediction_records: list[LabelPredictionRecord] = []
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
        self.verbosity_policy = resolve_detection_verbosity_policy(
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

    def _augment_model_spec_with_progress_policy(self, model_spec: ModelSpec) -> ModelSpec:
        params = dict(model_spec.params)
        params.setdefault('progress_policy', self.progress_policy.to_dict())
        if isinstance(params.get('stage_tuning_runtime'), dict):
            params['stage_tuning_runtime'] = {
                **dict(params['stage_tuning_runtime']),
                'verbosity_policy': dict(
                    params['stage_tuning_runtime'].get('verbosity_policy', self.verbosity_policy.to_dict())
                ),
            }
        return ModelSpec(
            adapter_name=model_spec.adapter_name,
            display_name=model_spec.display_name,
            tags=model_spec.tags,
            optional=model_spec.optional,
            params=params,
        )

    def _runner_context_metadata(self) -> dict[str, Any]:
        if not self.verbosity_policy.include_runner_context:
            return {}
        return {
            'benchmark_runtime_context': {
                'progress_policy': self.progress_policy.to_dict(),
                'verbosity_policy': self.verbosity_policy.to_dict(),
            }
        }

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
        self.incremental_persistence.persist_series_catalog(self.series_records)
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

    def _iter_over_series(
            self,
            model_spec: ModelSpec,
            model,
            dataset_series: tuple[DetectionSeriesRecord, ...],
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
                    existing_record.status.value if existing_record is not None else 'success',
                )
                continue
            self.progress.item_started(series_record.dataset_name, model.name, series_record.series_id)
            regime_diagnostics, routing_recommendation = self._build_series_context(series_record, model_spec)
            try:
                self._evaluate_series(
                    model_spec,
                    model,
                    series_record,
                    regime_diagnostics,
                    routing_recommendation,
                )
                self.progress.advance(RunStatus.SUCCESS.value)
            except ModelExecutionError as exc:
                self._append_failed_run_record(
                    model_spec=model_spec,
                    model=model,
                    series_record=series_record,
                    status=exc.status,
                    message=exc.message,
                    regime_diagnostics=regime_diagnostics,
                    routing_recommendation=routing_recommendation,
                )
                self.progress.advance(exc.status.value, exc.message)
            except Exception as exc:
                self._append_failed_run_record(
                    model_spec=model_spec,
                    model=model,
                    series_record=series_record,
                    status=RunStatus.FAILED,
                    message=str(exc),
                    regime_diagnostics=regime_diagnostics,
                    routing_recommendation=routing_recommendation,
                )
                self.progress.advance(RunStatus.FAILED.value, str(exc))

    def _handle_unavailable_model(
            self,
            *,
            model_spec: ModelSpec,
            model,
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
            regime_diagnostics, routing_recommendation = self._build_series_context(series_record, model_spec)
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

    def _build_series_context(
            self,
            series_record: DetectionSeriesRecord,
            model_spec: ModelSpec,
    ) -> tuple[DetectionRegimeDiagnostics, DetectionRoutingContext]:
        segments = infer_regime_segments(_record_values_array(series_record))
        regime_diagnostics = DetectionRegimeDiagnostics(
            n_segments=len(segments),
            segment_labels=tuple(segment.regime_label for segment in segments),
        )
        routing_recommendation = DetectionRoutingContext(
            primary_adapter=detection_family_for(model_spec.adapter_name),
        )
        return regime_diagnostics, routing_recommendation

    def _build_common_metadata(
            self,
            *,
            model_spec: ModelSpec,
            model,
            regime_diagnostics: DetectionRegimeDiagnostics,
            routing_recommendation: DetectionRoutingContext,
            extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            'adapter_name': model_spec.adapter_name,
            'model_adapter_family': detection_family_for(model_spec.adapter_name),
            'regime_diagnostics': regime_diagnostics.to_dict(),
            'routing_recommendation': routing_recommendation.to_dict(),
            'routing_recommendation_family': routing_recommendation.primary_adapter,
            **self._runner_context_metadata(),
            **dict(extra or {}),
        }

    def _append_failed_run_record(
            self,
            *,
            model_spec: ModelSpec,
            model,
            series_record: DetectionSeriesRecord,
            status: RunStatus,
            message: str,
            regime_diagnostics: DetectionRegimeDiagnostics,
            routing_recommendation: DetectionRoutingContext,
    ) -> BenchmarkRunRecord:
        run_record = BenchmarkRunRecord(
            run_id=self.run_id,
            benchmark=series_record.benchmark,
            dataset_name=series_record.dataset_name,
            subset=series_record.subset,
            series_id=series_record.series_id,
            model_name=model.name,
            status=status,
            tags=model.tags,
            message=message,
            metadata=self._build_common_metadata(
                model_spec=model_spec,
                model=model,
                regime_diagnostics=regime_diagnostics,
                routing_recommendation=routing_recommendation,
                extra={'optional': model.optional},
            ),
        )
        self.run_records.append(run_record)
        self.incremental_persistence.persist_item_result(
            series_record=series_record,
            run_record=run_record,
        )
        return run_record

    def _evaluate_series(
            self,
            model_spec: ModelSpec,
            model,
            series_record: DetectionSeriesRecord,
            regime_diagnostics: DetectionRegimeDiagnostics,
            routing_recommendation: DetectionRoutingContext,
    ) -> None:
        metric_count_before = len(self.metric_records)
        prediction_count_before = len(self.prediction_records)
        output, metadata = model.detect(series_record)
        actual, labels, _ = self.artifacts_recorder.validate_detection_output(series_record, output)
        metrics_summary = self.artifacts_recorder.record_metric_bundle(
            series_record=series_record,
            model_name=model.name,
            actual=actual,
            labels=labels,
            metadata=metadata,
        )
        self.artifacts_recorder.record_predictions(
            series_record=series_record,
            model_name=model.name,
            actual=actual,
            labels=labels,
        )
        metadata = self.post_fit_tuning.run(
            model_spec=model_spec,
            model=model,
            series_record=series_record,
            baseline_metadata=metadata,
            baseline_metrics_summary=metrics_summary,
            regime_diagnostics=regime_diagnostics,
            routing_recommendation=routing_recommendation,
        )
        run_record = BenchmarkRunRecord(
            run_id=self.run_id,
            benchmark=series_record.benchmark,
            dataset_name=series_record.dataset_name,
            subset=series_record.subset,
            series_id=series_record.series_id,
            model_name=model.name,
            status=RunStatus.SUCCESS,
            tags=model.tags,
            metrics_summary=metrics_summary,
            metadata=self._build_common_metadata(
                model_spec=model_spec,
                model=model,
                regime_diagnostics=regime_diagnostics,
                routing_recommendation=routing_recommendation,
                extra=metadata,
            ),
        )
        self.run_records.append(run_record)
        self.incremental_persistence.persist_item_result(
            series_record=series_record,
            run_record=run_record,
            metric_records=tuple(self.metric_records[metric_count_before:]),
            prediction_records=tuple(self.prediction_records[prediction_count_before:]),
        )


def run_anomaly_detection_suite(config: BenchmarkSuiteConfig) -> DetectionBenchmarkResult:
    return DetectionSuiteRunner(config).run_suite()
