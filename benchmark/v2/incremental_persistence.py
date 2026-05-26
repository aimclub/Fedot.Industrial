from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import (
    ArtifactRecord,
    BenchmarkRunRecord,
    BenchmarkSuiteConfig,
    DetectionSeriesRecord,
    ForecastingSeriesRecord,
    LabelPredictionRecord,
    DetectionPredictionRecord,
    MetricRecord,
    PredictionRecord,
    RunStatus,
    ensure_directory,
    to_plain_data,
)


def _safe_slug(value: str) -> str:
    """Convert a dynamic key into a stable file-safe slug."""
    slug = ''.join(character.lower() if character.isalnum() else '_' for character in str(value))
    return slug.strip('_') or 'item'


def _write_json_atomic(path: str | Path, payload: Any) -> None:
    """Persist JSON atomically to avoid half-written progress artifacts."""
    target_path = Path(path)
    ensure_directory(target_path.parent)
    temporary_path = target_path.with_name(f'{target_path.name}.tmp')
    with temporary_path.open('w', encoding='utf-8') as stream:
        json.dump(to_plain_data(payload), stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(target_path)


def _read_json(path: str | Path) -> Any:
    """Read JSON payload from a persisted progress artifact."""
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def _as_tuple(values: Any) -> tuple[Any, ...]:
    """Convert JSON list-ish values to tuple while preserving empty as ()"""
    if values is None:
        return ()
    return tuple(values)


def _series_record_from_payload(payload: dict[str, Any]) -> ForecastingSeriesRecord:
    """Restore ForecastingSeriesRecord from persisted JSON payload."""
    return ForecastingSeriesRecord(
        benchmark=str(payload['benchmark']),
        dataset_name=str(payload['dataset_name']),
        subset=str(payload['subset']),
        series_id=str(payload['series_id']),
        frequency=str(payload['frequency']),
        forecast_horizon=int(payload['forecast_horizon']),
        seasonal_period=int(payload['seasonal_period']),
        train_values=tuple(float(value) for value in payload.get('train_values', ())),
        test_values=tuple(float(value) for value in payload.get('test_values', ())),
        metadata=dict(payload.get('metadata', {})),
    )


def _run_record_from_payload(payload: dict[str, Any]) -> BenchmarkRunRecord:
    """Restore BenchmarkRunRecord from persisted JSON payload."""
    return BenchmarkRunRecord(
        run_id=str(payload['run_id']),
        benchmark=str(payload['benchmark']),
        dataset_name=str(payload['dataset_name']),
        subset=str(payload['subset']),
        series_id=str(payload['series_id']),
        model_name=str(payload['model_name']),
        status=RunStatus(str(payload['status'])),
        tags=_as_tuple(payload.get('tags')),
        message=str(payload.get('message', '')),
        metrics_summary={str(key): float(value) for key, value in dict(payload.get('metrics_summary', {})).items()},
        metadata=dict(payload.get('metadata', {})),
    )


def _metric_record_from_payload(payload: dict[str, Any]) -> MetricRecord:
    """Restore MetricRecord from persisted JSON payload."""
    return MetricRecord(
        run_id=str(payload['run_id']),
        benchmark=str(payload['benchmark']),
        dataset_name=str(payload['dataset_name']),
        subset=str(payload['subset']),
        series_id=str(payload['series_id']),
        model_name=str(payload['model_name']),
        metric_name=str(payload['metric_name']),
        metric_value=float(payload['metric_value']),
        status=RunStatus(str(payload['status'])),
        horizon_index=None if payload.get('horizon_index') is None else int(payload['horizon_index']),
    )


def _prediction_record_from_payload(payload: dict[str, Any]) -> PredictionRecord:
    """Restore forecasting PredictionRecord from persisted JSON payload."""
    return PredictionRecord(
        run_id=str(payload['run_id']),
        benchmark=str(payload['benchmark']),
        dataset_name=str(payload['dataset_name']),
        subset=str(payload['subset']),
        series_id=str(payload['series_id']),
        model_name=str(payload['model_name']),
        horizon_index=int(payload['horizon_index']),
        y_true=float(payload['y_true']),
        y_pred=float(payload['y_pred']),
        status=RunStatus(str(payload['status'])),
    )


def _detection_series_record_from_payload(payload: dict[str, Any]) -> DetectionSeriesRecord:
    """Restore DetectionSeriesRecord from persisted JSON payload."""
    values = tuple(tuple(float(value) for value in row) for row in payload.get('values', ()))
    target_raw = payload.get('target')
    target = None if target_raw is None else tuple(int(value) for value in target_raw)
    return DetectionSeriesRecord(
        benchmark=str(payload['benchmark']),
        dataset_name=str(payload['dataset_name']),
        subset=str(payload['subset']),
        series_id=str(payload['series_id']),
        values=values,
        target=target,
        timestamps=tuple(str(value) for value in payload.get('timestamps', ())),
        metadata=dict(payload.get('metadata', {})),
    )


def _detection_prediction_record_from_payload(payload: dict[str, Any]) -> LabelPredictionRecord:
    """Restore detection LabelPredictionRecord from persisted JSON payload."""
    return DetectionPredictionRecord(
        run_id=str(payload['run_id']),
        benchmark=str(payload['benchmark']),
        dataset_name=str(payload['dataset_name']),
        subset=str(payload['subset']),
        series_id=str(payload['series_id']),
        model_name=str(payload['model_name']),
        sample_index=int(payload['sample_index']),
        y_true=str(payload['y_true']),
        y_pred=str(payload['y_pred']),
        y_score=None if payload.get('y_score') is None else float(payload['y_score']),
        timestamp=None if payload.get('timestamp') is None else str(payload['timestamp']),
        status=RunStatus(str(payload['status'])),
    )


@dataclass(frozen=True)
class ForecastingResumeState:
    """Recovered forecasting state loaded from incremental progress artifacts."""

    series_records: tuple[ForecastingSeriesRecord, ...]
    run_records: tuple[BenchmarkRunRecord, ...]
    metric_records: tuple[MetricRecord, ...]
    prediction_records: tuple[PredictionRecord, ...]
    item_artifact_paths: dict[str, str]
    status_counts: dict[str, int]
    completed_items: int


@dataclass(frozen=True)
class DetectionResumeState:
    """Recovered detection state loaded from progress artifacts."""

    series_records: tuple[DetectionSeriesRecord, ...]
    run_records: tuple[BenchmarkRunRecord, ...]
    metric_records: tuple[MetricRecord, ...]
    prediction_records: tuple[DetectionPredictionRecord, ...]
    item_artifact_paths: dict[str, str]
    status_counts: dict[str, int]
    completed_items: int


class _BaseIncrementalPersistenceCoordinator:
    """Common logic for item-wise benchmark persistence and resume.

    Subclasses provide type-specific payload adapters for:
    - series catalog records;
    - prediction records;
    - assembled resume state dataclass.
    """

    def __init__(self, config: BenchmarkSuiteConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        self.enabled = bool(config.artifact_spec.persist_on_run)
        self.item_artifact_paths: dict[str, str] = {}
        self.status_counts: dict[str, int] = {}
        self.completed_items = 0

        if not self.enabled:
            self.run_dir = None
            self.progress_dir = None
            self.items_dir = None
            self.context_path = None
            self.series_catalog_path = None
            self.progress_index_path = None
            return

        self.run_dir = ensure_directory(Path(config.artifact_spec.output_dir) / run_id)
        self.progress_dir = ensure_directory(self.run_dir / 'progress')
        self.items_dir = ensure_directory(self.progress_dir / 'items')
        self.context_path = self.progress_dir / 'run_context.json'
        self.series_catalog_path = self.progress_dir / 'series_records.json'
        self.progress_index_path = self.progress_dir / 'run_progress.json'
        self.resuming_existing = bool(
            getattr(config.run_spec, 'resume_enabled', False)
            and self.progress_index_path.exists()
        )
        if not self.resuming_existing:
            self._write_run_context()
            self._write_progress_index()

    @staticmethod
    def resolve_run_id(config: BenchmarkSuiteConfig) -> str | None:
        """Return explicit resume id or latest resumable run directory name."""
        if not bool(config.artifact_spec.persist_on_run):
            return None
        run_spec = config.run_spec
        if not bool(getattr(run_spec, 'resume_enabled', False)):
            return None
        explicit_run_id = getattr(run_spec, 'resume_run_id', None)
        if explicit_run_id:
            return str(explicit_run_id)
        output_dir = Path(config.artifact_spec.output_dir)
        if not output_dir.exists():
            return None
        prefix = f'{run_spec.run_name}_'
        candidates = [
            path for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith(prefix) and (path / 'progress' / 'run_progress.json').exists()
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0].name

    def _write_run_context(self) -> None:
        """Persist static run metadata required for traceability."""
        if not self.enabled:
            return
        _write_json_atomic(
            self.context_path,
            {
                'run_id': self.run_id,
                'task_type': self.config.task_type.value,
                'metrics': list(self.config.metrics),
                'run_spec': to_plain_data(self.config.run_spec),
                'artifact_spec': to_plain_data(self.config.artifact_spec),
                'datasets': [to_plain_data(dataset) for dataset in self.config.datasets],
                'models': [to_plain_data(model) for model in self.config.models],
            },
        )

    def _write_progress_index(self) -> None:
        """Persist mutable progress index used by resume logic."""
        if not self.enabled:
            return
        _write_json_atomic(
            self.progress_index_path,
            {
                'run_id': self.run_id,
                'completed_items': int(self.completed_items),
                'status_counts': dict(self.status_counts),
                'item_artifacts': dict(self.item_artifact_paths),
            },
        )

    def persist_series_catalog(self, series_records: list[Any]) -> None:
        """Persist complete list of series processed in the run."""
        if not self.enabled:
            return
        _write_json_atomic(
            self.series_catalog_path,
            [to_plain_data(record) for record in series_records],
        )

    def _item_artifact_name(self, series_record: Any, run_record: BenchmarkRunRecord) -> str:
        """Build deterministic per-item artifact filename."""
        return '__'.join(
            (
                _safe_slug(series_record.dataset_name),
                _safe_slug(series_record.subset),
                _safe_slug(series_record.series_id),
                _safe_slug(run_record.model_name),
            )
        ) + '.json'

    @staticmethod
    def item_key(series_record: Any, model_name: str) -> str:
        """Build stable resume key used by runner to skip completed items."""
        return '::'.join(
            (
                str(series_record.benchmark),
                str(series_record.dataset_name),
                str(series_record.subset),
                str(series_record.series_id),
                str(model_name),
            )
        )

    def persist_item_result(
            self,
            *,
            series_record: Any,
            run_record: BenchmarkRunRecord,
            metric_records: tuple[MetricRecord, ...] = (),
            prediction_records: tuple[Any, ...] = (),
    ) -> None:
        """Atomically persist one item and update status counters."""
        if not self.enabled:
            return
        artifact_name = self._item_artifact_name(series_record, run_record)
        artifact_path = self.items_dir / artifact_name
        item_key = self.item_key(series_record, run_record.model_name)
        previous_path = self.item_artifact_paths.get(item_key)
        if previous_path and Path(previous_path).exists():
            previous_payload = _read_json(previous_path)
            previous_status = str(dict(previous_payload.get('run_record', {})).get('status', '')).lower()
            if previous_status:
                previous_count = self.status_counts.get(previous_status, 0)
                self.status_counts[previous_status] = max(0, previous_count - 1)
        _write_json_atomic(
            artifact_path,
            {
                'run_id': self.run_id,
                'series_record': to_plain_data(series_record),
                'run_record': to_plain_data(run_record),
                'metric_records': [to_plain_data(record) for record in metric_records],
                'prediction_records': [to_plain_data(record) for record in prediction_records],
            },
        )
        self.item_artifact_paths[item_key] = str(artifact_path)
        status = run_record.status.value
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        self.completed_items = len(self.item_artifact_paths)
        self._write_progress_index()

    def _series_record_from_payload(self, payload: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _prediction_record_from_payload(self, payload: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _build_resume_state(
            self,
            *,
            series_records: tuple[Any, ...],
            run_records: tuple[BenchmarkRunRecord, ...],
            metric_records: tuple[MetricRecord, ...],
            prediction_records: tuple[Any, ...],
            item_artifact_paths: dict[str, str],
            status_counts: dict[str, int],
            completed_items: int,
    ) -> Any:
        raise NotImplementedError

    def load_resume_state(self) -> Any | None:
        """Reconstruct runner state from persisted item artifacts."""
        if not self.enabled or not bool(getattr(self.config.run_spec, 'resume_enabled', False)):
            return None
        if not self.progress_dir.exists() or not self.progress_index_path.exists():
            return None

        progress_payload = _read_json(self.progress_index_path)
        item_artifact_paths = {
            str(key): str(value)
            for key, value in dict(progress_payload.get('item_artifacts', {})).items()
        }
        status_counts = {
            str(key): int(value)
            for key, value in dict(progress_payload.get('status_counts', {})).items()
        }
        completed_items = int(progress_payload.get('completed_items', len(item_artifact_paths)))

        series_records: list[Any] = []
        if self.series_catalog_path.exists():
            series_records = [
                self._series_record_from_payload(payload)
                for payload in list(_read_json(self.series_catalog_path) or [])
            ]

        run_records: list[BenchmarkRunRecord] = []
        metric_records: list[MetricRecord] = []
        prediction_records: list[Any] = []
        seen_run_keys: set[tuple[str, str, str, str, str]] = set()

        for artifact_path in item_artifact_paths.values():
            payload = _read_json(artifact_path)
            run_record = _run_record_from_payload(dict(payload.get('run_record', {})))
            run_key = (
                run_record.benchmark,
                run_record.dataset_name,
                run_record.subset,
                run_record.series_id,
                run_record.model_name,
            )
            if run_key not in seen_run_keys:
                run_records.append(run_record)
                seen_run_keys.add(run_key)
            metric_records.extend(
                _metric_record_from_payload(item)
                for item in list(payload.get('metric_records', ()))
            )
            prediction_records.extend(
                self._prediction_record_from_payload(item)
                for item in list(payload.get('prediction_records', ()))
            )

        self.item_artifact_paths = dict(item_artifact_paths)
        self.status_counts = dict(status_counts)
        self.completed_items = int(completed_items)
        self._write_progress_index()
        return self._build_resume_state(
            series_records=tuple(series_records),
            run_records=tuple(run_records),
            metric_records=tuple(metric_records),
            prediction_records=tuple(prediction_records),
            item_artifact_paths=dict(item_artifact_paths),
            status_counts=dict(status_counts),
            completed_items=int(completed_items),
        )

    def build_artifact_manifest(self) -> tuple[ArtifactRecord, ...]:
        """Return progress artifact entries for manifest rendering."""
        if not self.enabled:
            return ()
        return (
            ArtifactRecord(kind='structured', path=str(self.context_path), format='json'),
            ArtifactRecord(kind='structured', path=str(self.series_catalog_path), format='json'),
            ArtifactRecord(kind='structured', path=str(self.progress_index_path), format='json'),
        )


class ForecastingIncrementalPersistenceCoordinator(_BaseIncrementalPersistenceCoordinator):
    """Forecasting-specific persistence coordinator with typed resume payload."""

    def _series_record_from_payload(self, payload: dict[str, Any]) -> ForecastingSeriesRecord:
        return _series_record_from_payload(payload)

    def _prediction_record_from_payload(self, payload: dict[str, Any]) -> PredictionRecord:
        return _prediction_record_from_payload(payload)

    def _build_resume_state(self, **kwargs: Any) -> ForecastingResumeState:
        return ForecastingResumeState(**kwargs)


class DetectionIncrementalPersistenceCoordinator(_BaseIncrementalPersistenceCoordinator):
    """Detection-specific persistence coordinator with typed resume payload."""

    def _series_record_from_payload(self, payload: dict[str, Any]) -> DetectionSeriesRecord:
        return _detection_series_record_from_payload(payload)

    def _prediction_record_from_payload(self, payload: dict[str, Any]) -> DetectionPredictionRecord:
        return _detection_prediction_record_from_payload(payload)

    def _build_resume_state(self, **kwargs: Any) -> DetectionResumeState:
        return DetectionResumeState(**kwargs)
