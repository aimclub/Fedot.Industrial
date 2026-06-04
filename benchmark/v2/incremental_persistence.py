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
    ForecastingSeriesRecord,
    MetricRecord,
    PredictionRecord,
    RunStatus,
    ensure_directory,
    to_plain_data,
)


def _safe_slug(value: str) -> str:
    slug = ''.join(character.lower() if character.isalnum() else '_' for character in str(value))
    return slug.strip('_') or 'item'


def _write_json_atomic(path: str | Path, payload: Any) -> None:
    target_path = Path(path)
    ensure_directory(target_path.parent)
    temporary_path = target_path.with_name(f'{target_path.name}.tmp')
    with temporary_path.open('w', encoding='utf-8') as stream:
        json.dump(to_plain_data(payload), stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(target_path)


def _read_json(path: str | Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def _as_tuple(values: Any) -> tuple[Any, ...]:
    if values is None:
        return ()
    return tuple(values)


def _series_record_from_payload(payload: dict[str, Any]) -> ForecastingSeriesRecord:
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


def _metric_records_from_run_summary(run_record: BenchmarkRunRecord) -> tuple[MetricRecord, ...]:
    if run_record.status is not RunStatus.SUCCESS:
        return ()
    return tuple(
        MetricRecord(
            run_id=run_record.run_id,
            benchmark=run_record.benchmark,
            dataset_name=run_record.dataset_name,
            subset=run_record.subset,
            series_id=run_record.series_id,
            model_name=run_record.model_name,
            metric_name=str(metric_name),
            metric_value=float(metric_value),
            status=run_record.status,
        )
        for metric_name, metric_value in run_record.metrics_summary.items()
    )


def _prediction_record_from_payload(payload: dict[str, Any]) -> PredictionRecord:
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


@dataclass(frozen=True)
class ForecastingResumeState:
    """Recovered benchmark records loaded from incremental progress artifacts."""

    series_records: tuple[ForecastingSeriesRecord, ...]
    run_records: tuple[BenchmarkRunRecord, ...]
    metric_records: tuple[MetricRecord, ...]
    prediction_records: tuple[PredictionRecord, ...]
    item_artifact_paths: dict[str, str]
    status_counts: dict[str, int]
    completed_items: int


class ForecastingIncrementalPersistenceCoordinator:
    """Persist forecasting benchmark progress item-by-item and support resume."""

    def __init__(self, config: BenchmarkSuiteConfig, run_id: str):
        """Prepare run, progress and item directories for incremental persistence."""
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
        """Find an explicit or latest resumable run id for a benchmark config."""
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

    def persist_series_catalog(self, series_records: list[ForecastingSeriesRecord]) -> None:
        """Persist the full series catalog used by a forecasting run."""
        if not self.enabled:
            return
        _write_json_atomic(
            self.series_catalog_path,
            [to_plain_data(record) for record in series_records],
        )

    def _item_artifact_name(self, series_record: ForecastingSeriesRecord, run_record: BenchmarkRunRecord) -> str:
        return '__'.join(
            (
                _safe_slug(series_record.dataset_name),
                _safe_slug(series_record.subset),
                _safe_slug(series_record.series_id),
                _safe_slug(run_record.model_name),
            )
        ) + '.json'

    @staticmethod
    def item_key(series_record: ForecastingSeriesRecord, model_name: str) -> str:
        """Build a stable resume key for a series/model benchmark item."""
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
            series_record: ForecastingSeriesRecord,
            run_record: BenchmarkRunRecord,
            metric_records: tuple[MetricRecord, ...] = (),
            prediction_records: tuple[PredictionRecord, ...] = (),
    ) -> None:
        """Atomically persist one completed series/model item result."""
        if not self.enabled:
            return
        artifact_name = self._item_artifact_name(series_record, run_record)
        artifact_path = self.items_dir / artifact_name
        item_key = self.item_key(series_record, run_record.model_name)
        previous_path = self.item_artifact_paths.get(item_key)
        is_new_item = previous_path is None
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
        if is_new_item:
            self.completed_items = len(self.item_artifact_paths)
        self._write_progress_index()

    def load_resume_state(self) -> ForecastingResumeState | None:
        """Load previously persisted item records when resume mode is enabled."""
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

        series_records: list[ForecastingSeriesRecord] = []
        if self.series_catalog_path.exists():
            series_records = [
                _series_record_from_payload(payload)
                for payload in list(_read_json(self.series_catalog_path) or [])
            ]

        run_records: list[BenchmarkRunRecord] = []
        metric_records: list[MetricRecord] = []
        prediction_records: list[PredictionRecord] = []
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
            metric_payloads = list(payload.get('metric_records', ()))
            if metric_payloads:
                metric_records.extend(_metric_record_from_payload(item) for item in metric_payloads)
            else:
                metric_records.extend(_metric_records_from_run_summary(run_record))
            prediction_records.extend(
                _prediction_record_from_payload(item)
                for item in list(payload.get('prediction_records', ()))
            )

        self.item_artifact_paths = dict(item_artifact_paths)
        self.status_counts = dict(status_counts)
        self.completed_items = int(completed_items)
        self._write_progress_index()
        return ForecastingResumeState(
            series_records=tuple(series_records),
            run_records=tuple(run_records),
            metric_records=tuple(metric_records),
            prediction_records=tuple(prediction_records),
            item_artifact_paths=dict(item_artifact_paths),
            status_counts=dict(status_counts),
            completed_items=int(completed_items),
        )

    def build_artifact_manifest(self) -> tuple[ArtifactRecord, ...]:
        """Return manifest entries for progress-level persistence artifacts."""
        if not self.enabled:
            return ()
        return (
            ArtifactRecord(kind='structured', path=str(self.context_path), format='json'),
            ArtifactRecord(kind='structured', path=str(self.series_catalog_path), format='json'),
            ArtifactRecord(kind='structured', path=str(self.progress_index_path), format='json'),
        )
