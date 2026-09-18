from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from benchmark.industrial.core import ArtifactRecord, TaskType, ensure_directory, write_json
from benchmark.industrial.evaluation.markdown import dataframe_to_markdown
from benchmark.industrial.evaluation.result_analysis import infer_metric_direction, load_jsonl_table


@dataclass(frozen=True)
class TaskAggregationRule:
    task_type: str
    primary_metric: str
    metric_direction: str
    leaderboard_group_columns: tuple[str, ...]
    count_column: str
    prediction_index_columns: tuple[str, ...]

    @property
    def higher_is_better(self) -> bool:
        direction = self.metric_direction.lower()
        if direction not in {'higher', 'lower'}:
            raise ValueError(f'Unsupported metric direction: {self.metric_direction}')
        return direction == 'higher'


@dataclass(frozen=True)
class AggregationInputContract:
    required_record_files: tuple[str, ...] = (
        'records/runs.jsonl',
        'records/metrics.jsonl',
    )
    optional_record_files: tuple[str, ...] = (
        'records/predictions.jsonl',
        'records/errors.jsonl',
        'records/kernel_diagnostics.jsonl',
        'records/kernel_selection.jsonl',
        'errors.jsonl',
        'errors_summary.json',
    )
    optional_aggregate_files: tuple[str, ...] = (
        'aggregate/runs.csv',
        'aggregate/metrics.csv',
        'aggregate/predictions.csv',
        'aggregate/leaderboard.csv',
        'aggregate/run_metadata.json',
        'aggregate/summary.md',
    )


@dataclass(frozen=True)
class AggregationOutputContract:
    aggregate_dir: str = 'aggregate'
    required_files: tuple[str, ...] = (
        'aggregate/runs.csv',
        'aggregate/metrics.csv',
        'aggregate/predictions.csv',
        'aggregate/leaderboard.csv',
        'aggregate/run_metadata.json',
        'aggregate/summary.md',
        'aggregate/artifact_manifest.json',
    )
    optional_files: tuple[str, ...] = (
        'aggregate/errors.csv',
        'aggregate/kernel_diagnostics.csv',
        'aggregate/kernel_selection.csv',
    )


@dataclass(frozen=True)
class BenchmarkArtifactFrames:
    root: Path
    runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    kernel_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    kernel_selection: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkAggregationTables:
    rule: TaskAggregationRule
    runs: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    leaderboard: pd.DataFrame
    run_metadata: dict[str, Any]
    summary_markdown: str
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    kernel_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    kernel_selection: pd.DataFrame = field(default_factory=pd.DataFrame)


DEFAULT_INPUT_CONTRACT = AggregationInputContract()
DEFAULT_OUTPUT_CONTRACT = AggregationOutputContract()


_TASK_ALIASES = {
    'classification': TaskType.TS_CLASSIFICATION.value,
    'tsc': TaskType.TS_CLASSIFICATION.value,
    'ucr': TaskType.TS_CLASSIFICATION.value,
    'ts_classification': TaskType.TS_CLASSIFICATION.value,
    'regression': TaskType.TS_REGRESSION.value,
    'tser': TaskType.TS_REGRESSION.value,
    'ts_regression': TaskType.TS_REGRESSION.value,
    'forecast': TaskType.FORECASTING.value,
    'forecasting': TaskType.FORECASTING.value,
    'm4': TaskType.FORECASTING.value,
}


def resolve_task_aggregation_rule(
        task_type: str | TaskType,
        *,
        primary_metric: str | None = None,
        metric_direction: str | None = None,
) -> TaskAggregationRule:
    normalized_task = _normalize_task_type(task_type)
    metric = primary_metric or _default_metric_for_task(normalized_task)
    direction = metric_direction or infer_metric_direction(metric, default=_default_direction_for_task(normalized_task))
    if normalized_task == TaskType.FORECASTING.value:
        return TaskAggregationRule(
            task_type=normalized_task,
            primary_metric=metric,
            metric_direction=direction,
            leaderboard_group_columns=('benchmark', 'dataset_name', 'model_name'),
            count_column='n_series',
            prediction_index_columns=('dataset_name', 'series_id', 'model_name', 'horizon_index'),
        )
    return TaskAggregationRule(
        task_type=normalized_task,
        primary_metric=metric,
        metric_direction=direction,
        leaderboard_group_columns=('dataset_name', 'model_name'),
        count_column='n_runs',
        prediction_index_columns=('dataset_name', 'model_name', 'sample_index'),
    )


def load_benchmark_artifact_frames(root: str | Path) -> BenchmarkArtifactFrames:
    root_path = Path(root)
    aggregate_dir = root_path / DEFAULT_OUTPUT_CONTRACT.aggregate_dir
    records_dir = root_path / 'records'
    return BenchmarkArtifactFrames(
        root=root_path,
        runs=_load_record_or_aggregate(records_dir / 'runs.jsonl', aggregate_dir / 'runs.csv'),
        metrics=_load_record_or_aggregate(records_dir / 'metrics.jsonl', aggregate_dir / 'metrics.csv'),
        predictions=_load_record_or_aggregate(records_dir / 'predictions.jsonl', aggregate_dir / 'predictions.csv'),
        errors=_load_error_records(root_path),
        kernel_diagnostics=load_jsonl_table(records_dir / 'kernel_diagnostics.jsonl'),
        kernel_selection=load_jsonl_table(records_dir / 'kernel_selection.jsonl'),
        run_metadata=_read_json_if_exists(aggregate_dir / 'run_metadata.json'),
    )


def build_benchmark_aggregate_tables(
        root_or_frames: str | Path | BenchmarkArtifactFrames,
        *,
        task_type: str | TaskType,
        primary_metric: str | None = None,
        metric_direction: str | None = None,
) -> BenchmarkAggregationTables:
    frames = (
        root_or_frames
        if isinstance(root_or_frames, BenchmarkArtifactFrames)
        else load_benchmark_artifact_frames(root_or_frames)
    )
    rule = resolve_task_aggregation_rule(
        task_type,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
    )
    runs = _canonical_runs_frame(frames.runs)
    metrics = _canonical_metrics_frame(frames.metrics)
    predictions = _canonical_predictions_frame(frames.predictions, rule)
    leaderboard = build_leaderboard_frame(metrics, rule)
    run_metadata = _build_run_metadata(frames, runs, metrics, predictions, leaderboard, rule)
    summary_markdown = _build_summary_markdown(rule, run_metadata, leaderboard)
    return BenchmarkAggregationTables(
        rule=rule,
        runs=runs,
        metrics=metrics,
        predictions=predictions,
        leaderboard=leaderboard,
        run_metadata=run_metadata,
        summary_markdown=summary_markdown,
        errors=frames.errors.copy(),
        kernel_diagnostics=frames.kernel_diagnostics.copy(),
        kernel_selection=frames.kernel_selection.copy(),
    )


def build_leaderboard_frame(metrics: pd.DataFrame, rule: TaskAggregationRule) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=[*rule.leaderboard_group_columns, rule.primary_metric, rule.count_column, 'rank'])
    required_columns = set(rule.leaderboard_group_columns) | {'metric_name', 'metric_value'}
    missing_columns = required_columns - set(metrics.columns)
    if missing_columns:
        raise ValueError(f'Metrics table is missing required columns: {sorted(missing_columns)}')
    filtered = metrics[metrics['metric_name'].astype(str) == rule.primary_metric].copy()
    if 'status' in filtered.columns:
        filtered = filtered[filtered['status'].astype(str) == 'success'].copy()
    filtered['metric_value'] = pd.to_numeric(filtered['metric_value'], errors='coerce')
    filtered = filtered.dropna(subset=['metric_value'])
    if filtered.empty:
        return pd.DataFrame(columns=[*rule.leaderboard_group_columns, rule.primary_metric, rule.count_column, 'rank'])
    grouped = (
        filtered.groupby(list(rule.leaderboard_group_columns), dropna=False)
        .agg(metric_value=('metric_value', 'mean'), count=('metric_value', 'size'))
        .reset_index()
        .rename(columns={'metric_value': rule.primary_metric, 'count': rule.count_column})
    )
    grouped = grouped.sort_values(
        [rule.primary_metric, *rule.leaderboard_group_columns],
        ascending=[not rule.higher_is_better, *([True] * len(rule.leaderboard_group_columns))],
    ).reset_index(drop=True)
    grouped['rank'] = grouped[rule.primary_metric].rank(
        method='dense',
        ascending=not rule.higher_is_better,
    ).astype(int)
    return grouped[[*rule.leaderboard_group_columns, rule.primary_metric, rule.count_column, 'rank']]


def render_benchmark_aggregate_artifacts(
        root: str | Path,
        *,
        output_dir: str | Path | None = None,
        task_type: str | TaskType,
        primary_metric: str | None = None,
        metric_direction: str | None = None,
) -> tuple[ArtifactRecord, ...]:
    tables = build_benchmark_aggregate_tables(
        root,
        task_type=task_type,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
    )
    aggregate_dir = ensure_directory(Path(output_dir) if output_dir is not None else Path(root) / 'aggregate')
    manifest: list[ArtifactRecord] = []
    for name, frame in (
            ('runs', tables.runs),
            ('metrics', tables.metrics),
            ('predictions', tables.predictions),
            ('leaderboard', tables.leaderboard),
    ):
        manifest.append(_write_csv_artifact(frame, aggregate_dir / f'{name}.csv', kind='table'))
    for name, frame in (
            ('errors', tables.errors),
            ('kernel_diagnostics', tables.kernel_diagnostics),
            ('kernel_selection', tables.kernel_selection),
    ):
        if not frame.empty:
            manifest.append(_write_csv_artifact(frame, aggregate_dir / f'{name}.csv', kind='table'))

    metadata_path = aggregate_dir / 'run_metadata.json'
    write_json(metadata_path, tables.run_metadata)
    manifest.append(ArtifactRecord(kind='structured', path=str(metadata_path), format='json'))

    summary_path = aggregate_dir / 'summary.md'
    summary_path.write_text(tables.summary_markdown, encoding='utf-8')
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))

    manifest_path = aggregate_dir / 'artifact_manifest.json'
    manifest.append(ArtifactRecord(kind='structured', path=str(manifest_path), format='json'))
    write_json(manifest_path, [record.__dict__ for record in manifest])
    return tuple(manifest)


def _normalize_task_type(task_type: str | TaskType) -> str:
    value = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    normalized = value.strip().lower()
    resolved = _TASK_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(f'Unsupported task type for aggregation: {task_type}')
    return resolved


def _default_metric_for_task(task_type: str) -> str:
    if task_type == TaskType.TS_CLASSIFICATION.value:
        return 'accuracy'
    if task_type == TaskType.TS_REGRESSION.value:
        return 'rmse'
    return 'mae'


def _default_direction_for_task(task_type: str) -> str:
    return 'higher' if task_type == TaskType.TS_CLASSIFICATION.value else 'lower'


def _load_record_or_aggregate(record_path: Path, aggregate_path: Path) -> pd.DataFrame:
    if record_path.exists():
        return load_jsonl_table(record_path)
    if aggregate_path.exists():
        return pd.read_csv(aggregate_path)
    return pd.DataFrame()


def _load_error_records(root_path: Path) -> pd.DataFrame:
    candidates = (root_path / 'records' / 'errors.jsonl', root_path / 'errors.jsonl')
    frames = [load_jsonl_table(path) for path in candidates if path.exists()]
    frames = [frame for frame in frames if not frame.empty]
    if frames:
        return pd.concat(frames, ignore_index=True)
    summary_path = root_path / 'errors_summary.json'
    payload = _read_json_payload_if_exists(summary_path)
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    return pd.DataFrame([payload])


def _read_json_if_exists(path: Path) -> Mapping[str, Any]:
    payload = _read_json_payload_if_exists(path)
    if payload is None:
        return {}
    return payload if isinstance(payload, MappingABC) else {'payload': payload}


def _read_json_payload_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _canonical_runs_frame(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    frame = runs.copy()
    if 'status' in frame.columns:
        frame['status'] = frame['status'].astype(str)
    return frame.reset_index(drop=True)


def _canonical_metrics_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    frame = metrics.copy()
    if 'metric_value' in frame.columns:
        frame['metric_value'] = pd.to_numeric(frame['metric_value'], errors='coerce')
    if 'status' in frame.columns:
        frame['status'] = frame['status'].astype(str)
    return frame.reset_index(drop=True)


def _canonical_predictions_frame(predictions: pd.DataFrame, rule: TaskAggregationRule) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    sort_columns = [column for column in rule.prediction_index_columns if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns)
    return frame.reset_index(drop=True)


def _build_run_metadata(
        frames: BenchmarkArtifactFrames,
        runs: pd.DataFrame,
        metrics: pd.DataFrame,
        predictions: pd.DataFrame,
        leaderboard: pd.DataFrame,
        rule: TaskAggregationRule,
) -> dict[str, Any]:
    status_counts = runs['status'].astype(str).value_counts().sort_index().to_dict() if 'status' in runs.columns else {}
    run_ids = sorted(str(item) for item in runs['run_id'].dropna().unique()) if 'run_id' in runs.columns else []
    observed_datasets = (
        sorted(str(item) for item in metrics['dataset_name'].dropna().unique())
        if 'dataset_name' in metrics.columns
        else []
    )
    metadata = dict(frames.run_metadata)
    metadata.update(
        {
            'run_ids': run_ids,
            'source_root': str(frames.root),
            'task_type': rule.task_type,
            'primary_metric': rule.primary_metric,
            'metric_direction': rule.metric_direction,
            'status_counts': {str(key): int(value) for key, value in status_counts.items()},
            'record_counts': {
                'runs': int(len(runs)),
                'metrics': int(len(metrics)),
                'predictions': int(len(predictions)),
                'leaderboard': int(len(leaderboard)),
                'errors': int(len(frames.errors)),
                'kernel_diagnostics': int(len(frames.kernel_diagnostics)),
                'kernel_selection': int(len(frames.kernel_selection)),
            },
            'observed_dataset_count': int(len(observed_datasets)),
            'observed_datasets': observed_datasets,
            'input_contract': {
                'required_record_files': list(DEFAULT_INPUT_CONTRACT.required_record_files),
                'optional_record_files': list(DEFAULT_INPUT_CONTRACT.optional_record_files),
                'optional_aggregate_files': list(DEFAULT_INPUT_CONTRACT.optional_aggregate_files),
            },
            'output_contract': {
                'required_files': list(DEFAULT_OUTPUT_CONTRACT.required_files),
                'optional_files': list(DEFAULT_OUTPUT_CONTRACT.optional_files),
            },
        }
    )
    return metadata


def _build_summary_markdown(
        rule: TaskAggregationRule,
        run_metadata: Mapping[str, Any],
        leaderboard: pd.DataFrame,
) -> str:
    run_ids = run_metadata.get('run_ids', ())
    run_id = str(run_ids[0]) if run_ids else 'unknown'
    lines = [
        f'# Benchmark Aggregate Summary: {run_id}',
        '',
        f'- Task type: `{rule.task_type}`',
        f'- Primary metric: `{rule.primary_metric}`',
        f'- Metric direction: `{rule.metric_direction}`',
        f'- Successful runs: `{dict(run_metadata.get("status_counts", {})).get("success", 0)}`',
        f'- Observed datasets: `{run_metadata.get("observed_dataset_count", 0)}`',
        '',
        '## Leaderboard',
        '',
        dataframe_to_markdown(leaderboard, index=False) if not leaderboard.empty else 'No successful metric rows.',
    ]
    return '\n'.join(lines)


def _write_csv_artifact(frame: pd.DataFrame, path: Path, *, kind: str) -> ArtifactRecord:
    frame.to_csv(path, index=False)
    return ArtifactRecord(kind=kind, path=str(path), format='csv')


__all__ = [
    'AggregationInputContract',
    'AggregationOutputContract',
    'BenchmarkAggregationTables',
    'BenchmarkArtifactFrames',
    'DEFAULT_INPUT_CONTRACT',
    'DEFAULT_OUTPUT_CONTRACT',
    'TaskAggregationRule',
    'build_benchmark_aggregate_tables',
    'build_leaderboard_frame',
    'load_benchmark_artifact_frames',
    'render_benchmark_aggregate_artifacts',
    'resolve_task_aggregation_rule',
]
