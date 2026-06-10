from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


HIGHER_IS_BETTER_METRICS = {
    'accuracy',
    'balanced_accuracy',
    'f1',
    'f1_macro',
    'precision',
    'recall',
    'r2',
    'roc_auc',
}


@dataclass(frozen=True)
class ResultAnalysisSpec:
    metric_name: str
    metric_direction: str = 'higher'
    source_label: str = 'benchmark'
    task_type: str = ''
    dataset_column: str = 'dataset_name'
    model_column: str = 'model_name'
    value_column: str = 'metric_value'

    @property
    def higher_is_better(self) -> bool:
        direction = self.metric_direction.lower()
        if direction not in {'higher', 'lower'}:
            raise ValueError(f'Unsupported metric direction: {self.metric_direction}')
        return direction == 'higher'


def infer_metric_direction(metric_name: str, default: str = 'higher') -> str:
    normalized = str(metric_name).strip().lower()
    if not normalized:
        return default
    return 'higher' if normalized in HIGHER_IS_BETTER_METRICS else 'lower'


def load_result_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == '.csv':
        frame = pd.read_csv(table_path, sep=None, engine='python')
        if len(frame.columns) == 1:
            frame = pd.read_csv(table_path, sep=';', engine='python')
        return _normalize_decimal_comma_frame(frame)
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(table_path)
    if suffix == '.json':
        return pd.read_json(table_path)
    raise ValueError(f'Unsupported result table format: {table_path}')


def load_jsonl_table(path: str | Path) -> pd.DataFrame:
    rows = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return pd.DataFrame()
    with jsonl_path.open(encoding='utf-8') as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return pd.DataFrame(rows)


def load_incremental_metric_records(
        root: str | Path,
        *,
        metric_name: str | None = None,
        source_label: str = 'incremental_run',
        task_type: str = '',
        metric_direction: str | None = None,
) -> pd.DataFrame:
    root_path = Path(root)
    frames = [load_jsonl_table(path) for path in sorted(root_path.rglob('records/metrics.jsonl'))]
    if not frames:
        return _empty_normalized_frame()
    frame = pd.concat([item for item in frames if not item.empty], ignore_index=True)
    if frame.empty:
        return _empty_normalized_frame()
    if metric_name is not None and 'metric_name' in frame.columns:
        frame = frame[frame['metric_name'].astype(str) == str(metric_name)].copy()
    if 'status' in frame.columns:
        frame = frame[frame['status'].astype(str) == 'success'].copy()
    if frame.empty:
        return _empty_normalized_frame()
    columns = ['dataset_name', 'model_name', 'metric_name', 'metric_value']
    if 'series_id' in frame.columns:
        columns.append('series_id')
    normalized = frame.rename(columns={'metric_value': 'metric_value'})[columns].copy()
    normalized['source_label'] = source_label
    normalized['task_type'] = task_type
    normalized['metric_direction'] = metric_direction or infer_metric_direction(metric_name or '')
    normalized['metric_value'] = pd.to_numeric(normalized['metric_value'], errors='coerce')
    return normalized.dropna(subset=['metric_value']).reset_index(drop=True)


def load_incremental_run_records(root: str | Path) -> pd.DataFrame:
    root_path = Path(root)
    frames = [load_jsonl_table(path) for path in sorted(root_path.rglob('records/runs.jsonl'))]
    if not frames:
        return _empty_run_records_frame()
    return pd.concat([item for item in frames if not item.empty], ignore_index=True)


def load_incremental_kernel_diagnostics(root: str | Path) -> pd.DataFrame:
    root_path = Path(root)
    frames = [load_jsonl_table(path) for path in sorted(root_path.rglob('records/kernel_diagnostics.jsonl'))]
    if not frames:
        return _empty_kernel_diagnostics_frame()
    return pd.concat([item for item in frames if not item.empty], ignore_index=True)


def load_aggregate_metric_records(
        root: str | Path,
        *,
        metric_name: str | None = None,
        source_label: str = 'aggregate_run',
        task_type: str = '',
        metric_direction: str | None = None,
) -> pd.DataFrame:
    root_path = Path(root)
    paths = sorted(root_path.rglob('aggregate/metrics.csv'))
    if not paths:
        return _empty_normalized_frame()
    frames = [load_result_table(path) for path in paths]
    frame = pd.concat([item for item in frames if not item.empty], ignore_index=True)
    if frame.empty:
        return _empty_normalized_frame()
    if metric_name is not None and 'metric_name' in frame.columns:
        frame = frame[frame['metric_name'].astype(str) == str(metric_name)].copy()
    if 'status' in frame.columns:
        frame = frame[frame['status'].astype(str) == 'success'].copy()
    if frame.empty:
        return _empty_normalized_frame()
    columns = ['dataset_name', 'model_name', 'metric_name', 'metric_value']
    if 'series_id' in frame.columns:
        columns.append('series_id')
    normalized = frame[columns].copy()
    normalized['source_label'] = source_label
    normalized['task_type'] = task_type
    normalized['metric_direction'] = metric_direction or infer_metric_direction(metric_name or '')
    normalized['metric_value'] = pd.to_numeric(normalized['metric_value'], errors='coerce')
    return normalized.dropna(subset=['metric_value']).reset_index(drop=True)


def load_result_sources(
        sources: Sequence[Mapping[str, Any]],
        *,
        project_root: str | Path = '.',
        spec: ResultAnalysisSpec,
) -> pd.DataFrame:
    frames = []
    root = Path(project_root)
    for source in sources:
        source_kind = str(source.get('kind', 'table'))
        source_path = root / str(source['path'])
        source_label = str(source.get('source_label') or source.get('label') or spec.source_label)
        metric_name = str(source.get('metric_name') or spec.metric_name)
        task_type = str(source.get('task_type') or spec.task_type)
        metric_direction = str(source.get('metric_direction') or spec.metric_direction)
        if source_kind == 'incremental_metrics':
            frame = load_incremental_metric_records(
                source_path,
                metric_name=metric_name,
                source_label=source_label,
                task_type=task_type,
                metric_direction=metric_direction,
            )
        elif source_kind == 'aggregate_metrics':
            frame = load_aggregate_metric_records(
                source_path,
                metric_name=metric_name,
                source_label=source_label,
                task_type=task_type,
                metric_direction=metric_direction,
            )
        else:
            frame = load_result_table(source_path)
            frame = _apply_source_filters(frame, source.get('filters') or {})
            source_spec = ResultAnalysisSpec(
                metric_name=metric_name,
                metric_direction=metric_direction,
                source_label=source_label,
                task_type=task_type,
                dataset_column=str(source.get('dataset_column', spec.dataset_column)),
                model_column=str(source.get('model_column', spec.model_column)),
                value_column=str(source.get('value_column', spec.value_column)),
            )
            frame = normalize_result_table(frame, source_spec)
            strip_suffix = source.get('strip_model_suffix')
            if strip_suffix:
                frame['model_name'] = frame['model_name'].astype(str).str.replace(
                    str(strip_suffix),
                    '',
                    regex=False,
                )
        model_aliases = dict(source.get('model_aliases') or {})
        if model_aliases and not frame.empty:
            frame['model_name'] = frame['model_name'].replace(model_aliases)
        frames.append(frame)
    if not frames:
        return _empty_normalized_frame()
    return pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)


def build_coverage_frame(
        normalized: pd.DataFrame,
        *,
        expected_datasets: Sequence[str] = (),
        expected_dataset_count: int | None = None,
        source_label: str = 'benchmark',
        unit_column: str = 'dataset_name',
) -> pd.DataFrame:
    observed_column = unit_column if unit_column in normalized.columns else 'dataset_name'
    observed = tuple(sorted(set(str(item) for item in normalized.get(observed_column, ()))))
    expected = tuple(sorted(str(item) for item in expected_datasets))
    denominator = expected_dataset_count or len(expected) or len(observed)
    missing = tuple(name for name in expected if name not in observed)
    coverage = 1.0 if denominator == 0 else min(1.0, len(observed) / float(denominator))
    return pd.DataFrame(
        [
            {
                'source_label': source_label,
                'coverage_unit': observed_column,
                'expected_dataset_count': int(denominator),
                'observed_dataset_count': int(len(observed)),
                'coverage_pct': round(100.0 * coverage, 4),
                'missing_dataset_count': int(len(missing)),
                'missing_datasets': ', '.join(missing),
                'status': 'full' if coverage >= 0.999 else 'partial',
            }
        ]
    )


def build_status_summary_frame(run_records: pd.DataFrame) -> pd.DataFrame:
    if run_records.empty or 'status' not in run_records.columns:
        return pd.DataFrame(columns=['status', 'run_count', 'dataset_count', 'model_count'])
    return (
        run_records.groupby(run_records['status'].astype(str))
        .agg(
            run_count=('status', 'size'),
            dataset_count=('dataset_name', 'nunique'),
            model_count=('model_name', 'nunique'),
        )
        .rename_axis('status')
        .reset_index()
        .sort_values(['status'])
    )


def build_model_diagnostics_frame(
        run_records: pd.DataFrame,
        *,
        model_specs: Sequence[Any] = (),
) -> pd.DataFrame:
    if run_records.empty:
        return pd.DataFrame(
            columns=[
                'dataset_name',
                'model_name',
                'status',
                'selected_generators',
                'selected_generator_count',
                'important_generators',
                'n_kernels',
                'params',
                'tags',
                'message',
            ]
        )
    specs_by_name = {str(getattr(spec, 'display_name', '')): spec for spec in model_specs}
    rows = []
    for _, record in run_records.iterrows():
        model_name = str(record.get('model_name', ''))
        metadata = _as_mapping(record.get('metadata'))
        summary = _as_mapping(metadata.get('kernel_learning_summary'))
        spec = specs_by_name.get(model_name)
        selected = tuple(str(item) for item in _as_sequence(summary.get('selected_generators')))
        important = tuple(str(item) for item in _as_sequence(summary.get('important_generators')))
        rows.append(
            {
                'dataset_name': str(record.get('dataset_name', '')),
                'model_name': model_name,
                'status': str(record.get('status', '')),
                'selected_generators': ', '.join(selected),
                'selected_generator_count': int(len(selected)),
                'important_generators': ', '.join(important),
                'n_kernels': _safe_int(summary.get('n_kernels')),
                'params': json.dumps(getattr(spec, 'params', {}) if spec is not None else {}, sort_keys=True),
                'tags': ', '.join(str(item) for item in getattr(spec, 'tags', ())) if spec is not None else '',
                'message': str(record.get('message', '')),
            }
        )
    return pd.DataFrame(rows)


def build_generator_usage_frame(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty or 'selected_generators' not in diagnostics.columns:
        return pd.DataFrame(columns=['generator_name', 'selection_count', 'dataset_count', 'model_count'])
    rows = []
    for _, record in diagnostics.iterrows():
        for generator_name in _split_csv_cell(record.get('selected_generators')):
            rows.append(
                {
                    'generator_name': generator_name,
                    'dataset_name': record.get('dataset_name'),
                    'model_name': record.get('model_name'),
                }
            )
    if not rows:
        return pd.DataFrame(columns=['generator_name', 'selection_count', 'dataset_count', 'model_count'])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby('generator_name')
        .agg(
            selection_count=('generator_name', 'size'),
            dataset_count=('dataset_name', 'nunique'),
            model_count=('model_name', 'nunique'),
        )
        .reset_index()
        .sort_values(['selection_count', 'generator_name'], ascending=[False, True])
    )


def build_parameter_metric_frame(normalized: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    if normalized.empty or diagnostics.empty:
        return pd.DataFrame(columns=['dataset_name', 'model_name', 'metric_name', 'metric_value'])
    metric = normalized.groupby(['dataset_name', 'model_name', 'metric_name'], as_index=False)['metric_value'].mean()
    columns = ['dataset_name', 'model_name', 'selected_generator_count', 'n_kernels', 'params', 'tags']
    available = [column for column in columns if column in diagnostics.columns]
    merged = metric.merge(diagnostics[available].drop_duplicates(), on=['dataset_name', 'model_name'], how='left')
    return merged.sort_values(['dataset_name', 'model_name', 'metric_name']).reset_index(drop=True)


def normalize_result_table(
        frame: pd.DataFrame,
        spec: ResultAnalysisSpec,
) -> pd.DataFrame:
    """Normalize wide or long benchmark result tables into a stable long format."""
    if frame.empty:
        return _empty_normalized_frame()

    data = frame.copy()
    if _has_long_columns(data, spec):
        normalized = data.rename(
            columns={
                spec.dataset_column: 'dataset_name',
                spec.model_column: 'model_name',
                spec.value_column: 'metric_value',
            }
        )
        if 'metric_name' not in normalized.columns:
            normalized['metric_name'] = spec.metric_name
    else:
        normalized = _normalize_wide_frame(data, spec)

    result_columns = ['dataset_name', 'model_name', 'metric_name', 'metric_value']
    if 'series_id' in normalized.columns:
        result_columns.append('series_id')
    normalized = normalized[result_columns].copy()
    normalized['metric_value'] = _to_numeric_series(normalized['metric_value'])
    normalized = normalized.dropna(subset=['dataset_name', 'model_name', 'metric_value'])
    normalized['dataset_name'] = normalized['dataset_name'].astype(str)
    normalized['model_name'] = normalized['model_name'].astype(str)
    normalized['metric_name'] = normalized['metric_name'].astype(str)
    normalized['source_label'] = spec.source_label
    normalized['task_type'] = spec.task_type
    normalized['metric_direction'] = 'higher' if spec.higher_is_better else 'lower'
    return normalized.reset_index(drop=True)


def combine_result_tables(
        tables: Iterable[pd.DataFrame],
        specs: Sequence[ResultAnalysisSpec],
) -> pd.DataFrame:
    normalized_frames = [
        normalize_result_table(frame, spec)
        for frame, spec in zip(tables, specs)
    ]
    if not normalized_frames:
        return _empty_normalized_frame()
    return pd.concat(normalized_frames, ignore_index=True)


def build_best_per_dataset_frame(
        normalized: pd.DataFrame,
        *,
        metric_direction: str | None = None,
) -> pd.DataFrame:
    if normalized.empty:
        return pd.DataFrame(
            columns=[
                'dataset_name',
                'metric_name',
                'best_model',
                'best_metric_value',
                'source_label',
                'task_type',
                'metric_direction',
            ]
        )
    direction = metric_direction or str(normalized['metric_direction'].dropna().iloc[0])
    ascending = direction == 'lower'
    ordered = normalized.sort_values(['dataset_name', 'metric_value'], ascending=[True, ascending])
    best = ordered.groupby(['dataset_name', 'metric_name'], as_index=False).head(1).reset_index(drop=True)
    return (
        best.assign(best_model=best['model_name'], best_metric_value=best['metric_value'])[
            [
                'dataset_name',
                'metric_name',
                'best_model',
                'best_metric_value',
                'source_label',
                'task_type',
                'metric_direction',
            ]
        ]
        .sort_values(['dataset_name', 'metric_name'])
        .reset_index(drop=True)
    )


def build_mean_rank_frame(
        normalized: pd.DataFrame,
        *,
        metric_direction: str | None = None,
) -> pd.DataFrame:
    if normalized.empty:
        return pd.DataFrame(columns=['model_name', 'mean_rank', 'dataset_count'])
    direction = metric_direction or str(normalized['metric_direction'].dropna().iloc[0])
    ascending = direction == 'lower'
    ranked = normalized.copy()
    ranked['rank'] = ranked.groupby(['dataset_name', 'metric_name'])['metric_value'].rank(
        method='average',
        ascending=ascending,
    )
    return (
        ranked.groupby('model_name')
        .agg(mean_rank=('rank', 'mean'), dataset_count=('dataset_name', 'nunique'))
        .reset_index()
        .sort_values(['mean_rank', 'model_name'])
        .reset_index(drop=True)
    )


def build_topk_summary_frame(
        normalized: pd.DataFrame,
        *,
        top_k: Sequence[int] = (1, 3, 5),
        metric_direction: str | None = None,
) -> pd.DataFrame:
    rank_frame = _ranked_result_frame(normalized, metric_direction=metric_direction)
    columns = ['model_name', *(f'top_{k}' for k in top_k), 'top_half']
    if rank_frame.empty:
        return pd.DataFrame(columns=columns)

    dataset_count = rank_frame['dataset_name'].nunique()
    half_threshold = max(1, int(rank_frame.groupby('dataset_name')['model_name'].nunique().median() / 2))
    rows = []
    for model_name, group in rank_frame.groupby('model_name'):
        row = {'model_name': model_name}
        for k in top_k:
            row[f'top_{k}'] = int((group['rank'] <= k).sum())
        row['top_half'] = int((group['rank'] <= half_threshold).sum())
        row['dataset_count'] = int(dataset_count)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    sort_columns = [f'top_{rank}' for rank in sorted(top_k) if f'top_{rank}' in result.columns]
    return result.sort_values(sort_columns + ['model_name'], ascending=[False] * len(sort_columns) + [True])


def build_dataset_delta_frame(
        normalized: pd.DataFrame,
        *,
        target_model: str | None = None,
        metric_direction: str | None = None,
) -> pd.DataFrame:
    if normalized.empty:
        return pd.DataFrame(
            columns=[
                'dataset_name',
                'target_model',
                'target_metric',
                'best_reference_model',
                'best_reference_metric',
                'improvement',
                'relative_improvement_pct',
            ]
        )

    direction = metric_direction or str(normalized['metric_direction'].dropna().iloc[0])
    higher = direction == 'higher'
    target = target_model or _infer_target_model(normalized['model_name'].unique())
    rows = []
    for dataset_name, group in normalized.groupby('dataset_name'):
        target_rows = group[group['model_name'] == target]
        if target_rows.empty:
            continue
        target_metric = float(target_rows['metric_value'].mean())
        references = group[group['model_name'] != target]
        if references.empty:
            continue
        reference_scores = references.groupby('model_name')['metric_value'].mean()
        best_reference_model = reference_scores.idxmax() if higher else reference_scores.idxmin()
        best_reference_metric = float(reference_scores.loc[best_reference_model])
        improvement = target_metric - best_reference_metric if higher else best_reference_metric - target_metric
        denominator = abs(best_reference_metric) if best_reference_metric != 0 else 1.0
        rows.append(
            {
                'dataset_name': dataset_name,
                'target_model': target,
                'target_metric': target_metric,
                'best_reference_model': str(best_reference_model),
                'best_reference_metric': best_reference_metric,
                'improvement': improvement,
                'relative_improvement_pct': 100.0 * improvement / denominator,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                'dataset_name',
                'target_model',
                'target_metric',
                'best_reference_model',
                'best_reference_metric',
                'improvement',
                'relative_improvement_pct',
            ]
        )
    return pd.DataFrame(rows).sort_values('improvement', ascending=False).reset_index(drop=True)


def build_dataset_difficulty_frame(
        normalized: pd.DataFrame,
        *,
        metric_direction: str | None = None,
) -> pd.DataFrame:
    best = build_best_per_dataset_frame(normalized, metric_direction=metric_direction)
    if best.empty:
        return pd.DataFrame(columns=['dataset_name', 'best_metric', 'difficulty'])
    direction = metric_direction or str(normalized['metric_direction'].dropna().iloc[0])
    rows = []
    for _, row in best.iterrows():
        metric_value = float(row['best_metric_value'])
        difficulty = _difficulty_label(metric_value, higher_is_better=direction == 'higher')
        rows.append(
            {
                'dataset_name': row['dataset_name'],
                'best_model': row['best_model'],
                'best_metric': metric_value,
                'difficulty': difficulty,
            }
        )
    return pd.DataFrame(rows).sort_values(['difficulty', 'dataset_name']).reset_index(drop=True)


def _has_long_columns(frame: pd.DataFrame, spec: ResultAnalysisSpec) -> bool:
    columns = set(frame.columns)
    return {spec.dataset_column, spec.model_column, spec.value_column}.issubset(columns)


def _normalize_wide_frame(frame: pd.DataFrame, spec: ResultAnalysisSpec) -> pd.DataFrame:
    dataset_column = spec.dataset_column if spec.dataset_column in frame.columns else _find_dataset_column(frame)
    if dataset_column is None:
        data = frame.reset_index().rename(columns={'index': 'dataset_name'})
        dataset_column = 'dataset_name'
    else:
        data = frame.rename(columns={dataset_column: 'dataset_name'})
        dataset_column = 'dataset_name'

    ignored_columns = {
        dataset_column,
        'dataset_name',
        'dataset',
        'Dataset',
        'metric_name',
        'metric_direction',
        'dataset_category',
        'source_label',
        'task_type',
        'Difference %',
        'Metric dispersion by dataset',
    }
    value_columns = [
        column for column in data.columns
        if column not in ignored_columns and _to_numeric_series(data[column]).notna().any()
    ]
    melted = data.melt(
        id_vars=['dataset_name'],
        value_vars=value_columns,
        var_name='model_name',
        value_name='metric_value',
    )
    melted['metric_name'] = spec.metric_name
    return melted


def _apply_source_filters(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for column, expected in filters.items():
        if column not in result.columns:
            continue
        result = result[result[column].astype(str) == str(expected)].copy()
    return result


def _normalize_decimal_comma_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in data.columns:
        if data[column].dtype == object:
            data[column] = data[column].map(lambda value: value.replace(',', '.') if isinstance(value, str) else value)
    return data


def _to_numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')
    return pd.to_numeric(series, errors='coerce')


def _ranked_result_frame(
        normalized: pd.DataFrame,
        *,
        metric_direction: str | None = None,
) -> pd.DataFrame:
    if normalized.empty:
        return pd.DataFrame(columns=[*normalized.columns, 'rank'])
    direction = metric_direction or str(normalized['metric_direction'].dropna().iloc[0])
    ranked = normalized.copy()
    ranked['rank'] = ranked.groupby(['dataset_name', 'metric_name'])['metric_value'].rank(
        method='min',
        ascending=direction == 'lower',
    )
    return ranked


def _find_dataset_column(frame: pd.DataFrame) -> str | None:
    for column in ('dataset_name', 'dataset', 'Dataset', 'name'):
        if column in frame.columns:
            return column
    for column in frame.columns:
        if str(column).startswith('Unnamed:') and frame[column].notna().any():
            return str(column)
    return None


def _infer_target_model(model_names: Sequence[str]) -> str:
    ordered = tuple(str(name) for name in model_names)
    for marker in ('Fedot_Industrial', 'Industrial', 'Kernel', 'PDL'):
        for name in ordered:
            if marker.lower() in name.lower():
                return name
    if not ordered:
        raise ValueError('Cannot infer target model from an empty model list.')
    return ordered[-1]


def _difficulty_label(metric_value: float, *, higher_is_better: bool) -> str:
    score = metric_value if higher_is_better else 1.0 / (1.0 + max(metric_value, 0.0))
    if score >= 0.9:
        return 'easy'
    if score >= 0.75:
        return 'normal'
    if score >= 0.5:
        return 'hard'
    return 'extra_hard'


def _empty_normalized_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            'dataset_name',
            'model_name',
            'metric_name',
            'metric_value',
            'source_label',
            'task_type',
            'metric_direction',
        ]
    )


def _empty_run_records_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            'run_id',
            'benchmark',
            'dataset_name',
            'subset',
            'series_id',
            'model_name',
            'status',
            'tags',
            'message',
            'metrics_summary',
            'metadata',
        ]
    )


def _empty_kernel_diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            'run_id',
            'benchmark',
            'dataset_name',
            'subset',
            'series_id',
            'model_name',
            'status',
            'kernel_diagnostics',
            'kernel_selection',
            'summary',
        ]
    )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, MappingABC) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, SequenceABC):
        return value
    return (value,)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _split_csv_cell(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in str(value).split(',') if item.strip())


__all__ = [
    'HIGHER_IS_BETTER_METRICS',
    'ResultAnalysisSpec',
    'build_best_per_dataset_frame',
    'build_coverage_frame',
    'build_dataset_delta_frame',
    'build_dataset_difficulty_frame',
    'build_generator_usage_frame',
    'build_mean_rank_frame',
    'build_model_diagnostics_frame',
    'build_parameter_metric_frame',
    'build_status_summary_frame',
    'build_topk_summary_frame',
    'combine_result_tables',
    'infer_metric_direction',
    'load_aggregate_metric_records',
    'load_incremental_kernel_diagnostics',
    'load_incremental_metric_records',
    'load_incremental_run_records',
    'load_jsonl_table',
    'load_result_table',
    'load_result_sources',
    'normalize_result_table',
]
