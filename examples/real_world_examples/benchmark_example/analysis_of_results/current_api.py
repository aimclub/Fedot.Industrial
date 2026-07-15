from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.industrial import (
    ArtifactRecord,
    ModelSpec,
    ResultAnalysisSpec,
    build_model_diagnostics_frame,
    load_incremental_run_records,
    load_result_sources,
    render_benchmark_result_analysis_pack,
    build_forecast_comparison_from_aggregate_predictions,
    render_evolution_analysis_pack,
    build_forecast_comparison_from_progress_items,
    render_forecast_comparison_pack,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[3]
DEFAULTS_PATH = PACKAGE_ROOT / 'analysis_defaults.json'
DEFAULTS_VERSION = 'industrial_real_world_analysis@1'


@lru_cache(maxsize=1)
def load_analysis_defaults(path: str | Path = DEFAULTS_PATH) -> dict[str, Any]:
    defaults_path = Path(path)
    payload = json.loads(defaults_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Analysis defaults root must be a mapping: {defaults_path}')
    version = str(payload.get('version', ''))
    if version != DEFAULTS_VERSION:
        raise ValueError(f'Unsupported analysis defaults version: {version}')
    return payload


def available_analysis_names() -> tuple[str, ...]:
    return tuple(sorted(load_analysis_defaults()['analyses']))


def build_analysis_result_frame(analysis_name: str) -> pd.DataFrame:
    analysis = _analysis_payload(analysis_name)
    try:
        frame = load_result_sources(
            tuple(analysis.get('sources') or ()),
            project_root=PROJECT_ROOT,
            spec=build_analysis_spec(analysis_name),
        )
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        frame = pd.DataFrame()
    if frame.empty:
        fallback_path = _artifact_table_path(analysis_name, 'normalized_results.csv')
        if fallback_path.exists():
            return pd.read_csv(fallback_path)
    return frame


def build_analysis_spec(analysis_name: str) -> ResultAnalysisSpec:
    analysis = _analysis_payload(analysis_name)
    return ResultAnalysisSpec(
        metric_name=str(analysis['metric_name']),
        metric_direction=str(analysis['metric_direction']),
        source_label=str(analysis['source_label']),
        task_type=str(analysis['task_type']),
    )


def build_current_model_specs(task_type: str) -> tuple[ModelSpec, ...]:
    models = load_analysis_defaults()['models'][task_type]
    return tuple(ModelSpec(**payload) for payload in models)


def build_kernel_learning_reference_model_specs(task_type: str) -> tuple[ModelSpec, ...]:
    from benchmark.experiments.kernel_learning.configs import (
        build_forecasting_kernel_learning_models,
        build_tser_kernel_learning_models,
        build_ucr_kernel_learning_models,
    )

    if task_type == 'ts_classification':
        return build_ucr_kernel_learning_models()
    if task_type == 'ts_regression':
        return build_tser_kernel_learning_models()
    if task_type == 'forecasting':
        return build_forecasting_kernel_learning_models()
    return ()


def build_ucr_two_stage_context() -> dict[str, Any]:
    from benchmark.experiments.kernel_learning.configs import KernelLearningTwoStageUCRExperimentConfig

    config = KernelLearningTwoStageUCRExperimentConfig()
    return {
        'scenario': 'ucr_two_stage',
        'data_root': str(config.data_root),
        'stage1_output_dir': str(config.stage1_output_dir),
        'stage2_output_dir': str(config.stage2_output_dir),
        'stage1_run_id': config.stage1_run_id,
        'generator_names': config.generator_names,
        'metrics': config.metrics,
        'timeout_minutes': config.timeout_minutes,
        'pop_size': config.pop_size,
    }


def render_analysis_notebook_pack(
        analysis_name: str,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    analysis = _analysis_payload(analysis_name)
    target_dir = Path(output_dir) if output_dir is not None else _artifact_root() / analysis_name
    return render_benchmark_result_analysis_pack(
        build_analysis_result_frame(analysis_name),
        output_dir=target_dir,
        spec=build_analysis_spec(analysis_name),
        target_model=str(analysis.get('target_model') or ''),
        expected_datasets=tuple(analysis.get('expected_datasets') or ()),
        expected_dataset_count=analysis.get('expected_dataset_count'),
        coverage_unit_column=str(analysis.get('coverage_unit_column') or 'dataset_name'),
        diagnostics_frame=build_analysis_diagnostics_frame(analysis_name),
        source_metadata=build_analysis_source_metadata_frame(analysis_name),
        source_expected_dataset_counts=dict(analysis.get('source_expected_dataset_counts') or {}),
        reference_source_labels=tuple(analysis.get('reference_source_labels') or ()),
        target_source_labels=tuple(analysis.get('target_source_labels') or ()),
        best_target_source_labels=tuple(analysis.get('best_target_source_labels') or ()),
    )


def build_analysis_diagnostics_frame(analysis_name: str) -> pd.DataFrame:
    analysis = _analysis_payload(analysis_name)
    frames = []
    for source in analysis.get('sources') or ():
        if str(source.get('kind')) not in {'incremental_metrics', 'aggregate_metrics'}:
            continue
        root = PROJECT_ROOT / str(source['path'])
        run_records = load_incremental_run_records(root)
        if run_records.empty and str(source.get('kind')) == 'aggregate_metrics':
            run_records = _load_aggregate_run_records(root)
        if not run_records.empty:
            frames.append(
                build_model_diagnostics_frame(
                    run_records,
                    model_specs=build_current_model_specs(str(analysis['task_type'])),
                )
            )
    if not frames:
        fallback_path = _artifact_table_path(analysis_name, 'model_diagnostics.csv')
        if fallback_path.exists():
            return pd.read_csv(fallback_path)
        return build_model_diagnostics_frame(pd.DataFrame())
    return pd.concat(frames, ignore_index=True)


def build_analysis_source_metadata_frame(analysis_name: str) -> pd.DataFrame:
    analysis = _analysis_payload(analysis_name)
    rows = []
    for source in analysis.get('sources') or ():
        source_path = PROJECT_ROOT / str(source['path'])
        fallback_path = _artifact_table_path(analysis_name, 'normalized_results.csv')
        source_exists = source_path.exists()
        rows.append(
            {
                'label': str(source.get('label') or source.get('source_label') or analysis.get('source_label')),
                'kind': str(source.get('kind', 'table')),
                'path': str(source_path if source_exists else fallback_path),
                'exists_locally': source_exists or fallback_path.exists(),
                'metric_name': str(source.get('metric_name') or analysis.get('metric_name')),
                'source_version': str(source.get('source_version', '')),
                'source_date': str(source.get('source_date', '')),
            }
        )
    return pd.DataFrame(rows)


def render_forecasting_model_comparison_pack(
        output_dir: str | Path | None = None,
        *,
        artifact_name: str = 'forecasting_model_comparison',
) -> tuple[ArtifactRecord, ...]:
    target_dir = Path(output_dir) if output_dir is not None else _artifact_root() / artifact_name
    source = load_analysis_defaults()['external_sources']['forecasting_model_comparison']
    try:
        if str(source.get('kind')) == 'aggregate_predictions':
            history, actual, forecasts, metadata = build_forecast_comparison_from_aggregate_predictions(
                PROJECT_ROOT / source['path'],
                series_id=source.get('series_id'),
                dataset_name=source.get('dataset_name'),
                model_names=tuple(source.get('model_names') or ()),
                history_length=int(source.get('history_length') or 36),
            )
        else:
            history, actual, forecasts, metadata = build_forecast_comparison_from_progress_items(
                PROJECT_ROOT / source['path'],
                series_id=source.get('series_id'),
                dataset_name=source.get('dataset_name'),
                model_names=tuple(source.get('model_names') or ()),
            )
    except ValueError as exc:
        target_dir.mkdir(parents=True, exist_ok=True)
        summary_path = target_dir / 'summary.md'
        summary_path.write_text(
            '# Forecast Comparison Unavailable\n\n'
            f'- Source: `{PROJECT_ROOT / source["path"]}`\n'
            f'- Reason: {exc}\n',
            encoding='utf-8',
        )
        return (ArtifactRecord(kind='summary', path=str(summary_path), format='md'),)
    return render_forecast_comparison_pack(
        history=history,
        actual=actual,
        forecasts=forecasts,
        output_dir=target_dir,
        title='Forecasting benchmark: Industrial and baseline model forecasts',
        series_id=str(metadata.get('series_id') or source.get('series_id') or 'm4_series'),
        source_metadata=metadata,
    )


def render_pipeline_population_pack(
        output_dir: str | Path | None = None,
        *,
        composition_root: str | Path | None = None,
        dataset_limit: int | None = None,
) -> tuple[ArtifactRecord, ...]:
    defaults = load_analysis_defaults()
    source = defaults['external_sources']['composition_results']
    root = Path(composition_root) if composition_root is not None else PROJECT_ROOT / source['path']
    target_dir = Path(output_dir) if output_dir is not None else _artifact_root() / 'pipeline_population'
    return render_evolution_analysis_pack(root, target_dir, dataset_limit=dataset_limit)


def external_source_manifest() -> dict[str, Any]:
    defaults = load_analysis_defaults()
    return {
        key: {
            **value,
            'absolute_path': str(PROJECT_ROOT / value['path']),
            'exists_locally': (PROJECT_ROOT / value['path']).exists(),
        }
        for key, value in defaults['external_sources'].items()
    }


def preflight_summary() -> dict[str, Any]:
    defaults = load_analysis_defaults()
    return {
        'version': defaults['version'],
        'available_analyses': available_analysis_names(),
        'feature_generators': tuple(defaults['feature_generators']),
        'analysis_sources': {
            name: tuple(str(source.get('path')) for source in payload.get('sources') or ())
            for name, payload in defaults['analyses'].items()
        },
        'external_sources': external_source_manifest(),
    }


def _analysis_payload(analysis_name: str) -> dict[str, Any]:
    analyses = load_analysis_defaults()['analyses']
    if analysis_name not in analyses:
        raise ValueError(f'Unknown analysis name: {analysis_name}')
    return dict(analyses[analysis_name])


def _artifact_root() -> Path:
    return PROJECT_ROOT / load_analysis_defaults()['artifact_root']


def _artifact_table_path(analysis_name: str, file_name: str) -> Path:
    return _artifact_root() / analysis_name / 'tables' / file_name


def _load_aggregate_run_records(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob('aggregate/runs.csv')):
        frame = pd.read_csv(path)
        metadata = [
            {
                'adapter_name': row.get('adapter_name'),
                'metrics_summary': {
                    metric: row.get(metric)
                    for metric in ('mase', 'smape', 'owa', 'rmse', 'mae')
                    if metric in frame.columns
                },
            }
            for _, row in frame.iterrows()
        ]
        frame['metadata'] = metadata
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


__all__ = [
    'available_analysis_names',
    'build_analysis_result_frame',
    'build_analysis_spec',
    'build_analysis_diagnostics_frame',
    'build_analysis_source_metadata_frame',
    'build_current_model_specs',
    'build_kernel_learning_reference_model_specs',
    'build_ucr_two_stage_context',
    'external_source_manifest',
    'load_analysis_defaults',
    'preflight_summary',
    'render_analysis_notebook_pack',
    'render_forecasting_model_comparison_pack',
    'render_pipeline_population_pack',
]
