from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.industrial.core import (
    ArtifactRecord,
    ForecastingBenchmarkResult,
    MetricRecord,
    PredictionRecord,
    RunStatus,
    ensure_directory,
    to_plain_data,
    write_json,
)
from benchmark.industrial.evaluation.markdown import dataframe_to_markdown
from benchmark.industrial.experiments.verbosity import (
    ForecastingVerbosityPolicy,
    resolve_forecasting_verbosity_policy,
)


@dataclass(frozen=True)
class SeriesComparisonResult:
    series_id: str
    dataset_name: str
    model_names: tuple[str, ...]
    metrics_table: pd.DataFrame
    prediction_table: pd.DataFrame
    artifact_manifest: tuple[ArtifactRecord, ...] = ()


def _resolve_result_verbosity_policy(result: ForecastingBenchmarkResult) -> ForecastingVerbosityPolicy:
    return resolve_forecasting_verbosity_policy(
        result.config.run_spec.verbosity,
        options=result.config.run_spec.verbosity_options,
    )


def _strip_nested_keys(value: Any, *, keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_nested_keys(item, keys=keys)
            for key, item in value.items()
            if str(key) not in keys
        }
    if isinstance(value, list):
        return [_strip_nested_keys(item, keys=keys) for item in value]
    if isinstance(value, tuple):
        return [_strip_nested_keys(item, keys=keys) for item in value]
    return value


def _prune_generic_artifact_payload(payload: Any, verbosity_policy: ForecastingVerbosityPolicy) -> Any:
    keys_to_strip: set[str] = set()
    if not verbosity_policy.include_progress_policy:
        keys_to_strip.add('progress_policy')
    if not verbosity_policy.include_runner_context:
        keys_to_strip.add('benchmark_runtime_context')
    if not keys_to_strip:
        return payload
    return _strip_nested_keys(payload, keys=keys_to_strip)


def _series_record_lookup(result: ForecastingBenchmarkResult) -> dict[str, Any]:
    return {record.series_id: record for record in result.series_records}


def _slugify_model_name(name: str) -> str:
    return ''.join(character.lower() if character.isalnum() else '_' for character in name).strip('_')


def _intervals_from_mask(mask: np.ndarray, offset: int = 0) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = None
    for index, is_active in enumerate(np.asarray(mask, dtype=bool).reshape(-1)):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            intervals.append((offset + start, offset + index - 1))
            start = None
    if start is not None:
        intervals.append((offset + start, offset + len(mask) - 1))
    return intervals


def predictions_to_frame(records: tuple[PredictionRecord, ...]) -> pd.DataFrame:
    columns = [
        'run_id',
        'benchmark',
        'dataset_name',
        'subset',
        'series_id',
        'model_name',
        'horizon_index',
        'y_true',
        'y_pred',
        'status',
    ]
    return pd.DataFrame([to_plain_data(record) for record in records], columns=columns)


def metrics_to_frame(records: tuple[MetricRecord, ...]) -> pd.DataFrame:
    columns = [
        'run_id',
        'benchmark',
        'dataset_name',
        'subset',
        'series_id',
        'model_name',
        'metric_name',
        'metric_value',
        'status',
        'horizon_index',
    ]
    return pd.DataFrame([to_plain_data(record) for record in records], columns=columns)


def runs_to_frame(result: ForecastingBenchmarkResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in result.run_records:
        row = {
            'run_id': record.run_id,
            'benchmark': record.benchmark,
            'dataset_name': record.dataset_name,
            'subset': record.subset,
            'series_id': record.series_id,
            'model_name': record.model_name,
            'adapter_name': record.metadata.get('adapter_name'),
            'status': record.status.value,
            'message': record.message,
        }
        row.update(record.metrics_summary)
        rows.append(row)
    return pd.DataFrame(rows)


def _routing_aliases(adapter_name: str | None) -> set[str]:
    normalized = str(adapter_name or '').lower()
    aliases = {normalized}
    if normalized == 'ssa_forecaster':
        aliases.add('ssa_compat')
    return aliases


def build_regime_diagnostics_frame(
        result: ForecastingBenchmarkResult,
        primary_metric: str | None = None,
) -> pd.DataFrame:
    metric_name = primary_metric or result.aggregate_report.primary_metric
    rows: list[dict[str, Any]] = []
    grouped_records: dict[tuple[str, str, str, str], list[Any]] = {}
    for record in result.run_records:
        key = (record.benchmark, record.dataset_name, record.subset, record.series_id)
        grouped_records.setdefault(key, []).append(record)

    for (benchmark, dataset_name, subset, series_id), series_runs in grouped_records.items():
        reference = next((record for record in series_runs if record.metadata.get('regime_diagnostics')), None)
        if reference is None:
            continue
        diagnostics = dict(reference.metadata.get('regime_diagnostics', {}))
        routing = dict(reference.metadata.get('routing_recommendation', {}))
        successful = [
            record for record in series_runs
            if record.status is RunStatus.SUCCESS and metric_name in record.metrics_summary
        ]
        successful.sort(key=lambda record: float(record.metrics_summary.get(metric_name, float('inf'))))
        best_record = successful[0] if successful else None
        recommended_adapter = str(routing.get('primary_adapter', ''))
        recommended_candidates = tuple(str(item) for item in routing.get('candidate_adapters', ()))
        best_adapter = str(best_record.metadata.get('adapter_name', '')) if best_record is not None else ''
        recommended_family = reference.metadata.get('routing_recommendation_family')
        best_adapter_family = best_record.metadata.get('model_adapter_family') if best_record is not None else None
        best_metric = float(best_record.metrics_summary[metric_name]) if best_record is not None else None
        recommendation_hit = bool(recommended_adapter and recommended_adapter in _routing_aliases(best_adapter))
        recommendation_available = any(
            recommended_adapter in _routing_aliases(str(record.metadata.get('adapter_name', '')))
            for record in series_runs
        )
        family_recommendation_hit = bool(
            recommended_family
            and best_adapter_family
            and str(recommended_family) == str(best_adapter_family)
        )
        rows.append(
            {
                'benchmark': benchmark,
                'dataset_name': dataset_name,
                'subset': subset,
                'series_id': series_id,
                'series_length': diagnostics.get('series_length'),
                'dominant_period': diagnostics.get('dominant_period'),
                'acf_decay_rate': diagnostics.get('acf_decay_rate'),
                'spectral_concentration': diagnostics.get('spectral_concentration'),
                'spectral_flatness': diagnostics.get('spectral_flatness'),
                'local_linearity_score': diagnostics.get('local_linearity_score'),
                'switching_score': diagnostics.get('switching_score'),
                'regime_hint': diagnostics.get('regime_hint'),
                'recommended_adapter': recommended_adapter,
                'recommended_adapter_family': recommended_family,
                'recommended_candidates': ', '.join(recommended_candidates),
                'routing_confidence': routing.get('confidence'),
                'best_model_name': best_record.model_name if best_record is not None else None,
                'best_adapter_name': best_adapter or None,
                'best_adapter_family': best_adapter_family,
                'best_primary_metric': best_metric,
                'recommendation_available_in_run': recommendation_available,
                'recommendation_matches_best_available': recommendation_hit,
                'family_recommendation_matches_best': family_recommendation_hit,
            }
        )
    return pd.DataFrame(rows)


def build_routing_family_summary_frame(
        result: ForecastingBenchmarkResult,
        primary_metric: str | None = None,
) -> pd.DataFrame:
    regime_frame = build_regime_diagnostics_frame(result, primary_metric=primary_metric)
    if regime_frame.empty:
        return pd.DataFrame(
            columns=[
                'benchmark',
                'dataset_name',
                'subset',
                'recommended_adapter_family',
                'best_adapter_family',
                'n_series',
                'family_match_rate',
                'mean_best_primary_metric',
                'mean_routing_confidence',
            ]
        )

    frame = regime_frame.copy()
    frame['best_primary_metric'] = pd.to_numeric(frame['best_primary_metric'], errors='coerce')
    frame['routing_confidence'] = pd.to_numeric(frame['routing_confidence'], errors='coerce')
    frame['family_recommendation_matches_best'] = (
        frame['family_recommendation_matches_best'].fillna(False).astype(bool)
    )

    grouped = (
        frame.groupby(
            ['benchmark', 'dataset_name', 'subset', 'recommended_adapter_family', 'best_adapter_family'],
            dropna=False,
        )
        .agg(
            n_series=('series_id', 'count'),
            family_match_rate=('family_recommendation_matches_best', 'mean'),
            mean_best_primary_metric=('best_primary_metric', 'mean'),
            mean_routing_confidence=('routing_confidence', 'mean'),
        )
        .reset_index()
        .sort_values(
            ['benchmark', 'dataset_name', 'subset', 'recommended_adapter_family', 'best_adapter_family'],
            kind='stable',
        )
        .reset_index(drop=True)
    )
    return grouped


def _extract_okhs_fdmd_stage_payload(metadata: dict[str, Any]) -> dict[str, Any] | None:
    stage_payload = _extract_forecasting_stage_payload(metadata)
    if stage_payload is None:
        return None
    required_keys = ('trajectory_transform', 'decomposition', 'rank_truncation', 'forecast_head')
    if not all(stage_payload.get(key) for key in required_keys):
        return None
    return stage_payload


def _extract_forecasting_stage_payload(metadata: dict[str, Any]) -> dict[str, Any] | None:
    stage_keys = ('trajectory_transform', 'decomposition', 'rank_truncation', 'forecast_head')
    stage_payload = {
        key: dict(metadata.get(key) or {})
        for key in stage_keys
        if metadata.get(key)
    }
    return stage_payload or None


def _stringify_stage_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(to_plain_data(value), ensure_ascii=False, sort_keys=True)
    return str(value)


def build_okhs_fdmd_stage_frame(result: ForecastingBenchmarkResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in result.run_records:
        if record.status is not RunStatus.SUCCESS:
            continue
        stage_payload = _extract_okhs_fdmd_stage_payload(record.metadata)
        if stage_payload is None:
            continue

        trajectory = stage_payload['trajectory_transform']
        decomposition = stage_payload['decomposition']
        rank_truncation = stage_payload['rank_truncation']
        forecast_head = stage_payload['forecast_head']
        fit_diagnostics = dict(forecast_head.get('fit_diagnostics') or {})
        prediction_diagnostics = dict(forecast_head.get('prediction_diagnostics') or {})
        anti_smoothing = dict(forecast_head.get('anti_smoothing') or {})

        rows.append(
            {
                'benchmark': record.benchmark,
                'dataset_name': record.dataset_name,
                'subset': record.subset,
                'series_id': record.series_id,
                'model_name': record.model_name,
                'adapter_name': record.metadata.get('adapter_name'),
                'model_adapter_family': record.metadata.get('model_adapter_family'),
                'routing_recommendation_family': record.metadata.get('routing_recommendation_family'),
                'trajectory_window_policy': trajectory.get('window_policy'),
                'trajectory_resolved_window_size': trajectory.get('resolved_window_size'),
                'trajectory_expected_overlap_ratio': trajectory.get('expected_overlap_ratio'),
                'trajectory_effective_stride': trajectory.get('effective_stride'),
                'trajectory_effective_count': trajectory.get('effective_trajectory_count'),
                'trajectory_matrix_shape_before': _stringify_stage_value(
                    trajectory.get('trajectory_matrix_shape_before')
                ),
                'trajectory_matrix_shape_after': _stringify_stage_value(
                    trajectory.get('trajectory_matrix_shape_after')
                ),
                'decomposition_representation_policy': decomposition.get('representation_policy'),
                'decomposition_projected_shape': _stringify_stage_value(decomposition.get('projected_shape')),
                'decomposition_basis_shape': _stringify_stage_value(decomposition.get('basis_shape')),
                'decomposition_decode_supported': decomposition.get('decode_supported'),
                'decomposition_decode_reconstruction_error': decomposition.get('decode_reconstruction_error'),
                'decomposition_latent_window_size': decomposition.get('latent_window_size'),
                'decomposition_latent_stride': decomposition.get('latent_stride'),
                'rank_policy': rank_truncation.get('trajectory_rank_policy'),
                'rank_selected_rank': rank_truncation.get('selected_rank'),
                'rank_raw_selected_rank': rank_truncation.get('raw_selected_rank'),
                'rank_explained_variance_retained': rank_truncation.get('explained_variance_retained'),
                'rank_compression_ratio': rank_truncation.get('compression_ratio'),
                'forecast_q': forecast_head.get('q'),
                'forecast_q_policy': forecast_head.get('q_policy'),
                'forecast_mode_selection_policy': forecast_head.get('mode_selection_policy'),
                'forecast_prediction_mode_selection_policy': forecast_head.get(
                    'prediction_mode_selection_policy'
                ),
                'forecast_boundary_alignment_policy': forecast_head.get('boundary_alignment_policy'),
                'forecast_prediction_stability_threshold': forecast_head.get('prediction_stability_threshold'),
                'forecast_resolved_n_modes': fit_diagnostics.get('resolved_n_modes'),
                'forecast_selected_prediction_modes': prediction_diagnostics.get('n_selected_prediction_modes'),
                'forecast_boundary_discontinuity_abs_mean': prediction_diagnostics.get(
                    'boundary_discontinuity_abs_mean'
                ),
                'anti_smoothing_collapse_detected': anti_smoothing.get('collapse_detected'),
                'anti_smoothing_correction_applied': anti_smoothing.get('correction_applied'),
                'anti_smoothing_collapse_resolved': anti_smoothing.get('collapse_resolved'),
                'anti_smoothing_envelope_ratio_before': anti_smoothing.get('envelope_ratio_before'),
                'anti_smoothing_envelope_ratio_after': anti_smoothing.get('envelope_ratio_after'),
            }
        )
    return pd.DataFrame(rows)


def build_forecasting_stage_frame(result: ForecastingBenchmarkResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in result.run_records:
        if record.status is not RunStatus.SUCCESS:
            continue
        stage_payload = _extract_forecasting_stage_payload(record.metadata)
        if stage_payload is None:
            continue

        trajectory = dict(stage_payload.get('trajectory_transform') or {})
        decomposition = dict(stage_payload.get('decomposition') or {})
        rank_truncation = dict(stage_payload.get('rank_truncation') or {})
        forecast_head = dict(stage_payload.get('forecast_head') or {})

        rows.append(
            {
                'benchmark': record.benchmark,
                'dataset_name': record.dataset_name,
                'subset': record.subset,
                'series_id': record.series_id,
                'model_name': record.model_name,
                'adapter_name': record.metadata.get('adapter_name'),
                'model_adapter_family': record.metadata.get('model_adapter_family'),
                'routing_recommendation_family': record.metadata.get('routing_recommendation_family'),
                'has_trajectory_transform': bool(trajectory),
                'has_decomposition': bool(decomposition),
                'has_rank_truncation': bool(rank_truncation),
                'has_forecast_head': bool(forecast_head),
                'trajectory_window_size': trajectory.get('window_size', trajectory.get('resolved_window_size')),
                'trajectory_stride': trajectory.get('stride', trajectory.get('effective_stride')),
                'trajectory_features_shape': _stringify_stage_value(
                    trajectory.get('features_shape', trajectory.get('trajectory_matrix_shape_after'))
                ),
                'decomposition_strategy': decomposition.get('strategy', decomposition.get('representation_policy')),
                'decomposition_projected_shape': _stringify_stage_value(decomposition.get('projected_shape')),
                'rank_policy': rank_truncation.get('policy', rank_truncation.get('trajectory_rank_policy')),
                'rank_selected_rank': rank_truncation.get('selected_rank'),
                'rank_explained_variance_retained': rank_truncation.get('explained_variance_retained'),
                'forecast_head_json': _stringify_stage_value(forecast_head),
                'trajectory_transform_json': _stringify_stage_value(trajectory),
                'decomposition_json': _stringify_stage_value(decomposition),
                'rank_truncation_json': _stringify_stage_value(rank_truncation),
            }
        )
    return pd.DataFrame(rows)


def _is_hybrid_ensemble_record(metadata: dict[str, Any]) -> bool:
    return (
        str(metadata.get('model_family', '')).lower() == 'hybrid_ensemble'
        or str(metadata.get('adapter_name', '')).lower() == 'hybrid_ensemble_forecaster'
    )


def build_hybrid_ensemble_frame(result: ForecastingBenchmarkResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in result.run_records:
        if record.status is not RunStatus.SUCCESS or not _is_hybrid_ensemble_record(record.metadata):
            continue
        branch_calibration = dict(record.metadata.get('branch_calibration') or {})
        ensemble_head = dict(record.metadata.get('ensemble_head') or {})
        rows.append(
            {
                'benchmark': record.benchmark,
                'dataset_name': record.dataset_name,
                'subset': record.subset,
                'series_id': record.series_id,
                'model_name': record.model_name,
                'adapter_name': record.metadata.get('adapter_name'),
                'model_adapter_family': record.metadata.get('model_adapter_family'),
                'routing_recommendation_family': record.metadata.get('routing_recommendation_family'),
                'branch_names': _stringify_stage_value(record.metadata.get('branch_names')),
                'calibration_horizon': branch_calibration.get('calibration_horizon'),
                'ensemble_weights': _stringify_stage_value(
                    record.metadata.get('ensemble_weights', ensemble_head.get('weights'))
                ),
                'branch_metrics_json': _stringify_stage_value(branch_calibration.get('branch_metrics')),
                'branch_diagnostics_json': _stringify_stage_value(branch_calibration.get('branch_diagnostics')),
                'branch_predictions_json': _stringify_stage_value(record.metadata.get('branch_predictions')),
            }
        )
    return pd.DataFrame(rows)


def _extract_stage_tuning_report(metadata: dict[str, Any]) -> dict[str, Any] | None:
    report = metadata.get('stage_tuning_report')
    if not isinstance(report, dict):
        return None
    return dict(report)


def build_stage_tuning_frame(result: ForecastingBenchmarkResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in result.run_records:
        if record.status is not RunStatus.SUCCESS:
            continue
        report = _extract_stage_tuning_report(record.metadata)
        if report is None:
            continue
        sequential_result = dict(report.get('sequential_result') or {})
        baseline = dict(report.get('baseline_evaluation') or {})
        best = dict(report.get('best_evaluation') or {})
        baseline_metric = dict(baseline.get('metric') or {})
        best_metric = dict(best.get('metric') or {})
        rows.append(
            {
                'benchmark': record.benchmark,
                'dataset_name': record.dataset_name,
                'subset': record.subset,
                'series_id': record.series_id,
                'model_name': record.model_name,
                'adapter_name': record.metadata.get('adapter_name'),
                'model_adapter_family': record.metadata.get('model_adapter_family'),
                'routing_recommendation_family': record.metadata.get('routing_recommendation_family'),
                'metric_name': best_metric.get('metric_name', baseline_metric.get('metric_name')),
                'baseline_score': baseline_metric.get('metric_value'),
                'best_score': best_metric.get('metric_value'),
                'improved': report.get('improved', best_metric.get('metric_value', float('inf')) <= baseline_metric.get(
                    'metric_value', float('inf'))),
                'stage_count': len(sequential_result.get('stage_history', [])),
                'best_parameters_json': _stringify_stage_value(sequential_result.get('best_parameters')),
                'baseline_parameters_json': _stringify_stage_value(baseline.get('parameters')),
                'best_diagnostics_json': _stringify_stage_value(best.get('diagnostics')),
            }
        )
    return pd.DataFrame(rows)


def build_stage_tuning_family_frame(result: ForecastingBenchmarkResult) -> pd.DataFrame:
    tuning_frame = build_stage_tuning_frame(result)
    if tuning_frame.empty:
        return pd.DataFrame(
            columns=[
                'benchmark',
                'dataset_name',
                'model_adapter_family',
                'metric_name',
                'n_series',
                'improvement_rate',
                'routing_family_match_rate',
                'mean_baseline_score',
                'mean_best_score',
                'mean_absolute_gain',
                'mean_relative_gain',
            ]
        )

    frame = tuning_frame.copy()
    frame['baseline_score'] = pd.to_numeric(frame['baseline_score'], errors='coerce')
    frame['best_score'] = pd.to_numeric(frame['best_score'], errors='coerce')
    frame['absolute_gain'] = frame['baseline_score'] - frame['best_score']
    denominator = frame['baseline_score'].replace(0.0, np.nan).abs()
    frame['relative_gain'] = frame['absolute_gain'] / denominator
    frame['routing_family_match'] = (
        frame['model_adapter_family'].fillna('').astype(str)
        == frame['routing_recommendation_family'].fillna('').astype(str)
    )
    frame['improved'] = frame['improved'].fillna(False).astype(bool)

    summary = (
        frame.groupby(['benchmark', 'dataset_name', 'model_adapter_family', 'metric_name'], dropna=False)
        .agg(
            n_series=('series_id', 'count'),
            improvement_rate=('improved', 'mean'),
            routing_family_match_rate=('routing_family_match', 'mean'),
            mean_baseline_score=('baseline_score', 'mean'),
            mean_best_score=('best_score', 'mean'),
            mean_absolute_gain=('absolute_gain', 'mean'),
            mean_relative_gain=('relative_gain', 'mean'),
        )
        .reset_index()
    )
    return summary.sort_values(
        ['benchmark', 'dataset_name', 'model_adapter_family', 'metric_name']
    ).reset_index(drop=True)


def build_benchmark_leaderboard(
        result: ForecastingBenchmarkResult,
        primary_metric: str | None = None,
) -> pd.DataFrame:
    metric_name = primary_metric or result.aggregate_report.primary_metric
    run_frame = runs_to_frame(result)
    if run_frame.empty:
        return pd.DataFrame(columns=['benchmark', 'dataset_name', 'model_name', metric_name, 'n_series', 'rank'])
    successful = run_frame[run_frame['status'] == RunStatus.SUCCESS.value]
    if successful.empty:
        return pd.DataFrame(columns=['benchmark', 'dataset_name', 'model_name', metric_name, 'n_series', 'rank'])
    leaderboard = (
        successful.groupby(['benchmark', 'dataset_name', 'model_name'])[metric_name]
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': metric_name, 'count': 'n_series'})
        .sort_values(metric_name)
        .reset_index(drop=True)
    )
    leaderboard['rank'] = leaderboard[metric_name].rank(method='dense')
    return leaderboard


def _stable_write_table(frame: pd.DataFrame, path_without_suffix: Path) -> list[ArtifactRecord]:
    artifacts: list[ArtifactRecord] = []
    csv_path = path_without_suffix.with_suffix('.csv')
    frame.to_csv(csv_path, index=False)
    artifacts.append(ArtifactRecord(kind='table', path=str(csv_path), format='csv'))

    tex_path = path_without_suffix.with_suffix('.tex')
    tex_path.write_text(frame.to_latex(index=False, float_format=lambda value: f'{value:.4f}'), encoding='utf-8')
    artifacts.append(ArtifactRecord(kind='table', path=str(tex_path), format='tex'))

    parquet_path = path_without_suffix.with_suffix('.parquet')
    try:
        frame.to_parquet(parquet_path, index=False)
        artifacts.append(ArtifactRecord(kind='structured', path=str(parquet_path), format='parquet'))
    except Exception:
        pass
    return artifacts


def compare_models_on_series(
        result: ForecastingBenchmarkResult,
        series_id: str,
        output_dir: str | Path | None = None,
) -> SeriesComparisonResult:
    verbosity_policy = _resolve_result_verbosity_policy(result)
    predictions = predictions_to_frame(result.prediction_records)
    metrics = metrics_to_frame(result.metric_records)

    series_predictions = predictions[predictions['series_id'] == series_id].copy()
    if series_predictions.empty:
        raise ValueError(f'No prediction records found for series_id={series_id}.')

    dataset_name = str(series_predictions['dataset_name'].iloc[0])
    series_lookup = _series_record_lookup(result)
    series_record = series_lookup.get(series_id)
    series_metrics = metrics[(metrics['series_id'] == series_id) & (metrics['horizon_index'].isna())].copy()
    series_metrics = series_metrics.sort_values(['metric_name', 'metric_value', 'model_name'])
    series_predictions = series_predictions.sort_values(['model_name', 'horizon_index'])

    artifact_manifest: list[ArtifactRecord] = []
    if output_dir is not None:
        from benchmark.industrial.visualization.series_comparison import render_series_comparison
        artifact_manifest = render_series_comparison(
            result, series_id, dataset_name, series_predictions, metrics,
            series_record, verbosity_policy, output_dir)

    return SeriesComparisonResult(
        series_id=series_id,
        dataset_name=dataset_name,
        model_names=tuple(sorted(series_predictions['model_name'].unique())),
        metrics_table=series_metrics.reset_index(drop=True),
        prediction_table=series_predictions.reset_index(drop=True),
        artifact_manifest=tuple(artifact_manifest),
    )


def render_publication_pack(
        result: ForecastingBenchmarkResult,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    verbosity_policy = _resolve_result_verbosity_policy(result)
    target_dir = ensure_directory(output_dir or Path(result.config.artifact_spec.output_dir) / result.run_id)
    aggregate_dir = ensure_directory(target_dir / 'aggregate')
    series_dir = ensure_directory(target_dir / 'series')

    manifest: list[ArtifactRecord] = []
    metrics_frame = metrics_to_frame(result.metric_records)
    predictions_frame = predictions_to_frame(result.prediction_records)
    runs_frame = runs_to_frame(result)
    leaderboard = build_benchmark_leaderboard(result)

    for base_name, frame in (
            ('metrics', metrics_frame),
            ('predictions', predictions_frame),
            ('runs', runs_frame),
            ('leaderboard', leaderboard),
    ):
        manifest.extend(_stable_write_table(frame, aggregate_dir / base_name))

    regime_frame = build_regime_diagnostics_frame(result)
    if not regime_frame.empty:
        manifest.extend(_stable_write_table(regime_frame, aggregate_dir / 'regime_diagnostics'))

        routing_evaluation = regime_frame[
            [
                'benchmark',
                'dataset_name',
                'subset',
                'series_id',
                'regime_hint',
                'recommended_adapter',
                'recommended_adapter_family',
                'best_model_name',
                'best_adapter_name',
                'best_adapter_family',
                'best_primary_metric',
                'recommendation_available_in_run',
                'recommendation_matches_best_available',
                'family_recommendation_matches_best',
            ]
        ].copy()
        manifest.extend(_stable_write_table(routing_evaluation, aggregate_dir / 'routing_evaluation'))
        manifest.extend(_stable_write_table(routing_evaluation, aggregate_dir / 'routing_family_evaluation'))
        routing_family_summary = build_routing_family_summary_frame(
            result,
            primary_metric=result.aggregate_report.primary_metric,
        )
        if not routing_family_summary.empty:
            manifest.extend(_stable_write_table(routing_family_summary, aggregate_dir / 'routing_family_summary'))

    forecasting_stage_frame = build_forecasting_stage_frame(result)
    if not forecasting_stage_frame.empty:
        manifest.extend(_stable_write_table(forecasting_stage_frame, aggregate_dir / 'forecasting_stage_diagnostics'))

    hybrid_ensemble_frame = build_hybrid_ensemble_frame(result)
    if not hybrid_ensemble_frame.empty:
        manifest.extend(_stable_write_table(hybrid_ensemble_frame, aggregate_dir / 'hybrid_ensemble_diagnostics'))

    stage_tuning_frame = build_stage_tuning_frame(result)
    if not stage_tuning_frame.empty:
        manifest.extend(_stable_write_table(stage_tuning_frame, aggregate_dir / 'forecasting_stage_tuning'))
        stage_tuning_family_frame = build_stage_tuning_family_frame(result)
        if not stage_tuning_family_frame.empty:
            manifest.extend(
                _stable_write_table(stage_tuning_family_frame,
                                    aggregate_dir / 'forecasting_stage_tuning_family_summary')
            )

    okhs_fdmd_stage_frame = build_okhs_fdmd_stage_frame(result)
    if not okhs_fdmd_stage_frame.empty:
        manifest.extend(_stable_write_table(okhs_fdmd_stage_frame, aggregate_dir / 'okhs_fdmd_stage_diagnostics'))

    metadata_path = aggregate_dir / 'run_metadata.json'
    metadata_payload = {
        'run_id': result.run_id,
        'task_type': result.config.task_type.value,
        'primary_metric': result.aggregate_report.primary_metric,
        'status_counts': result.aggregate_report.status_counts,
        'dataset_specs': [to_plain_data(spec) for spec in result.config.datasets],
        'model_specs': [to_plain_data(spec) for spec in result.config.models],
        'verbosity_policy': verbosity_policy.to_dict(),
    }
    write_json(metadata_path, metadata_payload)
    manifest.append(ArtifactRecord(kind='structured', path=str(metadata_path), format='json'))

    summary_path = aggregate_dir / 'summary.md'
    summary_lines = [
        f'# Forecasting Benchmark Summary: {result.run_id}',
        '',
        f'- Primary metric: `{result.aggregate_report.primary_metric}`',
        f'- Successful runs: `{result.aggregate_report.status_counts.get("success", 0)}`',
        f'- Failed runs: `{result.aggregate_report.status_counts.get("failed", 0)}`',
        f'- Skipped runs: `{result.aggregate_report.status_counts.get("skipped", 0)}`',
        f'- Not available runs: `{result.aggregate_report.status_counts.get("not_available", 0)}`',
        '',
        '## Leaderboard',
        '',
    ]
    if leaderboard.empty:
        summary_lines.append('No successful benchmark runs were recorded.')
    else:
        summary_lines.append(dataframe_to_markdown(leaderboard, index=False))
    summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))

    successful_runs = runs_frame[runs_frame['status'] == RunStatus.SUCCESS.value].copy()
    if not successful_runs.empty:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        primary_metric = result.aggregate_report.primary_metric
        boxplot_figure, boxplot_axis = plt.subplots(figsize=(10, 5))
        successful_runs.boxplot(column=primary_metric, by='model_name', ax=boxplot_axis)
        boxplot_axis.set_title(f'{primary_metric.upper()} Distribution by Model')
        boxplot_axis.set_xlabel('Model')
        boxplot_axis.set_ylabel(primary_metric.upper())
        boxplot_axis.grid(alpha=0.2)
        boxplot_axis.figure.suptitle('')
        for extension in ('png', 'svg'):
            path = aggregate_dir / f'{primary_metric}_distribution.{extension}'
            boxplot_figure.savefig(path, dpi=200, bbox_inches='tight')
            manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(boxplot_figure)

        horizon_metrics = metrics_frame[
            (metrics_frame['metric_name'] == primary_metric) & (metrics_frame['horizon_index'].notna())
        ].copy()
        if not horizon_metrics.empty:
            horizon_plot = (
                horizon_metrics.groupby(['model_name', 'horizon_index'])['metric_value']
                .mean()
                .reset_index()
            )
            horizon_figure, horizon_axis = plt.subplots(figsize=(10, 5))
            for model_name, group in horizon_plot.groupby('model_name'):
                horizon_axis.plot(group['horizon_index'], group['metric_value'], label=model_name, linewidth=1.8)
            horizon_axis.set_title(f'Horizon vs {primary_metric.upper()}')
            horizon_axis.set_xlabel('Horizon')
            horizon_axis.set_ylabel(primary_metric.upper())
            horizon_axis.legend(frameon=False)
            horizon_axis.grid(alpha=0.2)
            for extension in ('png', 'svg'):
                path = aggregate_dir / f'horizon_vs_{primary_metric}.{extension}'
                horizon_figure.savefig(path, dpi=200, bbox_inches='tight')
                manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
            plt.close(horizon_figure)

        dataset_dir = ensure_directory(aggregate_dir / 'datasets')
        aggregate_metrics = metrics_frame[metrics_frame['horizon_index'].isna()].copy()
        for dataset_name in sorted(aggregate_metrics['dataset_name'].dropna().unique()):
            dataset_metrics = aggregate_metrics[aggregate_metrics['dataset_name'] == dataset_name].copy()
            if dataset_metrics.empty:
                continue
            dataset_slug = _slugify_model_name(str(dataset_name))

            for metric_name in sorted(dataset_metrics['metric_name'].dropna().unique()):
                metric_frame = dataset_metrics[dataset_metrics['metric_name'] == metric_name].copy()
                if metric_frame.empty or metric_frame['model_name'].nunique() == 0:
                    continue

                distribution_figure, distribution_axis = plt.subplots(figsize=(10, 5))
                metric_frame.boxplot(column='metric_value', by='model_name', ax=distribution_axis)
                distribution_axis.set_title(f'{dataset_name}: {metric_name.upper()} Distribution')
                distribution_axis.set_xlabel('Model')
                distribution_axis.set_ylabel(metric_name.upper())
                distribution_axis.figure.suptitle('')
                distribution_axis.grid(alpha=0.2)
                for extension in ('png', 'svg'):
                    path = dataset_dir / f'{dataset_slug}_metric_distribution_{metric_name}.{extension}'
                    distribution_figure.savefig(path, dpi=200, bbox_inches='tight')
                    manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                plt.close(distribution_figure)

                ranking_frame = (
                    metric_frame.groupby('model_name')['metric_value']
                    .mean()
                    .sort_values()
                    .reset_index()
                )
                ranking_figure, ranking_axis = plt.subplots(figsize=(10, 5))
                ranking_axis.bar(ranking_frame['model_name'], ranking_frame['metric_value'])
                ranking_axis.set_title(f'{dataset_name}: Mean {metric_name.upper()} by Model')
                ranking_axis.set_xlabel('Model')
                ranking_axis.set_ylabel(metric_name.upper())
                ranking_axis.tick_params(axis='x', rotation=25)
                ranking_axis.grid(alpha=0.2, axis='y')
                for extension in ('png', 'svg'):
                    path = dataset_dir / f'{dataset_slug}_model_ranking_{metric_name}.{extension}'
                    ranking_figure.savefig(path, dpi=200, bbox_inches='tight')
                    manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                plt.close(ranking_figure)

                horizon_frame = metrics_frame[
                    (metrics_frame['dataset_name'] == dataset_name)
                    & (metrics_frame['metric_name'] == metric_name)
                    & (metrics_frame['horizon_index'].notna())
                ].copy()
                if not horizon_frame.empty:
                    horizon_summary = (
                        horizon_frame.groupby(['model_name', 'horizon_index'])['metric_value']
                        .mean()
                        .reset_index()
                    )
                    dataset_horizon_figure, dataset_horizon_axis = plt.subplots(figsize=(10, 5))
                    for model_name, group in horizon_summary.groupby('model_name'):
                        dataset_horizon_axis.plot(
                            group['horizon_index'],
                            group['metric_value'],
                            label=model_name,
                            linewidth=1.8,
                        )
                    dataset_horizon_axis.set_title(f'{dataset_name}: Horizon-wise {metric_name.upper()}')
                    dataset_horizon_axis.set_xlabel('Horizon')
                    dataset_horizon_axis.set_ylabel(metric_name.upper())
                    dataset_horizon_axis.legend(frameon=False)
                    dataset_horizon_axis.grid(alpha=0.2)
                    for extension in ('png', 'svg'):
                        path = dataset_dir / f'{dataset_slug}_horizon_distribution_{metric_name}.{extension}'
                        dataset_horizon_figure.savefig(path, dpi=200, bbox_inches='tight')
                        manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                    plt.close(dataset_horizon_figure)

        okhs_rows = successful_runs[successful_runs['model_name'].str.contains('okhs', case=False, regex=False)]
        if not okhs_rows.empty:
            baseline_rows = successful_runs[
                ~successful_runs['model_name'].str.contains('okhs', case=False, regex=False)]
            pairwise_rows = []
            for _, okhs_row in okhs_rows.iterrows():
                comparable = baseline_rows[
                    (baseline_rows['benchmark'] == okhs_row['benchmark'])
                    & (baseline_rows['dataset_name'] == okhs_row['dataset_name'])
                    & (baseline_rows['series_id'] == okhs_row['series_id'])
                ]
                for _, baseline_row in comparable.iterrows():
                    pairwise_rows.append(
                        {
                            'benchmark': okhs_row['benchmark'],
                            'dataset_name': okhs_row['dataset_name'],
                            'series_id': okhs_row['series_id'],
                            'okhs_model': okhs_row['model_name'],
                            'baseline_model': baseline_row['model_name'],
                            primary_metric: okhs_row[primary_metric],
                            f'baseline_{primary_metric}': baseline_row[primary_metric],
                            'delta': okhs_row[primary_metric] - baseline_row[primary_metric],
                        }
                    )
            if pairwise_rows:
                pairwise_frame = pd.DataFrame(pairwise_rows).sort_values('delta')
                manifest.extend(_stable_write_table(pairwise_frame, aggregate_dir / 'okhs_pairwise_comparison'))

    available_series = predictions_frame['series_id'].drop_duplicates().tolist()
    for series_id in available_series[: min(3, len(available_series))]:
        comparison = compare_models_on_series(result, series_id=series_id, output_dir=series_dir / series_id)
        manifest.extend(comparison.artifact_manifest)

    return tuple(manifest)
