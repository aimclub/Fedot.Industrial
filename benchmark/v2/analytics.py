from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    ArtifactRecord,
    ForecastingBenchmarkResult,
    DetectionBenchmarkResult,
    MetricRecord,
    PredictionRecord,
    RunStatus,
    ensure_directory,
    to_plain_data,
    write_json,
)
from .markdown import dataframe_to_markdown
from .verbosity import (
    ForecastingVerbosityPolicy,
    DetectionVerbosityPolicy,
    resolve_forecasting_verbosity_policy,
    resolve_detection_verbosity_policy,
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


def _resolve_detection_result_verbosity_policy(result: DetectionBenchmarkResult) -> DetectionVerbosityPolicy:
    return resolve_detection_verbosity_policy(
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


def predictions_to_frame(records: tuple[Any, ...]) -> pd.DataFrame:
    return pd.DataFrame([to_plain_data(record) for record in records])


def metrics_to_frame(records: tuple[MetricRecord, ...]) -> pd.DataFrame:
    return pd.DataFrame([to_plain_data(record) for record in records])


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

def detection_runs_to_frame(result: DetectionBenchmarkResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in result.run_records:
        metadata = dict(record.metadata or {})
        row = {
            'run_id': record.run_id,
            'benchmark': record.benchmark,
            'dataset_name': record.dataset_name,
            'subset': record.subset,
            'series_id': record.series_id,
            'model_name': record.model_name,
            'adapter_name': metadata.get('adapter_name'),
            'model_adapter_family': metadata.get('model_adapter_family'),
            'canonical_name': metadata.get('canonical_name'),
            'family': metadata.get('family'),
            'threshold': metadata.get('threshold'),
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


def build_detection_family_summary_frame(
        result: DetectionBenchmarkResult,
        primary_metric: str | None = None,
) -> pd.DataFrame:
    """
    как отработало каждое семейство моделей?
    семейство детектора:
    feature_baseline
    neural_reconstruction
    baseline
    """
    # result.run_records -> таблица, один запуск модели на одной серии
    runs_frame = detection_runs_to_frame(result)
    if runs_frame.empty:
        return pd.DataFrame() # если запусков нет, возвращаем пустую таблицу

    # family summary считает качество моделей только на успешных запусках
    successful = runs_frame[runs_frame['status'] == RunStatus.SUCCESS.value].copy()
    if successful.empty:
        return pd.DataFrame()
    metric_names = [metric for metric in result.config.metrics if metric in successful.columns]
    # frame = regime_frame.copy()
    # frame['best_primary_metric'] = pd.to_numeric(frame['best_primary_metric'], errors='coerce')
    # frame['routing_confidence'] = pd.to_numeric(frame['routing_confidence'], errors='coerce')
    rows: list[dict[str, Any]] = []
    
    for group_key, group in successful.groupby(['benchmark', 'dataset_name', 'family'], dropna=False):
        benchmark, dataset_name, family = group_key
        row = {
            'benchmark': benchmark,
            'dataset_name': dataset_name,
            'family': family,
            'n_runs': int(len(group)),
            'n_models': int(group['model_name'].nunique()),
        }
        # среднее значение метрики внутри family-группы
        for metric_name in metric_names:
            row[f'mean_{metric_name}'] = float(group[metric_name].mean())
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values(
        ['benchmark', 'dataset_name', 'family'],
        kind='stable',
    ).reset_index(drop=True)


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


def _extract_detection_stage_payload(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """извлечение detection stage payload из metadata"""
    stage_payload = dict(metadata.get('stage_diagnostics') or {})
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


def build_detection_stage_frame(result: DetectionBenchmarkResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in result.run_records:
        # только успешные запуски
        if record.status is not RunStatus.SUCCESS:
            continue
        # metadata['stage_diagnostics']
        stage_payload = _extract_detection_stage_payload(record.metadata)
        if stage_payload is None:
            continue

        data_quality = dict(stage_payload.get('data_quality') or {})
        regime_segmentation = dict(stage_payload.get('regime_segmentation') or {})
        representation = dict(stage_payload.get('representation') or {})
        anomaly_scoring = dict(stage_payload.get('anomaly_scoring') or {})
        calibration = dict(stage_payload.get('calibration') or {})
        event_aggregation = dict(stage_payload.get('event_aggregation') or {})
        transfer_alignment = dict(stage_payload.get('transfer_alignment') or {})
        interpretation = dict(stage_payload.get('interpretation') or {})


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
                'canonical_name': record.metadata.get('canonical_name'),
                'family': record.metadata.get('family') or record.metadata.get('model_adapter_family'),
                'threshold': record.metadata.get('threshold'),
                'n_samples': data_quality.get('n_samples'),
                'n_channels': data_quality.get('n_channels'),
                'window_size': data_quality.get('window_size'),
                'stride': data_quality.get('stride'),
                'n_regime_segments': regime_segmentation.get('n_segments'),
                'segment_labels_json': _stringify_stage_value(regime_segmentation.get('segment_labels')),
                'representation_mode': representation.get('representation_mode'),
                'feature_shape': _stringify_stage_value(representation.get('feature_shape')),
                'anomaly_scoring_json': _stringify_stage_value(anomaly_scoring),
                'calibration_strategy': calibration.get('strategy'),
                'calibration_threshold': calibration.get('threshold'),
                'threshold_quantile': calibration.get('threshold_quantile'),
                'n_events': event_aggregation.get('n_events'),
                'min_event_length': event_aggregation.get('min_event_length'),
                'transfer_alignment_json': _stringify_stage_value(transfer_alignment),
                'risk_feature_rows': interpretation.get('risk_feature_rows'),
                'risk_feature_columns_json': _stringify_stage_value(interpretation.get('risk_feature_columns')),
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


def build_detection_leaderboard(result: DetectionBenchmarkResult) -> pd.DataFrame:
    """detection runner уже:
    какие runs успешные;
    как сгруппировать по model/dataset;
    как посчитать среднюю primary metric;
    как отсортировать;
    какой rank поставить"""
    return pd.DataFrame([to_plain_data(row) for row in result.aggregate_report.leaderboard_rows])


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
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        target_dir = ensure_directory(output_dir)
        pivot = series_predictions.pivot(index='horizon_index', columns='model_name', values='y_pred')
        truth = (
            series_predictions[['horizon_index', 'y_true']]
            .drop_duplicates()
            .sort_values('horizon_index')
            .set_index('horizon_index')
        )
        train_values = np.asarray(series_record.train_values, dtype=float) if series_record is not None else np.array(
            [])
        actual_test = np.asarray(series_record.test_values, dtype=float) if series_record is not None else truth[
            'y_true'].to_numpy()
        forecast_index = np.arange(len(train_values), len(train_values) + len(actual_test))
        history_index = np.arange(len(train_values))
        zoom_width = max(len(actual_test) * 3, min(len(train_values), 48))
        zoom_start = max(0, len(train_values) - zoom_width)

        overlay_figure, overlay_axis = plt.subplots(figsize=(12, 6))
        if len(train_values):
            overlay_axis.plot(history_index, train_values, label='train', linewidth=2.0, color='0.45')
        overlay_axis.plot(forecast_index, actual_test, label='actual', linewidth=2.5, color='black')
        for model_name in pivot.columns:
            overlay_axis.plot(forecast_index, pivot[model_name].to_numpy(), label=model_name, linewidth=1.8)
        overlay_axis.axvline(len(train_values) - 1, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
        overlay_axis.set_title(f'History and Forecast Comparison for {series_id}')
        overlay_axis.set_xlabel('Time Index')
        overlay_axis.set_ylabel('Value')
        overlay_axis.legend(frameon=False)
        overlay_axis.grid(alpha=0.2)
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_history_forecast_overlay.{extension}'
            overlay_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(overlay_figure)

        boundary_figure, boundary_axis = plt.subplots(figsize=(12, 6))
        if len(train_values):
            boundary_axis.plot(
                np.arange(zoom_start, len(train_values)),
                train_values[zoom_start:],
                label='train',
                linewidth=2.0,
                color='0.45',
            )
        boundary_axis.plot(forecast_index, actual_test, label='actual', linewidth=2.5, color='black')
        for model_name in pivot.columns:
            boundary_axis.plot(forecast_index, pivot[model_name].to_numpy(), label=model_name, linewidth=1.8)
        boundary_axis.axvline(len(train_values) - 1, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
        boundary_axis.set_title(f'Boundary Zoom for {series_id}')
        boundary_axis.set_xlabel('Time Index')
        boundary_axis.set_ylabel('Value')
        boundary_axis.legend(frameon=False, ncol=2)
        boundary_axis.grid(alpha=0.2)
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_boundary_zoom.{extension}'
            boundary_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(boundary_figure)

        delta_figure, delta_axis = plt.subplots(figsize=(10, 5))
        last_train_value = float(train_values[-1]) if len(train_values) else 0.0
        labels = ['actual'] + list(pivot.columns)
        deltas = [float(actual_test[0] - last_train_value)] if len(actual_test) else [0.0]
        deltas.extend(float(pivot[model_name].iloc[0] - last_train_value) for model_name in pivot.columns)
        delta_axis.bar(labels, deltas, color=['black'] + [f'C{index % 10}' for index in range(len(pivot.columns))])
        delta_axis.axhline(0.0, color='0.4', linestyle='--', linewidth=1.0)
        delta_axis.set_title(f'First-Step Forecast Delta for {series_id}')
        delta_axis.set_xlabel('Series / Model')
        delta_axis.set_ylabel('Delta from Last Train Value')
        delta_axis.grid(alpha=0.2, axis='y')
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_forecast_delta.{extension}'
            delta_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(delta_figure)

        residual_figure, residual_axis = plt.subplots(figsize=(9, 5))
        for model_name in pivot.columns:
            residual_axis.plot(
                forecast_index,
                actual_test - pivot[model_name].to_numpy(),
                label=model_name,
                linewidth=1.6,
            )
        residual_axis.axhline(0.0, color='black', linestyle='--', linewidth=1)
        residual_axis.set_title(f'Residuals for {series_id}')
        residual_axis.set_xlabel('Horizon')
        residual_axis.set_ylabel('Residual')
        residual_axis.legend(frameon=False)
        residual_axis.grid(alpha=0.2)
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_residuals.{extension}'
            residual_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(residual_figure)

        horizon_metrics = metrics[
            (metrics['series_id'] == series_id)
            & (metrics['metric_name'] == result.aggregate_report.primary_metric)
            & (metrics['horizon_index'].notna())
        ].copy()
        if not horizon_metrics.empty:
            horizon_figure, horizon_axis = plt.subplots(figsize=(9, 5))
            for model_name, group in horizon_metrics.groupby('model_name'):
                horizon_axis.plot(group['horizon_index'], group['metric_value'], label=model_name, linewidth=1.8)
            horizon_axis.set_title(f'Horizon Error Profile for {series_id}')
            horizon_axis.set_xlabel('Horizon')
            horizon_axis.set_ylabel(result.aggregate_report.primary_metric.upper())
            horizon_axis.legend(frameon=False)
            horizon_axis.grid(alpha=0.2)
            for extension in ('png', 'svg'):
                path = target_dir / f'{series_id}_horizon_profile.{extension}'
                horizon_figure.savefig(path, dpi=200, bbox_inches='tight')
                artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
            plt.close(horizon_figure)

        series_run_records = [record for record in result.run_records if record.series_id == series_id]
        regime_reference = next(
            (record for record in series_run_records if record.metadata.get('regime_diagnostics')),
            None,
        )
        if regime_reference is not None:
            model_summaries = []
            for record in sorted(series_run_records, key=lambda item: item.model_name):
                model_summaries.append(
                    {
                        'model_name': record.model_name,
                        'adapter_name': record.metadata.get('adapter_name'),
                        'status': record.status.value,
                        'primary_metric': record.metrics_summary.get(result.aggregate_report.primary_metric),
                    }
                )
            regime_payload = {
                'series_id': series_id,
                'dataset_name': dataset_name,
                'primary_metric': result.aggregate_report.primary_metric,
                'regime_diagnostics': regime_reference.metadata.get('regime_diagnostics', {}),
                'routing_recommendation': regime_reference.metadata.get('routing_recommendation', {}),
                'model_outcomes': model_summaries,
            }
            diagnostics_path = target_dir / f'{series_id}_regime_diagnostics.json'
            write_json(diagnostics_path, _prune_generic_artifact_payload(regime_payload, verbosity_policy))
            artifact_manifest.append(ArtifactRecord(kind='structured', path=str(diagnostics_path), format='json'))

        stage_records = [
            record for record in series_run_records
            if record.status is RunStatus.SUCCESS and _extract_forecasting_stage_payload(record.metadata) is not None
        ]
        if stage_records:
            stage_payload = {}
            for record in stage_records:
                stage_payload[record.model_name] = {
                    'adapter_name': record.metadata.get('adapter_name'),
                    'model_adapter_family': record.metadata.get('model_adapter_family'),
                    'routing_recommendation_family': record.metadata.get('routing_recommendation_family'),
                    **(_extract_forecasting_stage_payload(record.metadata) or {}),
                }
            stage_path = target_dir / f'{series_id}_forecasting_stage_diagnostics.json'
            write_json(stage_path, _prune_generic_artifact_payload(stage_payload, verbosity_policy))
            artifact_manifest.append(ArtifactRecord(kind='structured', path=str(stage_path), format='json'))

        hybrid_records = [
            record for record in series_run_records
            if record.status is RunStatus.SUCCESS and _is_hybrid_ensemble_record(record.metadata)
        ]
        if hybrid_records:
            hybrid_payload = {
                record.model_name: _prune_generic_artifact_payload(record.metadata, verbosity_policy)
                for record in hybrid_records
            }
            hybrid_path = target_dir / f'{series_id}_hybrid_ensemble_diagnostics.json'
            write_json(hybrid_path, hybrid_payload)
            artifact_manifest.append(ArtifactRecord(kind='structured', path=str(hybrid_path), format='json'))

        stage_tuning_records = [
            record for record in series_run_records
            if record.status is RunStatus.SUCCESS and _extract_stage_tuning_report(record.metadata) is not None
        ]
        if stage_tuning_records:
            stage_tuning_payload = {
                record.model_name: {
                    'adapter_name': record.metadata.get('adapter_name'),
                    'model_adapter_family': record.metadata.get('model_adapter_family'),
                    'routing_recommendation_family': record.metadata.get('routing_recommendation_family'),
                    **(
                        verbosity_policy.prune_stage_tuning_report(
                            _extract_stage_tuning_report(record.metadata) or {}
                        ) or {}
                    ),
                }
                for record in stage_tuning_records
            }
            stage_tuning_path = target_dir / f'{series_id}_forecasting_stage_tuning.json'
            write_json(stage_tuning_path, stage_tuning_payload)
            artifact_manifest.append(ArtifactRecord(kind='structured', path=str(stage_tuning_path), format='json'))

        okhs_records = [
            record for record in result.run_records
            if record.series_id == series_id and record.status is RunStatus.SUCCESS
            and 'okhs' in record.model_name.lower()
        ]
        if okhs_records:
            diagnostics_payload = {
                record.model_name: _prune_generic_artifact_payload(record.metadata, verbosity_policy)
                for record in okhs_records
            }
            diagnostics_path = target_dir / f'{series_id}_okhs_diagnostics.json'
            write_json(diagnostics_path, diagnostics_payload)
            artifact_manifest.append(ArtifactRecord(kind='structured', path=str(diagnostics_path), format='json'))

            okhs_fdmd_stage_payload = {}
            for record in okhs_records:
                stage_payload = _extract_okhs_fdmd_stage_payload(record.metadata)
                if stage_payload is None:
                    continue
                okhs_fdmd_stage_payload[record.model_name] = {
                    'adapter_name': record.metadata.get('adapter_name'),
                    'model_adapter_family': record.metadata.get('model_adapter_family'),
                    'routing_recommendation_family': record.metadata.get('routing_recommendation_family'),
                    **stage_payload,
                }
            if okhs_fdmd_stage_payload:
                stage_diagnostics_path = target_dir / f'{series_id}_okhs_fdmd_stage_diagnostics.json'
                write_json(
                    stage_diagnostics_path,
                    _prune_generic_artifact_payload(okhs_fdmd_stage_payload, verbosity_policy),
                )
                artifact_manifest.append(
                    ArtifactRecord(kind='structured', path=str(stage_diagnostics_path), format='json')
                )

            for record in okhs_records:
                fit_diagnostics = record.metadata.get('fdmd_fit_diagnostics', {})
                if not fit_diagnostics:
                    continue
                eigen_real = np.asarray(fit_diagnostics.get('eigenvalues_real', []), dtype=float)
                eigen_imag = np.asarray(fit_diagnostics.get('eigenvalues_imag', []), dtype=float)
                mode_norms = np.asarray(fit_diagnostics.get('mode_norms', []), dtype=float)
                prediction_diagnostics = record.metadata.get('fdmd_prediction_diagnostics', {})
                model_slug = _slugify_model_name(record.model_name)

                mode_figure, mode_axes = plt.subplots(1, 2, figsize=(12, 5))
                if len(eigen_real):
                    mode_axes[0].scatter(eigen_real, eigen_imag, alpha=0.85)
                mode_axes[0].axvline(0.0, color='0.5', linestyle='--', linewidth=1.0)
                mode_axes[0].set_title(f'Eigenvalues: {record.model_name}')
                mode_axes[0].set_xlabel('Re(lambda)')
                mode_axes[0].set_ylabel('Im(lambda)')
                mode_axes[0].grid(alpha=0.2)

                if len(mode_norms):
                    mode_axes[1].bar(np.arange(len(mode_norms)), mode_norms)
                discontinuity = prediction_diagnostics.get('boundary_discontinuity_abs_mean')
                resolved_modes = fit_diagnostics.get('resolved_n_modes')
                anti_smoothing = prediction_diagnostics.get('anti_smoothing_diagnostics', {})
                collapse_detected = anti_smoothing.get('collapse_detected')
                envelope_ratio = anti_smoothing.get('envelope_ratio_before')
                mode_axes[1].set_title(
                    f'Mode Norms (resolved={resolved_modes}, jump={discontinuity}, '
                    f'collapse={collapse_detected}, env={envelope_ratio})'
                )
                mode_axes[1].set_xlabel('Mode Index')
                mode_axes[1].set_ylabel('Norm')
                mode_axes[1].grid(alpha=0.2, axis='y')

                for extension in ('png', 'svg'):
                    path = target_dir / f'{series_id}_{model_slug}_okhs_modes.{extension}'
                    mode_figure.savefig(path, dpi=200, bbox_inches='tight')
                    artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                plt.close(mode_figure)

        havok_records = [
            record for record in result.run_records
            if record.series_id == series_id and record.status is RunStatus.SUCCESS
            and 'havok' in record.model_name.lower()
        ]
        if havok_records:
            diagnostics_payload = {
                record.model_name: _prune_generic_artifact_payload(record.metadata, verbosity_policy)
                for record in havok_records
            }
            diagnostics_path = target_dir / f'{series_id}_havok_diagnostics.json'
            write_json(diagnostics_path, diagnostics_payload)
            artifact_manifest.append(ArtifactRecord(kind='structured', path=str(diagnostics_path), format='json'))

            for record in havok_records:
                model_slug = _slugify_model_name(record.model_name)
                forcing_values = np.asarray(record.metadata.get('forcing_values', []), dtype=float)
                forcing_threshold = float(record.metadata.get('forcing_threshold', 0.0))
                forcing_mask = np.asarray(record.metadata.get('forcing_mask', []), dtype=bool)
                forcing_intervals = record.metadata.get('forcing_active_intervals') or [
                    [int(start), int(stop)] for start, stop in _intervals_from_mask(forcing_mask)
                ]
                forecast_forcing_values = np.asarray(record.metadata.get('forecast_forcing_values', []), dtype=float)
                forecast_forcing_mask = np.asarray(record.metadata.get('forecast_forcing_mask', []), dtype=bool)

                forcing_figure, forcing_axis = plt.subplots(figsize=(11, 4.5))
                if len(forcing_values):
                    forcing_axis.plot(
                        np.arange(len(forcing_values)),
                        forcing_values,
                        linewidth=1.8,
                        color='C0',
                        label='forcing',
                    )
                if forcing_threshold > 0:
                    forcing_axis.axhline(forcing_threshold, color='C3', linestyle='--', linewidth=1.0)
                    forcing_axis.axhline(-forcing_threshold, color='C3', linestyle='--', linewidth=1.0)
                for start, stop in forcing_intervals:
                    forcing_axis.axvspan(start, stop + 1, color='C3', alpha=0.12)
                forcing_axis.set_title(f'HAVOK Forcing Timeline: {record.model_name}')
                forcing_axis.set_xlabel('Latent Step')
                forcing_axis.set_ylabel('Forcing')
                forcing_axis.grid(alpha=0.2)
                if len(forcing_values):
                    forcing_axis.legend(frameon=False)
                for extension in ('png', 'svg'):
                    path = target_dir / f'{series_id}_{model_slug}_havok_forcing_timeline.{extension}'
                    forcing_figure.savefig(path, dpi=200, bbox_inches='tight')
                    artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                plt.close(forcing_figure)

                if record.model_name in pivot.columns:
                    event_figure, event_axis = plt.subplots(figsize=(12, 6))
                    if len(train_values):
                        event_axis.plot(history_index, train_values, label='train', linewidth=2.0, color='0.45')
                    event_axis.plot(forecast_index, actual_test, label='actual', linewidth=2.5, color='black')
                    event_axis.plot(
                        forecast_index,
                        pivot[record.model_name].to_numpy(),
                        label=record.model_name,
                        linewidth=1.8,
                        color='C0',
                    )
                    if len(forecast_forcing_mask):
                        for step_index, is_active in enumerate(forecast_forcing_mask[:len(forecast_index)]):
                            if is_active:
                                event_axis.axvspan(
                                    forecast_index[step_index] - 0.5,
                                    forecast_index[step_index] + 0.5,
                                    color='C3',
                                    alpha=0.12,
                                )
                    event_axis.axvline(len(train_values) - 1, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
                    event_axis.set_title(f'HAVOK Event Overlay for {series_id}')
                    event_axis.set_xlabel('Time Index')
                    event_axis.set_ylabel('Value')
                    event_axis.legend(frameon=False)
                    event_axis.grid(alpha=0.2)
                    for extension in ('png', 'svg'):
                        path = target_dir / f'{series_id}_{model_slug}_havok_event_overlay.{extension}'
                        event_figure.savefig(path, dpi=200, bbox_inches='tight')
                        artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                    plt.close(event_figure)

    return SeriesComparisonResult(
        series_id=series_id,
        dataset_name=dataset_name,
        model_names=tuple(sorted(series_predictions['model_name'].unique())),
        metrics_table=series_metrics.reset_index(drop=True),
        prediction_table=series_predictions.reset_index(drop=True),
        artifact_manifest=tuple(artifact_manifest),
    )


def render_detection_series_artifacts(
        result: DetectionBenchmarkResult,
        series_dir: Path,
) -> tuple[ArtifactRecord, ...]:
    """
    Обработка результата:
    1. Группировка по series_id и model_name.
    2. Для каждой группы сохранить таблицу:
        sample_index;
        timestamp;
        y_true;
        y_pred;
        y_score.
    3. Соответствующий run_record.
    4. Сохранить diagnostics.json:
        run_id;
        series_id;
        model_name;
        metrics_summary;
        stage_diagnostics;
        regime_diagnostics;
        threshold;
        canonical_name;
        family.
    5. Построить график:
        y_score;
        y_true;
        y_pred;
        threshold.
    """
    manifest: list[ArtifactRecord] = []
    predictions_frame = predictions_to_frame(result.prediction_records)
    if predictions_frame.empty:
        return tuple(manifest)
    # поля для per-series artifacts
    required_columns = {'benchmark', 'dataset_name', 'subset', 'series_id', 'model_name'}
    missing_columns = required_columns - set(predictions_frame.columns)
    if missing_columns:
        raise ValueError(f'Detection predictions are missing required columns: {sorted(missing_columns)}')

    metrics_frame = metrics_to_frame(result.metric_records)
    # для каждой predictions найти соответствующий BenchmarkRunRecord
    run_lookup = {
        (
            record.benchmark,
            record.dataset_name,
            record.subset,
            record.series_id,
            record.model_name,
        ): record
        for record in result.run_records
    }
    group_columns = ['benchmark', 'dataset_name', 'subset', 'series_id', 'model_name']
    
    for group_key, group in predictions_frame.groupby(group_columns, dropna=False, sort=False):
        # benchmark + dataset_name + subset + series_id + model_name
        benchmark, dataset_name, subset, series_id, model_name = (str(value) for value in group_key)
        model_slug = _slugify_model_name(model_name)
        target_dir = ensure_directory(series_dir / series_id / model_slug)

        prediction_table = group.sort_values('sample_index', kind='stable').reset_index(drop=True)
        prediction_path = target_dir / 'predictions.csv'
        # debug artifact для конкретного ряда (sample_index, timestamp, y_true, y_pred, y_score)
        prediction_table.to_csv(prediction_path, index=False)
        # metrics.csv метрики модели на серии
        manifest.append(ArtifactRecord(kind='table', path=str(prediction_path), format='csv'))
        series_metrics = metrics_frame[
            (metrics_frame['benchmark'].astype(str) == benchmark)
            & (metrics_frame['dataset_name'].astype(str) == dataset_name)
            & (metrics_frame['subset'].astype(str) == subset)
            & (metrics_frame['series_id'].astype(str) == series_id)
            & (metrics_frame['model_name'].astype(str) == model_name)
        ].copy() if not metrics_frame.empty else pd.DataFrame()
        if not series_metrics.empty:
            metrics_path = target_dir / 'metrics.csv'
            series_metrics.to_csv(metrics_path, index=False)
            manifest.append(ArtifactRecord(kind='table', path=str(metrics_path), format='csv'))
        run_record = run_lookup.get((benchmark, dataset_name, subset, series_id, model_name))
        metadata = dict(run_record.metadata or {}) if run_record is not None else {}
        # diagnostics (run, модель, серия, summary metrics, threshold, stage diagnostics, regime diagnostics)
        diagnostics_payload = {
            'run_id': result.run_id,
            'benchmark': benchmark,
            'dataset_name': dataset_name,
            'subset': subset,
            'series_id': series_id,
            'model_name': model_name,
            'status': run_record.status.value if run_record is not None else None,
            'message': run_record.message if run_record is not None else '',
            'metrics_summary': dict(run_record.metrics_summary) if run_record is not None else {},
            'canonical_name': metadata.get('canonical_name'),
            'family': metadata.get('family') or metadata.get('model_adapter_family'),
            'adapter_name': metadata.get('adapter_name'),
            'threshold': metadata.get('threshold'),
            'stage_diagnostics': metadata.get('stage_diagnostics'),
            'regime_diagnostics': metadata.get('regime_diagnostics'),
            'routing_recommendation': metadata.get('routing_recommendation'),
            'prediction_count': int(len(prediction_table)),
            'score_available': bool(
                'y_score' in prediction_table.columns
                and prediction_table['y_score'].notna().any()
            ),
        }
        diagnostics_path = target_dir / 'diagnostics.json'
        write_json(diagnostics_path, diagnostics_payload)
        manifest.append(ArtifactRecord(kind='structured', path=str(diagnostics_path), format='json'))

        if 'y_score' not in prediction_table.columns or not prediction_table['y_score'].notna().any():
            continue

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x_values = prediction_table['sample_index'].astype(int).to_numpy()
        scores = pd.to_numeric(prediction_table['y_score'], errors='coerce').to_numpy(dtype=float)
        true_labels = pd.to_numeric(prediction_table['y_true'], errors='coerce').fillna(0).to_numpy(dtype=int)
        predicted_labels = pd.to_numeric(prediction_table['y_pred'], errors='coerce').fillna(0).to_numpy(dtype=int)
        # anomaly score
        # threshold
        # оранжевые зоны true anomaly
        # зелёные зоны predicted anomaly
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.plot(x_values, scores, color='tab:blue', linewidth=1.4, label='Anomaly score')
        threshold = metadata.get('threshold')
        if threshold is not None:
            axis.axhline(float(threshold), color='tab:red', linestyle='--', linewidth=1.1, label='Threshold')
        true_legend_added = False
        for start, end in _intervals_from_mask(true_labels == 1):
            axis.axvspan(
                x_values[start],
                x_values[end],
                color='tab:orange',
                alpha=0.18,
                label='True anomaly' if not true_legend_added else None,
            )
            true_legend_added = True
        pred_legend_added = False
        for start, end in _intervals_from_mask(predicted_labels == 1):
            axis.axvspan(
                x_values[start],
                x_values[end],
                color='tab:green',
                alpha=0.14,
                label='Predicted anomaly' if not pred_legend_added else None,
            )
            pred_legend_added = True
        axis.set_title(f'Detection score timeline | {dataset_name} | {series_id} | {model_name}')
        axis.set_xlabel('Sample index')
        axis.set_ylabel('Anomaly score')
        axis.grid(alpha=0.2)
        axis.legend(loc='best')
        figure.tight_layout()
        for extension in result.config.artifact_spec.plot_formats:
            plot_path = target_dir / f'score_timeline.{extension}'
            figure.savefig(plot_path, dpi=200, bbox_inches='tight')
            manifest.append(ArtifactRecord(kind='plot', path=str(plot_path), format=extension))
        plt.close(figure)
    return tuple(manifest)


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


def render_detection_pack(
        result: DetectionBenchmarkResult,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    """
    главный artifact writer для detection
    преобразование готового DetectionBenchmarkResult: records в таблицы.
    сохранение таблиц/summary/metadata в aggregate/.
    возврат список ArtifactRecord, чтобы result знал, какие файлы были созданы.
    Per-series diagnostics: для каждой пары series_id + model_name отдельный набор артефактов
    """
    verbosity_policy = _resolve_detection_result_verbosity_policy(result)
    target_dir = ensure_directory(output_dir or Path(result.config.artifact_spec.output_dir) / result.run_id)
    aggregate_dir = ensure_directory(target_dir / 'aggregate')
    series_dir = ensure_directory(target_dir / 'series')

    manifest: list[ArtifactRecord] = []
    metrics_frame = metrics_to_frame(result.metric_records)
    predictions_frame = predictions_to_frame(result.prediction_records)
    detection_runs_frame = detection_runs_to_frame(result)
    leaderboard = build_detection_leaderboard(result)
    detection_stage_frame = build_detection_stage_frame(result)
    family_summary = build_detection_family_summary_frame(result)
    
    for base_name, frame in (
            ('metrics', metrics_frame),
            ('predictions', predictions_frame),
            ('runs', detection_runs_frame),
            ('leaderboard', leaderboard),
            ('detection_stage_diagnostics', detection_stage_frame),
            ('detection_family_summary', family_summary),
    ):
        if not frame.empty:
            manifest.extend(_stable_write_table(frame, aggregate_dir / base_name))
    manifest.extend(render_detection_series_artifacts(result, series_dir))
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
        f'# Detection Benchmark Summary: {result.run_id}',
        '',
        f'- Task type: `{result.config.task_type.value}`',
        f'- Primary metric: `{result.aggregate_report.primary_metric}`',
        f'- Successful runs: `{result.aggregate_report.status_counts.get("success", 0)}`',
        f'- Failed runs: `{result.aggregate_report.status_counts.get("failed", 0)}`',
        f'- Skipped runs: `{result.aggregate_report.status_counts.get("skipped", 0)}`',
        f'- Not available runs: `{result.aggregate_report.status_counts.get("not_available", 0)}`',
        '',
        '## Leaderboard',
        '',
    ]
    summary_lines.append(
        'No successful detection benchmark runs were recorded.'
        if leaderboard.empty
        else dataframe_to_markdown(leaderboard, index=False)
    )
    if not family_summary.empty:
        summary_lines.extend(
            [
                '',
                '## Detection Family Summary',
                '',
                dataframe_to_markdown(family_summary, index=False),
            ]
        )
    summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))
    return tuple(manifest)