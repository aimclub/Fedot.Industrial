from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from html import escape
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fedot_ind.core.models.ts_forecasting.regime_utils.regime_diagnostics import analyze_regime_diagnostics

try:  # pragma: no cover - fallback supports direct script execution without importing benchmark package root
    from .core import ArtifactRecord, ensure_directory
    from .markdown import dataframe_to_markdown
except ImportError:  # pragma: no cover
    from core import ArtifactRecord, ensure_directory  # type: ignore
    from markdown import dataframe_to_markdown  # type: ignore


LOWER_IS_BETTER_METRICS = ('owa', 'smape', 'mase', 'rmse', 'mae')
REGIME_NUMERIC_FIELDS = (
    'series_length',
    'dominant_period',
    'acf_decay_rate',
    'spectral_concentration',
    'spectral_flatness',
    'local_linearity_score',
    'switching_score',
)


@dataclass(frozen=True)
class ForecastingProgressItemPayload:
    path: Path
    payload: dict[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.payload.get('run_id', ''))

    @property
    def series_record(self) -> dict[str, Any]:
        return dict(self.payload.get('series_record', {}))

    @property
    def run_record(self) -> dict[str, Any]:
        return dict(self.payload.get('run_record', {}))

    @property
    def metric_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self.payload.get('metric_records', ()))

    @property
    def prediction_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self.payload.get('prediction_records', ()))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.run_record.get('metadata', {}))

    @property
    def dataset_name(self) -> str:
        return str(self.series_record.get('dataset_name', self.run_record.get('dataset_name', '')))

    @property
    def subset(self) -> str:
        return str(self.series_record.get('subset', self.run_record.get('subset', '')))

    @property
    def series_id(self) -> str:
        return str(self.series_record.get('series_id', self.run_record.get('series_id', '')))

    @property
    def model_name(self) -> str:
        return str(self.run_record.get('model_name', ''))

    @property
    def status(self) -> str:
        return str(self.run_record.get('status', ''))


@dataclass(frozen=True)
class ForecastingProgressVisualizationResult:
    items_frame: pd.DataFrame
    relative_gain_frame: pd.DataFrame
    relative_gain_summary: pd.DataFrame
    improvement_case_summary: pd.DataFrame
    regime_diagnostics_frame: pd.DataFrame
    regime_diagnostics_summary: pd.DataFrame
    regime_improvement_summary: pd.DataFrame
    artifact_manifest: tuple[ArtifactRecord, ...] = ()


def load_progress_item_payloads(
        items_dir: str | Path,
        *,
        model_name: str | None = None,
        status: str = 'success',
) -> tuple[ForecastingProgressItemPayload, ...]:
    directory = Path(items_dir)
    items: list[ForecastingProgressItemPayload] = []
    for path in sorted(directory.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        item = ForecastingProgressItemPayload(path=path, payload=payload)
        if model_name is not None and str(item.model_name).lower() != str(model_name).lower():
            continue
        if status and str(item.status).lower() != str(status).lower():
            continue
        items.append(item)
    return tuple(items)


def _safe_stage_tuning_report(item: ForecastingProgressItemPayload) -> dict[str, Any]:
    return dict(item.metadata.get('stage_tuning_report', {}))


def _safe_stage_tuning_comparison(item: ForecastingProgressItemPayload) -> dict[str, Any]:
    return dict(item.metadata.get('stage_tuning_comparison', {}))


def _safe_regime_diagnostics(item: ForecastingProgressItemPayload) -> dict[str, Any]:
    diagnostics = dict(item.metadata.get('regime_diagnostics', {}))
    required_keys = set(REGIME_NUMERIC_FIELDS + ('regime_hint',))
    if required_keys.issubset(diagnostics.keys()):
        return diagnostics
    history = _resolved_history(item)
    computed = analyze_regime_diagnostics(history).to_dict()
    if not diagnostics:
        return computed
    merged = dict(computed)
    merged.update({key: value for key, value in diagnostics.items() if value is not None})
    return merged


def _normalize_numeric_sequence(values: Any) -> np.ndarray:
    return np.asarray(values or (), dtype=float).reshape(-1)


def _baseline_forecast_from_prediction_records(item: ForecastingProgressItemPayload) -> np.ndarray:
    sorted_predictions = sorted(
        item.prediction_records,
        key=lambda record: int(record.get('horizon_index', 0) or 0),
    )
    return np.asarray([float(record['y_pred']) for record in sorted_predictions], dtype=float)


def _actual_forecast_target(item: ForecastingProgressItemPayload) -> np.ndarray:
    return np.asarray(item.series_record.get('test_values', ()), dtype=float).reshape(-1)


def _resolved_history(item: ForecastingProgressItemPayload) -> np.ndarray:
    return np.asarray(item.series_record.get('train_values', ()), dtype=float).reshape(-1)


def _compact_param_dict(params: dict[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(params or {}).items()
        if value is not None and str(key) != 'progress_policy'
    }


def _stringify_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:.4g}'
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_stringify_param_value(item) for item in value) + ']'
    return str(value)


def _parameter_mode_summary(
        param_dicts: list[dict[str, Any]],
        *,
        max_parameters: int = 6,
        top_k_values: int = 2,
) -> str:
    if not param_dicts:
        return 'n/a'
    keys = sorted({key for params in param_dicts for key in params.keys()})
    fragments: list[str] = []
    for key in keys[:max_parameters]:
        counter = Counter(_stringify_param_value(params[key]) for params in param_dicts if key in params)
        if not counter:
            continue
        total = sum(counter.values())
        top_values = counter.most_common(top_k_values)
        rendered = ', '.join(
            f'{value} ({count / total:.0%})'
            for value, count in top_values
        )
        fragments.append(f'{key}: {rendered}')
    return '; '.join(fragments) if fragments else 'n/a'


def _series_plot_slug(item: ForecastingProgressItemPayload) -> str:
    return f'{item.dataset_name}__{item.series_id}__{item.model_name}'.replace(' ', '_').lower()


def _extract_evaluation_fold_sequences(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    if not evaluation:
        return []
    folds = list((evaluation.get('split_metadata') or {}).get('folds', ()))
    forecast = list(evaluation.get('forecast', ()))
    target = list(evaluation.get('target', ()))
    per_fold_metric_values = list((evaluation.get('metric') or {}).get('per_fold_metric_values', ()))
    sequences: list[dict[str, Any]] = []
    cursor = 0
    for index, fold in enumerate(folds):
        fold_dict = dict(fold)
        fold_length = int(fold_dict.get('test_length', 0) or 0)
        next_cursor = cursor + fold_length
        sequences.append(
            {
                **fold_dict,
                'forecast': forecast[cursor:next_cursor],
                'target': target[cursor:next_cursor],
                'metric_value': _extract_metric_scalar(
                    fold_dict.get('metric'),
                    per_fold_metric_values[index] if index < len(per_fold_metric_values) else None,
                ),
            }
        )
        cursor = next_cursor
    return sequences


def _extract_metric_scalar(value: Any, fallback: Any = None) -> float | None:
    candidate = value
    if isinstance(candidate, dict):
        candidate = candidate.get('metric_value', fallback)
    if candidate is None:
        candidate = fallback
    if isinstance(candidate, dict):
        candidate = candidate.get('metric_value')
    if candidate is None:
        return None
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return None


def _resolve_symmetric_gain_span(values: Any, *, fallback: float = 1.0) -> float:
    gain_values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = gain_values[np.isfinite(gain_values)]
    if finite_values.size == 0:
        return float(fallback)
    max_abs_gain = float(np.max(np.abs(finite_values)))
    if not np.isfinite(max_abs_gain) or max_abs_gain <= 1e-12:
        return float(fallback)
    return max_abs_gain


def build_fold_comparison_frame(item: ForecastingProgressItemPayload) -> pd.DataFrame:
    report = _safe_stage_tuning_report(item)
    baseline_evaluation = dict(report.get('baseline_evaluation', {}))
    tuned_evaluation = dict(report.get('best_evaluation', {}))
    baseline_folds = _extract_evaluation_fold_sequences(baseline_evaluation)
    tuned_folds = _extract_evaluation_fold_sequences(tuned_evaluation)
    if not baseline_folds:
        return pd.DataFrame(
            columns=[
                'dataset_name',
                'series_id',
                'model_name',
                'fold_index',
                'train_start',
                'train_end',
                'test_start',
                'test_end',
                'train_length',
                'test_length',
                'metric_name',
                'baseline_metric',
                'tuned_metric',
                'absolute_gain',
                'relative_gain_pct',
                'improved',
            ]
        )

    metric_name = str((baseline_evaluation.get('metric') or {}).get('metric_name', 'metric')).lower()
    rows: list[dict[str, Any]] = []
    for index, baseline_fold in enumerate(baseline_folds):
        tuned_fold = tuned_folds[index] if index < len(tuned_folds) else {}
        baseline_metric = _extract_metric_scalar(baseline_fold.get('metric_value'))
        tuned_metric = _extract_metric_scalar(tuned_fold.get('metric_value'))
        absolute_gain = None
        relative_gain_pct = None
        improved = None
        if baseline_metric is not None and tuned_metric is not None:
            absolute_gain = float(baseline_metric - tuned_metric)
            relative_gain_pct = (
                float(100.0 * absolute_gain / abs(baseline_metric))
                if abs(baseline_metric) > 1e-12 else np.nan
            )
            improved = bool(absolute_gain > 0)
        rows.append(
            {
                'dataset_name': item.dataset_name,
                'series_id': item.series_id,
                'model_name': item.model_name,
                'fold_index': int(baseline_fold.get('fold_index', index + 1)),
                'train_start': int(baseline_fold.get('train_start', 0)),
                'train_end': int(baseline_fold.get('train_end', 0)),
                'test_start': int(baseline_fold.get('test_start', 0)),
                'test_end': int(baseline_fold.get('test_end', 0)),
                'train_length': int(baseline_fold.get('train_length', 0)),
                'test_length': int(baseline_fold.get('test_length', 0)),
                'metric_name': metric_name,
                'baseline_metric': baseline_metric,
                'tuned_metric': tuned_metric,
                'absolute_gain': absolute_gain,
                'relative_gain_pct': relative_gain_pct,
                'improved': improved,
            }
        )
    return pd.DataFrame(rows)


def build_relative_gain_frame(items: tuple[ForecastingProgressItemPayload, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        comparison = _safe_stage_tuning_comparison(item)
        if not comparison:
            continue
        report = _safe_stage_tuning_report(item)
        baseline_params = _compact_param_dict(
            dict((report.get('baseline_evaluation') or {}).get('parameters', {}))
        )
        tuned_params = _compact_param_dict(
            dict((report.get('best_evaluation') or {}).get('parameters', comparison.get('best_parameters', {})))
        )
        baseline_metrics = dict(comparison.get('baseline_metrics', {}))
        tuned_metrics = dict(comparison.get('tuned_metrics', {}))
        for metric_name in LOWER_IS_BETTER_METRICS:
            if metric_name not in baseline_metrics or metric_name not in tuned_metrics:
                continue
            baseline_value = float(baseline_metrics[metric_name])
            tuned_value = float(tuned_metrics[metric_name])
            absolute_gain = baseline_value - tuned_value
            relative_gain_pct = (
                100.0 * absolute_gain / abs(baseline_value)
                if abs(baseline_value) > 1e-12 else np.nan
            )
            rows.append(
                {
                    'run_id': item.run_id,
                    'dataset_name': item.dataset_name,
                    'subset': item.subset,
                    'series_id': item.series_id,
                    'model_name': item.model_name,
                    'metric_name': metric_name,
                    'baseline_value': baseline_value,
                    'tuned_value': tuned_value,
                    'absolute_gain': absolute_gain,
                    'relative_gain_pct': relative_gain_pct,
                    'improved': bool(absolute_gain > 0),
                    'baseline_params': baseline_params,
                    'tuned_params': tuned_params,
                }
            )
    return pd.DataFrame(rows)


def build_relative_gain_summary(
        gain_frame: pd.DataFrame,
        *,
        baseline_param_summary: str,
        tuned_param_summary: str,
) -> pd.DataFrame:
    if gain_frame.empty:
        return pd.DataFrame(
            columns=[
                'metric_name',
                'series_count',
                'mean_relative_gain_pct',
                'median_relative_gain_pct',
                'q25_relative_gain_pct',
                'q75_relative_gain_pct',
                'mean_absolute_gain',
                'improved_case_count',
                'not_improved_case_count',
                'improvement_rate',
                'improvement_rate_pct',
                'no_improvement_rate_pct',
                'baseline_params_summary',
                'tuned_params_summary',
            ]
        )

    rows: list[dict[str, Any]] = []
    for metric_name, frame in gain_frame.groupby('metric_name', sort=False):
        relative_values = frame['relative_gain_pct'].astype(float)
        absolute_values = frame['absolute_gain'].astype(float)
        rows.append(
            {
                'metric_name': metric_name,
                'series_count': int(len(frame)),
                'mean_relative_gain_pct': float(relative_values.mean()),
                'median_relative_gain_pct': float(relative_values.median()),
                'q25_relative_gain_pct': float(relative_values.quantile(0.25)),
                'q75_relative_gain_pct': float(relative_values.quantile(0.75)),
                'mean_absolute_gain': float(absolute_values.mean()),
                'improved_case_count': int(frame['improved'].sum()),
                'not_improved_case_count': int((~frame['improved']).sum()),
                'improvement_rate': float(frame['improved'].mean()),
                'improvement_rate_pct': float(100.0 * frame['improved'].mean()),
                'no_improvement_rate_pct': float(100.0 * (1.0 - frame['improved'].mean())),
                'baseline_params_summary': baseline_param_summary,
                'tuned_params_summary': tuned_param_summary,
            }
        )
    summary = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(LOWER_IS_BETTER_METRICS)}
    return summary.sort_values('metric_name', key=lambda series: series.map(order.get)).reset_index(drop=True)


def build_improvement_case_summary(gain_frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
        'scope',
        'metric_name',
        'total_cases',
        'improved_case_count',
        'not_improved_case_count',
        'improvement_rate_pct',
        'no_improvement_rate_pct',
    )
    if gain_frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    overall_total = int(len(gain_frame))
    overall_improved = int(gain_frame['improved'].sum())
    rows.append(
        {
            'scope': 'overall',
            'metric_name': 'all',
            'total_cases': overall_total,
            'improved_case_count': overall_improved,
            'not_improved_case_count': int(overall_total - overall_improved),
            'improvement_rate_pct': float(100.0 * overall_improved / overall_total),
            'no_improvement_rate_pct': float(100.0 * (overall_total - overall_improved) / overall_total),
        }
    )
    for metric_name, frame in gain_frame.groupby('metric_name', sort=False):
        total = int(len(frame))
        improved = int(frame['improved'].sum())
        rows.append(
            {
                'scope': 'metric',
                'metric_name': metric_name,
                'total_cases': total,
                'improved_case_count': improved,
                'not_improved_case_count': int(total - improved),
                'improvement_rate_pct': float(100.0 * improved / total),
                'no_improvement_rate_pct': float(100.0 * (total - improved) / total),
            }
        )
    order = {'all': -1, **{name: index for index, name in enumerate(LOWER_IS_BETTER_METRICS)}}
    return pd.DataFrame(rows).sort_values('metric_name', key=lambda series: series.map(order.get)).reset_index(drop=True)


def build_regime_diagnostics_frame(items: tuple[ForecastingProgressItemPayload, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        diagnostics = _safe_regime_diagnostics(item)
        rows.append(
            {
                'run_id': item.run_id,
                'dataset_name': item.dataset_name,
                'subset': item.subset,
                'series_id': item.series_id,
                'model_name': item.model_name,
                'series_length': diagnostics.get('series_length'),
                'dominant_period': diagnostics.get('dominant_period'),
                'acf_decay_rate': diagnostics.get('acf_decay_rate'),
                'spectral_concentration': diagnostics.get('spectral_concentration'),
                'spectral_flatness': diagnostics.get('spectral_flatness'),
                'local_linearity_score': diagnostics.get('local_linearity_score'),
                'switching_score': diagnostics.get('switching_score'),
                'regime_hint': diagnostics.get('regime_hint'),
            }
        )
    return pd.DataFrame(rows)


def build_regime_diagnostics_summary(regime_frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
        'regime_hint',
        'series_count',
        'mean_series_length',
        'median_dominant_period',
        'mean_acf_decay_rate',
        'mean_spectral_concentration',
        'mean_spectral_flatness',
        'mean_local_linearity_score',
        'mean_switching_score',
    )
    if regime_frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for regime_hint, frame in regime_frame.groupby('regime_hint', dropna=False, sort=False):
        rows.append(
            {
                'regime_hint': regime_hint,
                'series_count': int(len(frame)),
                'mean_series_length': float(frame['series_length'].astype(float).mean()),
                'median_dominant_period': float(frame['dominant_period'].astype(float).median()) if frame['dominant_period'].notna().any() else np.nan,
                'mean_acf_decay_rate': float(frame['acf_decay_rate'].astype(float).mean()),
                'mean_spectral_concentration': float(frame['spectral_concentration'].astype(float).mean()),
                'mean_spectral_flatness': float(frame['spectral_flatness'].astype(float).mean()),
                'mean_local_linearity_score': float(frame['local_linearity_score'].astype(float).mean()),
                'mean_switching_score': float(frame['switching_score'].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values('series_count', ascending=False).reset_index(drop=True)


def build_regime_improvement_summary(
        gain_frame: pd.DataFrame,
        regime_frame: pd.DataFrame,
) -> pd.DataFrame:
    columns = (
        'metric_name',
        'regime_hint',
        'case_count',
        'improved_case_count',
        'not_improved_case_count',
        'improvement_rate_pct',
        'no_improvement_rate_pct',
        'mean_relative_gain_pct',
        'median_relative_gain_pct',
        'mean_series_length',
        'mean_dominant_period',
        'mean_acf_decay_rate',
        'mean_spectral_concentration',
        'mean_spectral_flatness',
        'mean_local_linearity_score',
        'mean_switching_score',
    )
    if gain_frame.empty or regime_frame.empty:
        return pd.DataFrame(columns=columns)
    join_columns = ['run_id', 'dataset_name', 'subset', 'series_id', 'model_name']
    merged = gain_frame.merge(regime_frame, on=join_columns, how='left')
    rows: list[dict[str, Any]] = []
    for (metric_name, regime_hint), frame in merged.groupby(['metric_name', 'regime_hint'], dropna=False, sort=False):
        rows.append(
            {
                'metric_name': metric_name,
                'regime_hint': regime_hint,
                'case_count': int(len(frame)),
                'improved_case_count': int(frame['improved'].sum()),
                'not_improved_case_count': int((~frame['improved']).sum()),
                'improvement_rate_pct': float(100.0 * frame['improved'].mean()),
                'no_improvement_rate_pct': float(100.0 * (1.0 - frame['improved'].mean())),
                'mean_relative_gain_pct': float(frame['relative_gain_pct'].astype(float).mean()),
                'median_relative_gain_pct': float(frame['relative_gain_pct'].astype(float).median()),
                'mean_series_length': float(frame['series_length'].astype(float).mean()),
                'mean_dominant_period': float(frame['dominant_period'].astype(float).mean()) if frame['dominant_period'].notna().any() else np.nan,
                'mean_acf_decay_rate': float(frame['acf_decay_rate'].astype(float).mean()),
                'mean_spectral_concentration': float(frame['spectral_concentration'].astype(float).mean()),
                'mean_spectral_flatness': float(frame['spectral_flatness'].astype(float).mean()),
                'mean_local_linearity_score': float(frame['local_linearity_score'].astype(float).mean()),
                'mean_switching_score': float(frame['switching_score'].astype(float).mean()),
            }
        )
    order = {name: index for index, name in enumerate(LOWER_IS_BETTER_METRICS)}
    return pd.DataFrame(rows).sort_values(
        ['metric_name', 'case_count'],
        key=lambda series: series.map(order.get) if series.name == 'metric_name' else series,
        ascending=[True, False],
    ).reset_index(drop=True)


@dataclass
class ForecastingProgressItemsVisualizer:
    items_dir: str | Path
    output_dir: str | Path
    model_name: str | None = None
    series_ids: tuple[str, ...] = ()
    max_series_plots: int | None = 3
    plot_formats: tuple[str, ...] = ('png', 'svg')

    def __post_init__(self):
        self.items_dir = Path(self.items_dir)
        self.output_dir = ensure_directory(self.output_dir)
        self._items_cache: tuple[ForecastingProgressItemPayload, ...] | None = None

    def load_items(self) -> tuple[ForecastingProgressItemPayload, ...]:
        if self._items_cache is None:
            self._items_cache = load_progress_item_payloads(
                self.items_dir,
                model_name=self.model_name,
                status='success',
            )
        return self._items_cache

    def _build_items_frame(self, items: tuple[ForecastingProgressItemPayload, ...]) -> pd.DataFrame:
        rows = []
        for item in items:
            rows.append(
                {
                    'run_id': item.run_id,
                    'dataset_name': item.dataset_name,
                    'subset': item.subset,
                    'series_id': item.series_id,
                    'model_name': item.model_name,
                    'status': item.status,
                    'history_length': len(item.series_record.get('train_values', ())),
                    'forecast_horizon': int(item.series_record.get('forecast_horizon', 0)),
                }
            )
        return pd.DataFrame(rows)

    def _select_series_items(self, items: tuple[ForecastingProgressItemPayload, ...]) -> tuple[ForecastingProgressItemPayload, ...]:
        if self.series_ids:
            requested = {str(series_id) for series_id in self.series_ids}
            return tuple(item for item in items if item.series_id in requested)
        if self.max_series_plots is None:
            return tuple(items)
        return tuple(items[:max(0, int(self.max_series_plots))])

    def _save_figure(self, figure, target_stem: Path) -> list[ArtifactRecord]:
        manifest: list[ArtifactRecord] = []
        for extension in self.plot_formats:
            path = target_stem.with_suffix(f'.{extension}')
            figure.savefig(path, dpi=200, bbox_inches='tight')
            manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        return manifest

    def _render_history_forecast_plot(self, item: ForecastingProgressItemPayload) -> list[ArtifactRecord]:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        history = _resolved_history(item)
        actual = _actual_forecast_target(item)
        baseline_forecast = _baseline_forecast_from_prediction_records(item)
        comparison = _safe_stage_tuning_comparison(item)
        tuned_forecast = _normalize_numeric_sequence(comparison.get('tuned_forecast'))
        baseline_metrics = dict(comparison.get('baseline_metrics', {}))
        tuned_metrics = dict(comparison.get('tuned_metrics', {}))

        forecast_index = np.arange(len(history), len(history) + len(actual))
        figure, (axis, zoom_axis) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            gridspec_kw={'height_ratios': [2.6, 1.4]},
        )
        history_index = np.arange(len(history))
        forecast_zone_start = len(history) - 0.5
        forecast_zone_end = len(history) + len(actual) - 0.5
        axis.plot(history_index, history, color='tab:blue', linewidth=1.5, label='История')
        axis.plot(forecast_index, actual, color='black', linewidth=2.0, label='Актуальные данные')
        if len(baseline_forecast) == len(actual):
            axis.plot(
                forecast_index,
                baseline_forecast,
                color='tab:red',
                linestyle='--',
                linewidth=1.4,
                alpha=0.65,
                label='Базовый прогноз',
            )
        selected_forecast = tuned_forecast if len(tuned_forecast) == len(actual) else baseline_forecast
        selected_label = 'Прогноз после тюнинга' if len(tuned_forecast) == len(actual) else 'Прогноз модели'
        if len(selected_forecast) == len(actual):
            axis.plot(
                forecast_index,
                selected_forecast,
                color='tab:green',
                linewidth=2.0,
                label=selected_label,
            )
        axis.axvspan(forecast_zone_start, forecast_zone_end, color='tab:orange', alpha=0.08, label='Зона прогноза')
        axis.axvline(len(history) - 1, color='gray', linestyle=':', linewidth=1.0)
        axis.set_title(f'{item.model_name} | {item.dataset_name} | {item.series_id}')
        axis.set_xlabel('Индекс времени')
        axis.set_ylabel('Значение ряда')
        axis.grid(alpha=0.2)
        axis.legend(loc='best')

        zoom_context = max(30, int(max(1, len(actual)) * 5))
        zoom_start = max(0, len(history) - zoom_context)
        zoom_end = len(history) + len(actual)
        zoom_index = np.arange(zoom_start, zoom_end)
        zoom_series = np.concatenate([history[zoom_start:], actual])
        zoom_axis.plot(history_index[zoom_start:], history[zoom_start:], color='tab:blue', linewidth=1.5, label='История')
        zoom_axis.plot(forecast_index, actual, color='black', linewidth=2.0, label='Актуальные данные')
        if len(baseline_forecast) == len(actual):
            zoom_axis.plot(
                forecast_index,
                baseline_forecast,
                color='tab:red',
                linestyle='--',
                linewidth=1.4,
                alpha=0.65,
                label='Базовый прогноз',
            )
        if len(selected_forecast) == len(actual):
            zoom_axis.plot(
                forecast_index,
                selected_forecast,
                color='tab:green',
                linewidth=2.0,
                label=selected_label,
            )
        zoom_axis.axvspan(forecast_zone_start, forecast_zone_end, color='tab:orange', alpha=0.12)
        zoom_axis.axvline(len(history) - 1, color='gray', linestyle=':', linewidth=1.0)
        zoom_axis.set_xlim(zoom_start, zoom_end - 1 + 0.5)
        zoom_axis.set_title('Увеличение зоны прогноза и ближайшей предыстории')
        zoom_axis.set_xlabel('Индекс времени')
        zoom_axis.set_ylabel('Значение')
        zoom_axis.grid(alpha=0.2)
        if len(zoom_series):
            y_min = float(np.nanmin(zoom_series))
            y_max = float(np.nanmax(zoom_series))
            margin = max(1e-6, 0.08 * (y_max - y_min if y_max > y_min else abs(y_max) + 1.0))
            zoom_axis.set_ylim(y_min - margin, y_max + margin)
        zoom_axis.legend(loc='best')

        title_lines = []
        if baseline_metrics:
            title_lines.append(
                'Baseline: ' + ', '.join(
                    f'{metric.upper()}={float(value):.3f}'
                    for metric, value in baseline_metrics.items()
                    if metric in ('rmse', 'mae', 'smape', 'mase', 'owa')
                )
            )
        if tuned_metrics:
            title_lines.append(
                'Tuned: ' + ', '.join(
                    f'{metric.upper()}={float(value):.3f}'
                    for metric, value in tuned_metrics.items()
                    if metric in ('rmse', 'mae', 'smape', 'mase', 'owa')
                )
            )
        if title_lines:
            axis.text(
                0.01,
                0.02,
                '\n'.join(title_lines),
                transform=axis.transAxes,
                fontsize=9,
                va='bottom',
                ha='left',
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'lightgray'},
            )

        figure.tight_layout()
        target_dir = ensure_directory(self.output_dir / 'series_history_forecast')
        manifest = self._save_figure(figure, target_dir / f'{_series_plot_slug(item)}__history_forecast')
        plt.close(figure)
        return manifest

    def _render_fold_diagnostics_plot(self, item: ForecastingProgressItemPayload) -> list[ArtifactRecord]:
        report = _safe_stage_tuning_report(item)
        baseline_evaluation = dict(report.get('baseline_evaluation', {}))
        tuned_evaluation = dict(report.get('best_evaluation', {}))
        baseline_folds = _extract_evaluation_fold_sequences(baseline_evaluation)
        tuned_folds = _extract_evaluation_fold_sequences(tuned_evaluation)
        if not baseline_folds:
            return []
        fold_frame = build_fold_comparison_frame(item)

        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.gridspec import GridSpec

        fold_count = len(baseline_folds)
        column_count = 2
        row_count = max(1, ceil(fold_count / column_count))
        figure = plt.figure(figsize=(14, 4 + 3.2 * row_count))
        grid = GridSpec(row_count + 1, column_count, figure=figure, height_ratios=[1.2] + [1.0] * row_count)

        timeline_axis = figure.add_subplot(grid[0, :])
        series = _resolved_history(item)
        timeline_axis.plot(np.arange(len(series)), series, color='lightgray', linewidth=1.2, label='История для CV')
        train_legend_added = False
        test_legend_added = False
        gain_values = fold_frame['relative_gain_pct'].dropna().astype(float).to_numpy() if not fold_frame.empty else np.asarray([])
        max_abs_gain = _resolve_symmetric_gain_span(gain_values)
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs_gain, vcenter=0.0, vmax=max_abs_gain)
        cmap = matplotlib.colormaps.get_cmap('RdYlGn')
        for fold in baseline_folds:
            train_start = int(fold.get('train_start', 0))
            train_end = int(fold.get('train_end', train_start))
            test_start = int(fold.get('test_start', train_end + 1))
            test_end = int(fold.get('test_end', test_start))
            timeline_axis.axvspan(
                train_start,
                train_end,
                color='tab:blue',
                  alpha=0.08,
                  label='Train fold' if not train_legend_added else None,
              )
            fold_gain = None
            if not fold_frame.empty:
                match = fold_frame[fold_frame['fold_index'] == int(fold.get('fold_index', 0))]
                if not match.empty:
                    fold_gain = match['relative_gain_pct'].iloc[0]
            test_color = cmap(norm(float(fold_gain))) if fold_gain is not None and pd.notna(fold_gain) else 'tab:orange'
            timeline_axis.axvspan(
                test_start,
                test_end,
                color=test_color,
                alpha=0.28,
                label='Validation fold (gain-colored)' if not test_legend_added else None,
            )
            train_legend_added = True
            test_legend_added = True
            timeline_axis.text((test_start + test_end) / 2.0, np.nanmax(series), f'F{int(fold.get("fold_index", 0))}',
                               ha='center', va='bottom', fontsize=8)
        timeline_axis.set_title(f'Временные фолды stage tuning | {item.series_id}')
        timeline_axis.set_xlabel('Индекс внутри train-серии')
        timeline_axis.set_ylabel('Значение')
        timeline_axis.grid(alpha=0.2)
        timeline_axis.legend(loc='best')
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        colorbar = figure.colorbar(sm, ax=timeline_axis, fraction=0.025, pad=0.02)
        colorbar.set_label('Relative gain per fold, %')

        for index, fold in enumerate(baseline_folds):
            axis = figure.add_subplot(grid[1 + index // column_count, index % column_count])
            tuned_fold = tuned_folds[index] if index < len(tuned_folds) else {}
            baseline_target = np.asarray(fold.get('target', ()), dtype=float)
            baseline_forecast = np.asarray(fold.get('forecast', ()), dtype=float)
            tuned_forecast = np.asarray(tuned_fold.get('forecast', ()), dtype=float)
            x_index = np.arange(int(fold.get('test_start', 0)), int(fold.get('test_start', 0)) + len(baseline_target))
            axis.plot(x_index, baseline_target, color='black', linewidth=1.8, label='Актуальное')
            axis.plot(x_index, baseline_forecast, color='tab:red', linestyle='--', linewidth=1.5, label='Baseline')
            if len(tuned_forecast) == len(baseline_target):
                axis.plot(x_index, tuned_forecast, color='tab:green', linewidth=1.8, label='Tuned')
            baseline_metric = fold.get('metric_value')
            tuned_metric = tuned_fold.get('metric_value')
            metric_name = str((baseline_evaluation.get('metric') or {}).get('metric_name', 'metric')).upper()
            if baseline_metric is not None and tuned_metric is not None:
                subtitle = f'{metric_name}: {float(baseline_metric):.3f} -> {float(tuned_metric):.3f}'
            elif baseline_metric is not None:
                subtitle = f'{metric_name}: {float(baseline_metric):.3f}'
            else:
                subtitle = metric_name
            axis.set_title(f'Fold {int(fold.get("fold_index", index + 1))} | {subtitle}')
            axis.set_xlabel('Индекс времени')
            axis.set_ylabel('Значение')
            axis.grid(alpha=0.2)
            if index == 0:
                axis.legend(loc='best')

        figure.tight_layout()
        target_dir = ensure_directory(self.output_dir / 'series_fold_diagnostics')
        stem = target_dir / f'{_series_plot_slug(item)}__fold_diagnostics'
        manifest = self._save_figure(figure, stem)
        plt.close(figure)
        manifest.extend(
            self._write_frame_bundle(
                fold_frame,
                target_dir / f'{_series_plot_slug(item)}__fold_metric_summary',
            )
        )
        return manifest

    def _write_frame_bundle(self, frame: pd.DataFrame, target_stem: Path) -> list[ArtifactRecord]:
        ensure_directory(target_stem.parent)
        manifest: list[ArtifactRecord] = []
        csv_path = target_stem.with_suffix('.csv')
        frame.to_csv(csv_path, index=False)
        manifest.append(ArtifactRecord(kind='table', path=str(csv_path), format='csv'))
        md_path = target_stem.with_suffix('.md')
        md_path.write_text(dataframe_to_markdown(frame, index=False), encoding='utf-8')
        manifest.append(ArtifactRecord(kind='summary', path=str(md_path), format='md'))
        return manifest

    def _render_series_relative_gain_plot(
            self,
            item: ForecastingProgressItemPayload,
            gain_frame: pd.DataFrame,
    ) -> list[ArtifactRecord]:
        series_frame = gain_frame[
            (gain_frame['dataset_name'] == item.dataset_name)
            & (gain_frame['series_id'] == item.series_id)
            & (gain_frame['model_name'] == item.model_name)
        ].copy()
        if series_frame.empty:
            return []

        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ordered_metrics = [metric for metric in LOWER_IS_BETTER_METRICS if metric in set(series_frame['metric_name'])]
        if not ordered_metrics:
            return []
        plot_frame = (
            series_frame.set_index('metric_name')
            .reindex(ordered_metrics)
            .reset_index()
        )
        labels = [metric.upper() for metric in ordered_metrics]
        values = plot_frame['relative_gain_pct'].astype(float).to_numpy()
        colors = ['tab:green' if value >= 0 else 'tab:red' for value in values]
        figure, axis = plt.subplots(figsize=(11, 5))
        bars = axis.bar(labels, values, color=colors, alpha=0.85)
        axis.axhline(0.0, color='gray', linestyle=':', linewidth=1.0)
        axis.set_title(f'Относительное изменение метрик | {item.dataset_name} | {item.series_id}')
        axis.set_xlabel('Метрика')
        axis.set_ylabel('Прирост качества, % (положительное значение = улучшение)')
        axis.grid(alpha=0.2, axis='y')
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + (1 if value >= 0 else -1) * max(0.5, abs(value) * 0.02),
                f'{value:.1f}%',
                ha='center',
                va='bottom' if value >= 0 else 'top',
                fontsize=9,
            )

        baseline_params = _compact_param_dict(
            dict(((_safe_stage_tuning_report(item).get('baseline_evaluation') or {}).get('parameters', {})))
        )
        tuned_params = _compact_param_dict(
            dict(((_safe_stage_tuning_report(item).get('best_evaluation') or {}).get('parameters', {})))
        )
        axis.text(
            0.01,
            0.02,
            'Baseline params: ' + '; '.join(f'{key}={_stringify_param_value(value)}' for key, value in baseline_params.items())
            + '\n'
            + 'Tuned params: ' + '; '.join(f'{key}={_stringify_param_value(value)}' for key, value in tuned_params.items()),
            transform=axis.transAxes,
            fontsize=9,
            va='bottom',
            ha='left',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'lightgray'},
        )

        target_dir = ensure_directory(self.output_dir / 'series_relative_gain')
        stem = target_dir / f'{_series_plot_slug(item)}__relative_metric_gain'
        manifest = self._save_figure(figure, stem)
        plt.close(figure)
        manifest.extend(
            self._write_frame_bundle(
                plot_frame[[
                    'metric_name',
                    'baseline_value',
                    'tuned_value',
                    'absolute_gain',
                    'relative_gain_pct',
                    'improved',
                ]],
                stem,
            )
        )
        return manifest

    def _render_metric_pair_scatter(self, gain_frame: pd.DataFrame) -> list[ArtifactRecord]:
        if gain_frame.empty:
            return []

        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ordered_metrics = [metric for metric in LOWER_IS_BETTER_METRICS if metric in set(gain_frame['metric_name'])]
        if not ordered_metrics:
            return []

        column_count = 2
        row_count = max(1, ceil(len(ordered_metrics) / column_count))
        figure, axes = plt.subplots(row_count, column_count, figsize=(14, 4.5 * row_count))
        flat_axes = np.atleast_1d(axes).reshape(-1)

        for axis, metric_name in zip(flat_axes, ordered_metrics):
            metric_frame = gain_frame[gain_frame['metric_name'] == metric_name].copy()
            baseline = metric_frame['baseline_value'].astype(float).to_numpy()
            tuned = metric_frame['tuned_value'].astype(float).to_numpy()
            improved_mask = metric_frame['improved'].astype(bool).to_numpy()
            axis.scatter(
                baseline[~improved_mask],
                tuned[~improved_mask],
                color='tab:red',
                alpha=0.5,
                s=26,
                label='Без улучшения',
            )
            axis.scatter(
                baseline[improved_mask],
                tuned[improved_mask],
                color='tab:green',
                alpha=0.6,
                s=26,
                label='Есть улучшение',
            )
            axis.scatter(
                baseline,
                tuned,
                facecolors='none',
                edgecolors='black',
                alpha=0.5,
                s=70,
                linewidths=0.8,
            )
            diagonal_min = float(np.nanmin(np.concatenate([baseline, tuned])))
            diagonal_max = float(np.nanmax(np.concatenate([baseline, tuned])))
            margin = max(1e-6, 0.05 * (diagonal_max - diagonal_min if diagonal_max > diagonal_min else abs(diagonal_max) + 1.0))
            axis.plot(
                [diagonal_min - margin, diagonal_max + margin],
                [diagonal_min - margin, diagonal_max + margin],
                color='gray',
                linestyle=':',
                linewidth=1.0,
            )
            axis.set_xlim(diagonal_min - margin, diagonal_max + margin)
            axis.set_ylim(diagonal_min - margin, diagonal_max + margin)
            improvement_rate = 100.0 * float(metric_frame['improved'].mean())
            axis.set_title(f'{metric_name.upper()} | improvement rate = {improvement_rate:.1f}%')
            axis.set_xlabel('Baseline')
            axis.set_ylabel('Tuned')
            axis.grid(alpha=0.2)
            axis.legend(loc='best')

        for axis in flat_axes[len(ordered_metrics):]:
            axis.axis('off')

        figure.suptitle('Baseline vs tuned: сравнение метрик по сериям', fontsize=16)
        figure.tight_layout()
        target_dir = ensure_directory(self.output_dir / 'aggregate')
        manifest = self._save_figure(figure, target_dir / 'baseline_vs_tuned_metric_scatter')
        plt.close(figure)
        return manifest

    def _build_summary_context(
            self,
            items_frame: pd.DataFrame,
            gain_summary: pd.DataFrame,
            improvement_case_summary: pd.DataFrame,
            regime_diagnostics_summary: pd.DataFrame,
            regime_improvement_summary: pd.DataFrame,
            artifact_manifest: list[ArtifactRecord],
    ) -> dict[str, Any]:
        manifest_rows: list[dict[str, Any]] = []
        for record in artifact_manifest:
            path = Path(record.path)
            try:
                relative_path = path.relative_to(self.output_dir)
            except ValueError:
                relative_path = path
            manifest_rows.append(
                {
                    'kind': record.kind,
                    'format': record.format,
                    'path': str(relative_path).replace('\\', '/'),
                }
            )
        return {
            'item_count': int(len(items_frame)),
            'series_count': int(items_frame['series_id'].nunique()) if not items_frame.empty else 0,
            'dataset_count': int(items_frame['dataset_name'].nunique()) if not items_frame.empty else 0,
            'metrics': gain_summary.to_dict(orient='records'),
            'improvement_cases': improvement_case_summary.to_dict(orient='records'),
            'regime_diagnostics_summary': regime_diagnostics_summary.to_dict(orient='records'),
            'regime_improvement_summary': regime_improvement_summary.to_dict(orient='records'),
            'artifacts': manifest_rows,
        }

    def _write_summary_page(
            self,
            items_frame: pd.DataFrame,
            gain_summary: pd.DataFrame,
            improvement_case_summary: pd.DataFrame,
            regime_diagnostics_summary: pd.DataFrame,
            regime_improvement_summary: pd.DataFrame,
            artifact_manifest: list[ArtifactRecord],
    ) -> list[ArtifactRecord]:
        if items_frame.empty:
            return []

        summary_context = self._build_summary_context(
            items_frame=items_frame,
            gain_summary=gain_summary,
            improvement_case_summary=improvement_case_summary,
            regime_diagnostics_summary=regime_diagnostics_summary,
            regime_improvement_summary=regime_improvement_summary,
            artifact_manifest=artifact_manifest,
        )
        aggregate_dir = ensure_directory(self.output_dir / 'aggregate')
        markdown_path = aggregate_dir / 'visualization_summary.md'
        html_path = aggregate_dir / 'visualization_summary.html'

        markdown_sections = [
            '# Отчёт по визуализации benchmark-результатов',
            '',
            '## Общая сводка',
            '',
            f"- Items: {summary_context['item_count']}",
            f"- Уникальных рядов: {summary_context['series_count']}",
            f"- Датасетов: {summary_context['dataset_count']}",
            '',
            '## Сводка по метрикам',
            '',
            dataframe_to_markdown(gain_summary, index=False) if not gain_summary.empty else 'Нет данных',
            '',
            '## Процент случаев, где тюнинг помог',
            '',
            dataframe_to_markdown(improvement_case_summary, index=False) if not improvement_case_summary.empty else 'Нет данных',
            '',
            '## Агрегаты regime diagnostics',
            '',
            dataframe_to_markdown(regime_diagnostics_summary, index=False) if not regime_diagnostics_summary.empty else 'Нет данных',
            '',
            '## Связка regime diagnostics и эффекта тюнинга',
            '',
            dataframe_to_markdown(regime_improvement_summary, index=False) if not regime_improvement_summary.empty else 'Нет данных',
            '',
            '## Артефакты',
            '',
        ]
        for artifact in summary_context['artifacts']:
            markdown_sections.append(f"- `{artifact['kind']}` [{artifact['path']}]({artifact['path']})")
        markdown_path.write_text('\n'.join(markdown_sections), encoding='utf-8')

        def _frame_to_html(frame: pd.DataFrame) -> str:
            if frame.empty:
                return '<p>Нет данных</p>'
            return frame.to_html(index=False, border=0, classes='summary-table')

        artifact_links = '\n'.join(
            f"<li><code>{escape(artifact['kind'])}</code> <a href=\"{escape(artifact['path'])}\">{escape(artifact['path'])}</a></li>"
            for artifact in summary_context['artifacts']
        )
        history_plots = [
            artifact['path']
            for artifact in summary_context['artifacts']
            if artifact['kind'] == 'plot' and artifact['path'].startswith('series_history_forecast/') and artifact['path'].endswith('.png')
        ]
        history_gallery = '\n'.join(
            f"""
            <div class="history-card">
              <div class="history-label">{escape(Path(path).name)}</div>
              <a href="{escape(path)}"><img src="{escape(path)}" alt="{escape(path)}"></a>
            </div>
            """
            for path in history_plots
        ) or '<p>Нет history-графиков</p>'
        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Benchmark visualization summary</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; line-height: 1.45; }}
    h1, h2 {{ color: #1f2937; }}
    .summary-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    .summary-table th, .summary-table td {{ border: 1px solid #d1d5db; padding: 6px 10px; text-align: left; }}
    .summary-table th {{ background: #f3f4f6; }}
    .stats {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 24px; }}
    .stat-card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px 16px; min-width: 160px; background: #fafafa; }}
    .history-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    .history-card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 10px; background: #fafafa; }}
    .history-card img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
    .history-label {{ margin-bottom: 8px; font-size: 13px; color: #374151; }}
  </style>
</head>
<body>
  <h1>Отчёт по визуализации benchmark-результатов</h1>
  <div class="stats">
    <div class="stat-card"><strong>Items</strong><br>{summary_context['item_count']}</div>
    <div class="stat-card"><strong>Ряды</strong><br>{summary_context['series_count']}</div>
    <div class="stat-card"><strong>Датасеты</strong><br>{summary_context['dataset_count']}</div>
  </div>
  <h2>Сводка по метрикам</h2>
  {_frame_to_html(gain_summary)}
  <h2>Процент случаев, где тюнинг помог</h2>
  {_frame_to_html(improvement_case_summary)}
  <h2>Агрегаты regime diagnostics</h2>
  {_frame_to_html(regime_diagnostics_summary)}
  <h2>Связка regime diagnostics и эффекта тюнинга</h2>
  {_frame_to_html(regime_improvement_summary)}
  <h2>History-графики по сериям</h2>
  <div class="history-gallery">
    {history_gallery}
  </div>
  <h2>Артефакты</h2>
  <ul>
    {artifact_links}
  </ul>
</body>
</html>
"""
        html_path.write_text(html_content, encoding='utf-8')
        return [
            ArtifactRecord(kind='summary', path=str(markdown_path), format='md'),
            ArtifactRecord(kind='summary', path=str(html_path), format='html'),
        ]

    def render(self) -> ForecastingProgressVisualizationResult:
        items = self.load_items()
        items_frame = self._build_items_frame(items)
        gain_frame = build_relative_gain_frame(items)
        baseline_param_summary = _parameter_mode_summary(
            [row for row in gain_frame['baseline_params'].tolist()] if not gain_frame.empty else []
        )
        tuned_param_summary = _parameter_mode_summary(
            [row for row in gain_frame['tuned_params'].tolist()] if not gain_frame.empty else []
        )
        gain_summary = build_relative_gain_summary(
            gain_frame,
            baseline_param_summary=baseline_param_summary,
            tuned_param_summary=tuned_param_summary,
        )
        improvement_case_summary = build_improvement_case_summary(gain_frame)
        regime_diagnostics_frame = build_regime_diagnostics_frame(items)
        regime_diagnostics_summary = build_regime_diagnostics_summary(regime_diagnostics_frame)
        regime_improvement_summary = build_regime_improvement_summary(gain_frame, regime_diagnostics_frame)

        artifact_manifest: list[ArtifactRecord] = []
        if not items_frame.empty:
            artifact_manifest.extend(self._write_frame_bundle(items_frame, self.output_dir / 'aggregate' / 'items_overview'))
        if not gain_frame.empty:
            artifact_manifest.extend(self._write_frame_bundle(gain_frame.drop(columns=['baseline_params', 'tuned_params']),
                                                             self.output_dir / 'aggregate' / 'relative_metric_gains'))
        if not gain_summary.empty:
            artifact_manifest.extend(self._write_frame_bundle(gain_summary,
                                                             self.output_dir / 'aggregate' / 'relative_metric_gain_summary'))
            artifact_manifest.extend(self._render_metric_pair_scatter(gain_frame))
        if not improvement_case_summary.empty:
            artifact_manifest.extend(
                self._write_frame_bundle(
                    improvement_case_summary,
                    self.output_dir / 'aggregate' / 'improvement_case_summary',
                )
            )
        if not regime_diagnostics_frame.empty:
            artifact_manifest.extend(
                self._write_frame_bundle(
                    regime_diagnostics_frame,
                    self.output_dir / 'aggregate' / 'regime_diagnostics',
                )
            )
        if not regime_diagnostics_summary.empty:
            artifact_manifest.extend(
                self._write_frame_bundle(
                    regime_diagnostics_summary,
                    self.output_dir / 'aggregate' / 'regime_diagnostics_summary',
                )
            )
        if not regime_improvement_summary.empty:
            artifact_manifest.extend(
                self._write_frame_bundle(
                    regime_improvement_summary,
                    self.output_dir / 'aggregate' / 'regime_improvement_summary',
                )
            )

        selected_items = self._select_series_items(items)
        for item in selected_items:
            artifact_manifest.extend(self._render_history_forecast_plot(item))
            artifact_manifest.extend(self._render_fold_diagnostics_plot(item))
            artifact_manifest.extend(self._render_series_relative_gain_plot(item, gain_frame))
        artifact_manifest.extend(
            self._write_summary_page(
                items_frame=items_frame,
                gain_summary=gain_summary,
                improvement_case_summary=improvement_case_summary,
                regime_diagnostics_summary=regime_diagnostics_summary,
                regime_improvement_summary=regime_improvement_summary,
                artifact_manifest=artifact_manifest,
            )
        )

        return ForecastingProgressVisualizationResult(
            items_frame=items_frame,
            relative_gain_frame=gain_frame,
            relative_gain_summary=gain_summary,
            improvement_case_summary=improvement_case_summary,
            regime_diagnostics_frame=regime_diagnostics_frame,
            regime_diagnostics_summary=regime_diagnostics_summary,
            regime_improvement_summary=regime_improvement_summary,
            artifact_manifest=tuple(artifact_manifest),
        )


def visualize_forecasting_progress_items(
        items_dir: str | Path,
        *,
        output_dir: str | Path,
        model_name: str | None = None,
        series_ids: tuple[str, ...] = (),
        max_series_plots: int | None = 3,
        plot_formats: tuple[str, ...] = ('png', 'svg'),
) -> ForecastingProgressVisualizationResult:
    return ForecastingProgressItemsVisualizer(
        items_dir=items_dir,
        output_dir=output_dir,
        model_name=model_name,
        series_ids=series_ids,
        max_series_plots=max_series_plots,
        plot_formats=plot_formats,
    ).render()


__all__ = [
    'ForecastingProgressItemPayload',
    'ForecastingProgressItemsVisualizer',
    'ForecastingProgressVisualizationResult',
    'build_fold_comparison_frame',
    'build_improvement_case_summary',
    'build_regime_diagnostics_frame',
    'build_regime_diagnostics_summary',
    'build_regime_improvement_summary',
    'build_relative_gain_frame',
    'build_relative_gain_summary',
    'load_progress_item_payloads',
    'visualize_forecasting_progress_items',
]
