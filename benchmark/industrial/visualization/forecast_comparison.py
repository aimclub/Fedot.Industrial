from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from benchmark.industrial.core import ArtifactRecord, ensure_directory
from benchmark.industrial.evaluation.markdown import dataframe_to_markdown


def build_forecast_comparison_frame(
        *,
        history: Sequence[float],
        actual: Sequence[float],
        forecasts: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for index, value in enumerate(_as_float_array(history)):
        rows.append({'split': 'history', 'step': int(index), 'model_name': 'actual_history', 'value': float(value)})

    history_length = len(history)
    for index, value in enumerate(_as_float_array(actual)):
        rows.append({'split': 'actual', 'step': int(history_length + index), 'model_name': 'actual', 'value': float(value)})

    for model_name, forecast_values in forecasts.items():
        forecast = _as_float_array(forecast_values)
        if len(forecast) != len(actual):
            raise ValueError(
                f'Forecast length for {model_name!r} is {len(forecast)}, expected {len(actual)}.'
            )
        for index, value in enumerate(forecast):
            rows.append(
                {
                    'split': 'forecast',
                    'step': int(history_length + index),
                    'model_name': str(model_name),
                    'value': float(value),
                }
            )
    return pd.DataFrame(rows)


def build_forecast_metric_frame(
        *,
        actual: Sequence[float],
        forecasts: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    actual_array = _as_float_array(actual)
    rows = []
    for model_name, forecast_values in forecasts.items():
        forecast = _as_float_array(forecast_values)
        if len(forecast) != len(actual_array):
            raise ValueError(
                f'Forecast length for {model_name!r} is {len(forecast)}, expected {len(actual_array)}.'
            )
        error = forecast - actual_array
        rows.append(
            {
                'model_name': str(model_name),
                'mae': float(np.mean(np.abs(error))),
                'rmse': float(np.sqrt(np.mean(np.square(error)))),
                'smape': _smape(actual_array, forecast),
            }
        )
    return pd.DataFrame(rows).sort_values(['mae', 'model_name']).reset_index(drop=True)


def render_forecast_comparison_pack(
        *,
        history: Sequence[float],
        actual: Sequence[float],
        forecasts: Mapping[str, Sequence[float]],
        output_dir: str | Path,
        title: str = 'Forecast Comparison',
        series_id: str = 'preview_series',
        source_metadata: Mapping[str, Any] | None = None,
        plot_formats: Sequence[str] = ('png',),
) -> tuple[ArtifactRecord, ...]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    target_dir = ensure_directory(output_dir)
    tables_dir = ensure_directory(target_dir / 'tables')
    plots_dir = ensure_directory(target_dir / 'plots')
    manifest: list[ArtifactRecord] = []

    comparison = build_forecast_comparison_frame(history=history, actual=actual, forecasts=forecasts)
    metrics = build_forecast_metric_frame(actual=actual, forecasts=forecasts)
    manifest.extend(_write_table(comparison, tables_dir / 'forecast_comparison'))
    manifest.extend(_write_table(metrics, tables_dir / 'forecast_metrics'))
    metadata = {
        'title': title,
        'series_id': series_id,
        'history_length': len(history),
        'forecast_horizon': len(actual),
        'models': tuple(str(model_name) for model_name in forecasts),
        **dict(source_metadata or {}),
    }
    metadata_path = target_dir / 'source_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    manifest.append(ArtifactRecord(kind='metadata', path=str(metadata_path), format='json'))

    summary_path = target_dir / 'summary.md'
    summary_path.write_text(_summary_markdown(title, series_id, metrics), encoding='utf-8')
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))

    figure, axis = plt.subplots(figsize=(11, 5))
    history_array = _as_float_array(history)
    actual_array = _as_float_array(actual)
    history_index = np.arange(len(history_array))
    forecast_index = np.arange(len(history_array), len(history_array) + len(actual_array))
    axis.plot(history_index, history_array, color='0.35', linewidth=1.8, label='History')
    axis.plot(forecast_index, actual_array, color='black', linewidth=2.2, label='Actual')
    for model_name, forecast_values in forecasts.items():
        axis.plot(
            forecast_index,
            _as_float_array(forecast_values),
            linewidth=1.6,
            label=_short_model_name(str(model_name)),
        )
    axis.axvspan(forecast_index[0] - 0.5, forecast_index[-1] + 0.5, color='tab:orange', alpha=0.08)
    axis.set_title(title)
    axis.set_xlabel('Time step')
    axis.set_ylabel('Value')
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2, loc='best')
    manifest.extend(_save_figure(figure, plots_dir / 'multi_model_forecast', plot_formats))

    return tuple(manifest)


def build_forecast_comparison_from_aggregate_predictions(
        root: str | Path,
        *,
        series_id: str | None = None,
        dataset_name: str | None = None,
        model_names: Sequence[str] = (),
        history_length: int = 36,
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, tuple[float, ...]], dict[str, Any]]:
    root_path = Path(root)
    prediction_paths = sorted(root_path.rglob('aggregate/predictions.csv'))
    if not prediction_paths:
        raise ValueError(f'No aggregate/predictions.csv files found under {root_path}.')
    predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True)
    if predictions.empty:
        raise ValueError(f'Aggregate predictions under {root_path} are empty.')
    if dataset_name is not None and 'dataset_name' in predictions.columns:
        predictions = predictions[predictions['dataset_name'].astype(str) == str(dataset_name)].copy()
    if predictions.empty:
        raise ValueError(f'No aggregate predictions matched dataset_name={dataset_name!r}.')
    selected_series_id = str(series_id or sorted(predictions['series_id'].astype(str).unique())[0])
    series_predictions = predictions[predictions['series_id'].astype(str) == selected_series_id].copy()
    if series_predictions.empty:
        raise ValueError(f'No aggregate predictions matched series_id={series_id!r}.')
    if 'status' in series_predictions.columns:
        series_predictions = series_predictions[series_predictions['status'].astype(str) == 'success'].copy()
    if series_predictions.empty:
        raise ValueError(f'Aggregate predictions for series {selected_series_id!r} have no successful rows.')

    allowed = {str(model_name) for model_name in model_names}
    actual_frame = (
        series_predictions[['horizon_index', 'y_true']]
        .dropna()
        .drop_duplicates()
        .sort_values('horizon_index')
    )
    actual = tuple(float(value) for value in actual_frame['y_true'])
    forecasts: dict[str, tuple[float, ...]] = {}
    for model_name, group in series_predictions.groupby(series_predictions['model_name'].astype(str)):
        if allowed and str(model_name) not in allowed:
            continue
        ordered = group.sort_values('horizon_index')
        forecast = tuple(float(value) for value in ordered['y_pred'])
        if len(forecast) == len(actual):
            forecasts[str(model_name)] = forecast
    if not actual or not forecasts:
        raise ValueError(f'Aggregate predictions for series {selected_series_id!r} do not form a full comparison.')

    history = _load_history_from_series_artifacts(root_path, selected_series_id, history_length=history_length)

    metadata = {
        'source_root': str(root_path),
        'source_kind': 'aggregate_predictions',
        'dataset_name': str(series_predictions['dataset_name'].iloc[0]) if 'dataset_name' in series_predictions else '',
        'series_id': selected_series_id,
        'subset': str(series_predictions['subset'].iloc[0]) if 'subset' in series_predictions else '',
        'forecast_horizon': len(actual),
        'source_prediction_rows': int(len(series_predictions)),
    }
    return history, actual, forecasts, metadata


def build_forecast_comparison_from_progress_items(
        root: str | Path,
        *,
        series_id: str | None = None,
        dataset_name: str | None = None,
        model_names: Sequence[str] = (),
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, tuple[float, ...]], dict[str, Any]]:
    """Build a comparison payload from persisted forecasting progress item JSON files."""
    root_path = Path(root)
    item_paths = sorted(root_path.rglob('progress/items/*.json'))
    if not item_paths:
        raise ValueError(f'No forecasting progress item JSON files found under {root_path}.')

    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in item_paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        record = dict(payload.get('series_record') or {})
        current_series_id = str(record.get('series_id', ''))
        current_dataset = str(record.get('dataset_name', ''))
        if series_id is not None and current_series_id != str(series_id):
            continue
        if dataset_name is not None and current_dataset != str(dataset_name):
            continue
        grouped.setdefault(current_series_id, []).append(payload)

    if not grouped:
        raise ValueError(
            f'No forecasting progress items matched series_id={series_id!r}, dataset_name={dataset_name!r}.'
        )

    selected_series_id = str(series_id or sorted(grouped)[0])
    items = grouped[selected_series_id]
    first_record = dict(items[0].get('series_record') or {})
    history = tuple(float(value) for value in first_record.get('train_values') or ())
    actual = tuple(float(value) for value in first_record.get('test_values') or ())
    if not history or not actual:
        raise ValueError(f'Progress item for series {selected_series_id!r} does not contain train/test values.')

    allowed = {str(model_name) for model_name in model_names}
    forecasts: dict[str, tuple[float, ...]] = {}
    for item in items:
        run_record = dict(item.get('run_record') or {})
        model_name = str(run_record.get('model_name') or item.get('model_name') or '')
        if allowed and model_name not in allowed:
            continue
        prediction_records = list(item.get('prediction_records') or ())
        if not prediction_records:
            continue
        ordered = sorted(prediction_records, key=lambda record: int(record.get('horizon_index', 0)))
        forecast = tuple(float(record['y_pred']) for record in ordered if record.get('status') == 'success')
        if len(forecast) == len(actual):
            forecasts[model_name] = forecast

    if not forecasts:
        raise ValueError(
            f'Progress items for series {selected_series_id!r} do not contain successful prediction_records. '
            'Run the benchmark with prediction persistence enabled before rendering forecast comparison.'
        )

    metadata = {
        'source_root': str(root_path),
        'source_kind': 'forecasting_progress_items',
        'dataset_name': str(first_record.get('dataset_name', '')),
        'series_id': selected_series_id,
        'subset': str(first_record.get('subset', '')),
        'forecast_horizon': len(actual),
        'source_item_count': len(items),
    }
    return history, actual, forecasts, metadata


def file_md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_history_from_series_artifacts(
        root: Path,
        series_id: str,
        *,
        history_length: int,
) -> tuple[float, ...]:
    for path in sorted((root / 'series' / series_id).glob('*.csv')):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for column in ('history', 'train', 'train_values', 'y_history', 'value'):
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors='coerce').dropna().tail(history_length)
                if not values.empty:
                    return tuple(float(value) for value in values)
    return ()


def _short_model_name(model_name: str) -> str:
    replacements = {
        'KernelEnsembleForecaster_identity_shapelet': 'KLF shapelet',
        'KernelEnsembleForecaster_embedding_nystrom_okhs': 'KLF okhs',
        'LaggedRidgeForecaster': 'Lagged Ridge',
        'NaiveLastValue': 'Naive',
    }
    if model_name in replacements:
        return replacements[model_name]
    return model_name.replace('KernelEnsembleForecaster_', 'KLF ').replace('_', ' ')[:36]


def _write_table(frame: pd.DataFrame, path_without_suffix: Path) -> tuple[ArtifactRecord, ...]:
    csv_path = path_without_suffix.with_suffix('.csv')
    md_path = path_without_suffix.with_suffix('.md')
    frame.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(frame, index=False) if not frame.empty else 'No rows.', encoding='utf-8')
    return (
        ArtifactRecord(kind='table', path=str(csv_path), format='csv'),
        ArtifactRecord(kind='summary', path=str(md_path), format='md'),
    )


def _summary_markdown(title: str, series_id: str, metrics: pd.DataFrame) -> str:
    return '\n'.join(
        [
            f'# {title}',
            '',
            f'- Series: `{series_id}`',
            f'- Compared models: `{len(metrics)}`',
            '',
            '## Metrics',
            '',
            dataframe_to_markdown(metrics, index=False) if not metrics.empty else 'No metric rows.',
        ]
    )


def _save_figure(figure, path_without_suffix: Path, plot_formats: Sequence[str]) -> tuple[ArtifactRecord, ...]:
    import matplotlib.pyplot as plt

    manifest: list[ArtifactRecord] = []
    for extension in plot_formats:
        path = path_without_suffix.with_suffix(f'.{extension}')
        figure.savefig(path, dpi=200, bbox_inches='tight')
        manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
    plt.close(figure)
    return tuple(manifest)


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(tuple(values), dtype=float).reshape(-1)


def _smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(forecast)
    mask = denominator > 1e-12
    if not np.any(mask):
        return 0.0
    return float(200.0 * np.mean(np.abs(actual[mask] - forecast[mask]) / denominator[mask]))


__all__ = [
    'build_forecast_comparison_frame',
    'build_forecast_comparison_from_aggregate_predictions',
    'build_forecast_comparison_from_progress_items',
    'build_forecast_metric_frame',
    'file_md5',
    'render_forecast_comparison_pack',
]
