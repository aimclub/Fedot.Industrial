from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmark.industrial.visualization.forecasting import (
    build_fold_comparison_frame,
    build_relative_gain_frame,
    load_progress_item_payloads,
    visualize_forecasting_progress_items,
)


def _write_item(
        items_dir: Path,
        *,
        run_id: str,
        series_id: str,
        history_shift: float,
        baseline_metrics: dict[str, float],
        tuned_metrics: dict[str, float],
        baseline_params: dict,
        tuned_params: dict,
) -> None:
    history = [history_shift + float(index) for index in range(24)]
    horizon = 4
    test_values = [history_shift + 24.0, history_shift + 25.0, history_shift + 26.0, history_shift + 27.0]
    baseline_prediction = [value - 1.5 for value in test_values]
    tuned_prediction = [value - 0.5 for value in test_values]
    fold_target = test_values[:2] + test_values[2:]
    baseline_fold_forecast = [value - 1.0 for value in fold_target]
    tuned_fold_forecast = [value - 0.4 for value in fold_target]

    payload = {
        'run_id': run_id,
        'series_record': {
            'benchmark': 'm4',
            'dataset_name': 'm4_daily_full',
            'subset': 'Daily',
            'series_id': series_id,
            'frequency': 'Daily',
            'forecast_horizon': horizon,
            'seasonal_period': 1,
            'train_values': history,
            'test_values': test_values,
        },
        'run_record': {
            'run_id': run_id,
            'benchmark': 'm4',
            'dataset_name': 'm4_daily_full',
            'subset': 'Daily',
            'series_id': series_id,
            'model_name': 'lagged_forecaster',
            'status': 'success',
            'tags': ['baseline', 'forecasting'],
            'message': 'ready',
            'metrics_summary': baseline_metrics,
            'metadata': {
                'stage_tuning_report': {
                    'model_name': 'lagged_forecaster',
                    'canonical_model_name': 'lagged_forecaster',
                    'family': 'lagged_linear',
                    'sequential_result': {
                        'best_parameters': tuned_params,
                        'stage_history': [
                            {'stage': 'trajectory_transform'},
                            {'stage': 'forecast_head'},
                        ],
                    },
                    'baseline_evaluation': {
                        'parameters': baseline_params,
                        'forecast': baseline_fold_forecast,
                        'target': fold_target,
                        'metric': {
                            'metric_name': 'rmse',
                            'metric_value': baseline_metrics['rmse'],
                            'per_horizon_metrics': [baseline_metrics['rmse']] * horizon,
                            'per_fold_metric_values': [baseline_metrics['rmse'] + 0.2, baseline_metrics['rmse'] - 0.2],
                        },
                        'split_metadata': {
                            'fold_count': 2,
                            'folds': [
                                {
                                    'fold_index': 1,
                                    'train_start': 0,
                                    'train_end': 11,
                                    'test_start': 12,
                                    'test_end': 13,
                                    'train_length': 12,
                                    'test_length': 2,
                                    'split_kind': 'time_series_split',
                                    'validation_horizon': 2,
                                    'gap': 0,
                                    'metric': baseline_metrics['rmse'] + 0.2,
                                },
                                {
                                    'fold_index': 2,
                                    'train_start': 0,
                                    'train_end': 13,
                                    'test_start': 14,
                                    'test_end': 15,
                                    'train_length': 14,
                                    'test_length': 2,
                                    'split_kind': 'time_series_split',
                                    'validation_horizon': 2,
                                    'gap': 0,
                                    'metric': baseline_metrics['rmse'] - 0.2,
                                },
                            ],
                        },
                    },
                    'best_evaluation': {
                        'parameters': tuned_params,
                        'forecast': tuned_fold_forecast,
                        'target': fold_target,
                        'metric': {
                            'metric_name': 'rmse',
                            'metric_value': tuned_metrics['rmse'],
                            'per_horizon_metrics': [tuned_metrics['rmse']] * horizon,
                            'per_fold_metric_values': [tuned_metrics['rmse'] + 0.1, tuned_metrics['rmse'] - 0.1],
                        },
                        'split_metadata': {
                            'fold_count': 2,
                            'folds': [
                                {
                                    'fold_index': 1,
                                    'train_start': 0,
                                    'train_end': 11,
                                    'test_start': 12,
                                    'test_end': 13,
                                    'train_length': 12,
                                    'test_length': 2,
                                    'split_kind': 'time_series_split',
                                    'validation_horizon': 2,
                                    'gap': 0,
                                    'metric': tuned_metrics['rmse'] + 0.1,
                                },
                                {
                                    'fold_index': 2,
                                    'train_start': 0,
                                    'train_end': 13,
                                    'test_start': 14,
                                    'test_end': 15,
                                    'train_length': 14,
                                    'test_length': 2,
                                    'split_kind': 'time_series_split',
                                    'validation_horizon': 2,
                                    'gap': 0,
                                    'metric': tuned_metrics['rmse'] - 0.1,
                                },
                            ],
                        },
                    },
                },
                'stage_tuning_comparison': {
                    'best_parameters': tuned_params,
                    'baseline_metrics': baseline_metrics,
                    'tuned_metrics': tuned_metrics,
                    'improved_metrics': {key: tuned_metrics[key] < baseline_metrics[key] for key in baseline_metrics},
                    'absolute_gain': {key: baseline_metrics[key] - tuned_metrics[key] for key in baseline_metrics},
                    'tuned_forecast': tuned_prediction,
                },
            },
        },
        'metric_records': [],
        'prediction_records': [
            {
                'run_id': run_id,
                'benchmark': 'm4',
                'dataset_name': 'm4_daily_full',
                'subset': 'Daily',
                'series_id': series_id,
                'model_name': 'lagged_forecaster',
                'horizon_index': index + 1,
                'y_true': y_true,
                'y_pred': y_pred,
                'status': 'success',
            }
            for index, (y_true, y_pred) in enumerate(zip(test_values, baseline_prediction))
        ],
    }
    path = items_dir / f'm4_daily_full__daily__{series_id.lower()}__lagged_forecaster.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def test_build_relative_gain_frame_extracts_metric_improvement(tmp_path: Path) -> None:
    items_dir = tmp_path / 'items'
    items_dir.mkdir(parents=True)
    _write_item(
        items_dir,
        run_id='run_a',
        series_id='D1',
        history_shift=0.0,
        baseline_metrics={'rmse': 10.0, 'mae': 8.0, 'smape': 5.0, 'mase': 4.0, 'owa': 2.0},
        tuned_metrics={'rmse': 8.0, 'mae': 6.0, 'smape': 4.0, 'mase': 3.0, 'owa': 1.5},
        baseline_params={'window_size': 10, 'stride': 1, 'alpha': 1.0},
        tuned_params={'window_size': 16, 'stride': 3, 'alpha': 1.0},
    )

    items = load_progress_item_payloads(items_dir, model_name='lagged_forecaster')
    frame = build_relative_gain_frame(items)

    assert not frame.empty
    rmse_row = frame[frame['metric_name'] == 'rmse'].iloc[0]
    assert rmse_row['series_id'] == 'D1'
    assert rmse_row['absolute_gain'] == 2.0
    assert rmse_row['relative_gain_pct'] == 20.0


def test_build_fold_comparison_frame_extracts_per_fold_deltas(tmp_path: Path) -> None:
    items_dir = tmp_path / 'items'
    items_dir.mkdir(parents=True)
    _write_item(
        items_dir,
        run_id='run_a',
        series_id='D1',
        history_shift=0.0,
        baseline_metrics={'rmse': 10.0, 'mae': 8.0, 'smape': 5.0, 'mase': 4.0, 'owa': 2.0},
        tuned_metrics={'rmse': 8.0, 'mae': 6.0, 'smape': 4.0, 'mase': 3.0, 'owa': 1.5},
        baseline_params={'window_size': 10, 'stride': 1, 'alpha': 1.0},
        tuned_params={'window_size': 16, 'stride': 3, 'alpha': 1.0},
    )

    item = load_progress_item_payloads(items_dir, model_name='lagged_forecaster')[0]
    fold_frame = build_fold_comparison_frame(item)

    assert len(fold_frame) == 2
    assert set(fold_frame['fold_index']) == {1, 2}
    assert 'relative_gain_pct' in fold_frame.columns
    assert fold_frame['improved'].all()


def test_visualize_forecasting_progress_items_writes_expected_artifacts(tmp_path: Path) -> None:
    items_dir = tmp_path / 'items'
    output_dir = tmp_path / 'artifacts'
    items_dir.mkdir(parents=True)

    _write_item(
        items_dir,
        run_id='run_a',
        series_id='D1',
        history_shift=0.0,
        baseline_metrics={'rmse': 10.0, 'mae': 8.0, 'smape': 5.0, 'mase': 4.0, 'owa': 2.0},
        tuned_metrics={'rmse': 8.0, 'mae': 6.0, 'smape': 4.0, 'mase': 3.0, 'owa': 1.5},
        baseline_params={'window_size': 10, 'stride': 1, 'alpha': 1.0},
        tuned_params={'window_size': 16, 'stride': 3, 'alpha': 1.0},
    )
    _write_item(
        items_dir,
        run_id='run_b',
        series_id='D2',
        history_shift=10.0,
        baseline_metrics={'rmse': 12.0, 'mae': 9.0, 'smape': 6.0, 'mase': 4.5, 'owa': 2.5},
        tuned_metrics={'rmse': 9.0, 'mae': 7.0, 'smape': 4.5, 'mase': 3.5, 'owa': 1.7},
        baseline_params={'window_size': 10, 'stride': 1, 'alpha': 1.0},
        tuned_params={'window_size': 20, 'stride': 2, 'alpha': 0.5},
    )

    result = visualize_forecasting_progress_items(
        items_dir,
        output_dir=output_dir,
        model_name='lagged_forecaster',
        series_ids=('D1',),
        max_series_plots=1,
        plot_formats=('png',),
    )

    assert len(result.items_frame) == 2
    assert not result.relative_gain_summary.empty
    assert not result.improvement_case_summary.empty
    summary_csv = output_dir / 'aggregate' / 'relative_metric_gain_summary.csv'
    gains_csv = output_dir / 'aggregate' / 'relative_metric_gains.csv'
    improvement_csv = output_dir / 'aggregate' / 'improvement_case_summary.csv'
    regime_csv = output_dir / 'aggregate' / 'regime_diagnostics.csv'
    regime_summary_csv = output_dir / 'aggregate' / 'regime_diagnostics_summary.csv'
    regime_improvement_csv = output_dir / 'aggregate' / 'regime_improvement_summary.csv'
    scatter_plot = output_dir / 'aggregate' / 'baseline_vs_tuned_metric_scatter.png'
    summary_md = output_dir / 'aggregate' / 'visualization_summary.md'
    summary_html = output_dir / 'aggregate' / 'visualization_summary.html'
    history_plot = output_dir / 'series_history_forecast' / 'm4_daily_full__d1__lagged_forecaster__history_forecast.png'
    fold_plot = output_dir / 'series_fold_diagnostics' / 'm4_daily_full__d1__lagged_forecaster__fold_diagnostics.png'
    fold_csv = output_dir / 'series_fold_diagnostics' / 'm4_daily_full__d1__lagged_forecaster__fold_metric_summary.csv'
    series_gain_plot = output_dir / 'series_relative_gain' / 'm4_daily_full__d1__lagged_forecaster__relative_metric_gain.png'
    series_gain_csv = output_dir / 'series_relative_gain' / 'm4_daily_full__d1__lagged_forecaster__relative_metric_gain.csv'

    assert summary_csv.exists()
    assert gains_csv.exists()
    assert improvement_csv.exists()
    assert regime_csv.exists()
    assert regime_summary_csv.exists()
    assert regime_improvement_csv.exists()
    assert scatter_plot.exists()
    assert summary_md.exists()
    assert summary_html.exists()
    assert history_plot.exists()
    assert fold_plot.exists()
    assert fold_csv.exists()
    assert series_gain_plot.exists()
    assert series_gain_csv.exists()

    summary_frame = pd.read_csv(summary_csv)
    assert set(summary_frame['metric_name']) == {'owa', 'smape', 'mase', 'rmse', 'mae'}
    assert 'improvement_rate_pct' in summary_frame.columns
    assert 'no_improvement_rate_pct' in summary_frame.columns

    improvement_frame = pd.read_csv(improvement_csv)
    assert 'overall' in set(improvement_frame['scope'])

    regime_frame = pd.read_csv(regime_csv)
    assert 'regime_hint' in regime_frame.columns
    assert regime_frame['series_length'].notna().all()

    regime_summary_frame = pd.read_csv(regime_summary_csv)
    assert 'mean_spectral_concentration' in regime_summary_frame.columns

    regime_improvement_frame = pd.read_csv(regime_improvement_csv)
    assert 'improvement_rate_pct' in regime_improvement_frame.columns
    assert 'regime_hint' in regime_improvement_frame.columns

    fold_frame = pd.read_csv(fold_csv)
    assert set(fold_frame['fold_index']) == {1, 2}
    assert 'relative_gain_pct' in fold_frame.columns

    summary_text = summary_md.read_text(encoding='utf-8')
    assert 'Отчёт по визуализации benchmark-результатов' in summary_text
    assert 'baseline_vs_tuned_metric_scatter.png' in summary_text

    html_text = summary_html.read_text(encoding='utf-8')
    assert 'm4_daily_full__d1__lagged_forecaster__history_forecast.png' in html_text


def test_visualize_forecasting_progress_items_renders_all_series_when_max_series_plots_none(tmp_path: Path) -> None:
    items_dir = tmp_path / 'items'
    output_dir = tmp_path / 'artifacts'
    items_dir.mkdir(parents=True)

    _write_item(
        items_dir,
        run_id='run_a',
        series_id='D1',
        history_shift=0.0,
        baseline_metrics={'rmse': 10.0, 'mae': 8.0, 'smape': 5.0, 'mase': 4.0, 'owa': 2.0},
        tuned_metrics={'rmse': 8.0, 'mae': 6.0, 'smape': 4.0, 'mase': 3.0, 'owa': 1.5},
        baseline_params={'window_size': 10, 'stride': 1, 'alpha': 1.0},
        tuned_params={'window_size': 16, 'stride': 3, 'alpha': 1.0},
    )
    _write_item(
        items_dir,
        run_id='run_b',
        series_id='D2',
        history_shift=10.0,
        baseline_metrics={'rmse': 12.0, 'mae': 9.0, 'smape': 6.0, 'mase': 4.5, 'owa': 2.5},
        tuned_metrics={'rmse': 9.0, 'mae': 7.0, 'smape': 4.5, 'mase': 3.5, 'owa': 1.7},
        baseline_params={'window_size': 10, 'stride': 1, 'alpha': 1.0},
        tuned_params={'window_size': 20, 'stride': 2, 'alpha': 0.5},
    )

    visualize_forecasting_progress_items(
        items_dir,
        output_dir=output_dir,
        model_name='lagged_forecaster',
        max_series_plots=None,
        plot_formats=('png',),
    )

    history_plot_d1 = output_dir / 'series_history_forecast' / 'm4_daily_full__d1__lagged_forecaster__history_forecast.png'
    history_plot_d2 = output_dir / 'series_history_forecast' / 'm4_daily_full__d2__lagged_forecaster__history_forecast.png'
    assert history_plot_d1.exists()
    assert history_plot_d2.exists()

    summary_html = output_dir / 'aggregate' / 'visualization_summary.html'
    html_text = summary_html.read_text(encoding='utf-8')
    assert 'm4_daily_full__d1__lagged_forecaster__history_forecast.png' in html_text
    assert 'm4_daily_full__d2__lagged_forecaster__history_forecast.png' in html_text


def test_visualize_forecasting_progress_items_handles_zero_fold_gain(tmp_path: Path) -> None:
    items_dir = tmp_path / 'items'
    output_dir = tmp_path / 'artifacts'
    items_dir.mkdir(parents=True)
    baseline_metrics = {'rmse': 10.0, 'mae': 8.0, 'smape': 5.0, 'mase': 4.0, 'owa': 2.0}

    _write_item(
        items_dir,
        run_id='run_a',
        series_id='D1',
        history_shift=0.0,
        baseline_metrics=baseline_metrics,
        tuned_metrics=baseline_metrics,
        baseline_params={'window_size': 10, 'stride': 1, 'alpha': 1.0},
        tuned_params={'window_size': 10, 'stride': 1, 'alpha': 1.0},
    )
    item_path = items_dir / 'm4_daily_full__daily__d1__lagged_forecaster.json'
    payload = json.loads(item_path.read_text(encoding='utf-8'))
    baseline_evaluation = payload['run_record']['metadata']['stage_tuning_report']['baseline_evaluation']
    tuned_evaluation = payload['run_record']['metadata']['stage_tuning_report']['best_evaluation']
    tuned_evaluation['metric']['per_fold_metric_values'] = list(baseline_evaluation['metric']['per_fold_metric_values'])
    for tuned_fold, baseline_fold in zip(
            tuned_evaluation['split_metadata']['folds'],
            baseline_evaluation['split_metadata']['folds'],
    ):
        tuned_fold['metric'] = baseline_fold['metric']
    item_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    visualize_forecasting_progress_items(
        items_dir,
        output_dir=output_dir,
        model_name='lagged_forecaster',
        series_ids=('D1',),
        max_series_plots=1,
        plot_formats=('png',),
    )

    fold_plot = output_dir / 'series_fold_diagnostics' / 'm4_daily_full__d1__lagged_forecaster__fold_diagnostics.png'
    assert fold_plot.exists()
