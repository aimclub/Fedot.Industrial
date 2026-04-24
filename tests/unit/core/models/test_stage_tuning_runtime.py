import numpy as np
import pytest

pytest.importorskip('torch')

from fedot_ind.core.models.ts_forecasting.forecasting_runtime import ForecastingSplitKind, ForecastingSplitSpec
from fedot_ind.core.models.ts_forecasting.lagged_model.low_rank_lagged_ridge_forecaster import (
    LowRankLaggedRidgeForecasterImplementation,
)
from fedot_ind.core.models.ts_forecasting.progress_policy import ForecastingProgressPolicy
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning import ForecastingStageName
from fedot_ind.core.models.ts_forecasting.stage_tuning_runtime import (
    ForecastingSeriesEvaluator,
    ForecastingSeriesStageTuningRunner,
    _normalize_base_params,
    evaluate_forecasting_model_on_series,
    run_forecasting_stage_tuning_on_series,
)


def _trend_with_oscillation(length: int = 120) -> np.ndarray:
    time = np.arange(length, dtype=float)
    return 0.2 * time + 1.2 * np.sin(time / 5.0)


def test_evaluate_forecasting_model_on_series_returns_runtime_report_for_lagged_ridge():
    series = _trend_with_oscillation()

    evaluation = evaluate_forecasting_model_on_series(
        'lagged_ridge_forecaster',
        time_series=series,
        forecast_horizon=8,
        params={'window_size': 18, 'stride': 2, 'alpha': 0.5},
        metric_name='rmse',
    )

    assert evaluation.metric.metric_name == 'rmse'
    assert len(evaluation.forecast) == 80
    assert len(evaluation.target) == 80
    assert evaluation.split_metadata['split_kind'] == ForecastingSplitKind.TIME_SERIES_SPLIT.value
    assert evaluation.split_metadata['fold_count'] == 10
    assert 'trajectory_transform' in evaluation.diagnostics


def test_forecasting_series_evaluator_class_matches_wrapper_contract():
    series = _trend_with_oscillation()

    evaluation = ForecastingSeriesEvaluator(
        'lagged_ridge_forecaster',
        time_series=series,
        forecast_horizon=8,
        params={'window_size': 18, 'stride': 2, 'alpha': 0.5},
        metric_name='rmse',
    ).run()

    assert evaluation.metric.metric_name == 'rmse'
    assert evaluation.split_metadata['fold_count'] == 10
    assert evaluation.canonical_model_name == 'lagged_ridge_forecaster'


def test_evaluate_forecasting_model_on_series_supports_smape_for_low_rank_forecaster():
    series = _trend_with_oscillation(144)

    evaluation = evaluate_forecasting_model_on_series(
        'low_rank_lagged_ridge_forecaster',
        time_series=series,
        forecast_horizon=10,
        params={'window_size': 24, 'stride': 2, 'alpha': 1.0, 'explained_variance': 0.9},
        metric_name='smape',
    )

    assert evaluation.metric.metric_name == 'smape'
    assert np.isfinite(evaluation.metric.metric_value)
    assert evaluation.family == 'low_rank_linear'


def test_run_forecasting_stage_tuning_on_series_improves_or_matches_baseline():
    series = _trend_with_oscillation(132)

    result = run_forecasting_stage_tuning_on_series(
        'lagged_ridge_forecaster',
        time_series=series,
        forecast_horizon=8,
        base_params={'window_size': 10, 'stride': 1, 'alpha': 4.0},
        stage_updates={
            ForecastingStageName.TRAJECTORY.value: {'window_size': 20},
            ForecastingStageName.FORECAST_HEAD.value: {'alpha': 1.0},
        },
        metric_name='rmse',
        max_values_per_parameter=2,
        max_stage_candidates=4,
    )

    assert result.metadata['best_score'] <= result.metadata['baseline_score']
    assert result.sequential_result.stage_history[0]['stage'] == ForecastingStageName.TRAJECTORY.value
    assert result.best_evaluation.metric.metric_name == 'rmse'
    assert result.best_evaluation.split_metadata['fold_count'] == 10


def test_forecasting_series_stage_tuning_runner_class_matches_wrapper_contract():
    series = _trend_with_oscillation(132)

    result = ForecastingSeriesStageTuningRunner(
        'lagged_ridge_forecaster',
        time_series=series,
        forecast_horizon=8,
        base_params={'window_size': 10, 'stride': 1, 'alpha': 4.0},
        stage_updates={
            ForecastingStageName.TRAJECTORY.value: {'window_size': 20},
            ForecastingStageName.FORECAST_HEAD.value: {'alpha': 1.0},
        },
        metric_name='rmse',
        max_values_per_parameter=2,
        max_stage_candidates=4,
    ).run()

    assert result.metadata['best_score'] <= result.metadata['baseline_score']
    assert result.sequential_result.stage_history[0]['stage'] == ForecastingStageName.TRAJECTORY.value


def test_run_forecasting_stage_tuning_on_series_propagates_progress_policy():
    series = _trend_with_oscillation(132)

    result = run_forecasting_stage_tuning_on_series(
        'lagged_ridge_forecaster',
        time_series=series,
        forecast_horizon=8,
        base_params={'window_size': 10, 'stride': 1, 'alpha': 4.0},
        metric_name='rmse',
        max_values_per_parameter=2,
        max_stage_candidates=4,
        progress_policy=ForecastingProgressPolicy(enabled=True, stage_tuning_enabled=True),
    )

    assert result.metadata['progress_policy']['enabled'] is True
    assert result.sequential_result.metadata['progress_policy']['stage_tuning_enabled'] is True


def test_normalize_base_params_backfills_lagged_forecaster_defaults():
    resolved = _normalize_base_params(
        {'channel_model': 'ridge', 'window_size': 10},
        model_name='lagged_forecaster',
    )

    assert resolved['channel_model'] == 'ridge'
    assert resolved['window_size'] == 10
    assert resolved['stride'] == 1
    assert resolved['alpha'] == 1.0


def test_implementation_run_stage_tuning_on_series_uses_runtime_bridge():
    implementation = LowRankLaggedRidgeForecasterImplementation(
        params={'window_size': 18, 'stride': 2, 'alpha': 1.0, 'explained_variance': 0.9}
    )
    series = _trend_with_oscillation(128)

    result = implementation.run_stage_tuning_on_series(
        series,
        forecast_horizon=8,
        metric_name='mae',
        split_spec=ForecastingSplitSpec(
            kind=ForecastingSplitKind.ROLLING_WINDOW,
            validation_horizon=8,
            max_train_size=48,
            step_length=8,
            n_splits=3,
        ),
        stage_updates={
            ForecastingStageName.DECOMPOSITION_RANK.value: {'rank': 3},
            ForecastingStageName.FORECAST_HEAD.value: {'alpha': 0.5},
        },
        max_values_per_parameter=2,
        max_stage_candidates=4,
    )

    assert result['best_evaluation']['metric']['metric_name'] == 'mae'
    assert result['sequential_result']['stage_history'][-1]['stage'] == ForecastingStageName.FORECAST_HEAD.value
    assert result['best_evaluation']['split_metadata']['split_kind'] == ForecastingSplitKind.ROLLING_WINDOW.value


def test_okhs_runtime_bridge_supports_stubbed_backend(monkeypatch):
    class FakeOKHSRuntimeModel:
        def __init__(self, forecast_horizon, params=None):
            self.forecast_horizon = int(forecast_horizon)
            self.params = dict(params or {})

        def fit(self, time_series):
            self.series = np.asarray(time_series, dtype=float)
            return self

        def get_optimization_info(self):
            return {
                'resolved_window_size': int(self.params.get('window_size', 18)),
                'trajectory_representation_policy': 'projected',
                'trajectory_preprocessing': {'selected_rank': 4},
                'projection_metadata': {'decode_supported': True},
                'fdmd_prediction_diagnostics': {'anti_smoothing_diagnostics': {'collapse_detected': False}},
            }

        def predict(self, time_series=None, forecast_horizon=None):
            del time_series, forecast_horizon
            horizon = int(self.forecast_horizon)
            return np.linspace(1.0, 1.0 + horizon - 1, num=horizon)

        def get_diagnostics(self):
            return {
                'model_family': 'operator_model',
                'trajectory_transform': {
                    'resolved_window_size': int(self.params.get('window_size', 18)),
                },
                'decomposition': {'decode_supported': True},
                'rank_truncation': {'selected_rank': 4},
                'forecast_head': {},
            }

    monkeypatch.setattr(
        'fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_runtime.build_okhs_fdmd_forecaster',
        lambda forecast_horizon, params=None, series_length=None: FakeOKHSRuntimeModel(
            forecast_horizon=forecast_horizon,
            params={**dict(params or {}), 'series_length': series_length},
        ),
        raising=False,
    )

    evaluation = evaluate_forecasting_model_on_series(
        'okhs_fdmd_forecaster',
        time_series=_trend_with_oscillation(96),
        forecast_horizon=6,
        params={'window_size': 18, 'q': 0.7, 'n_modes': 4},
        metric_name='rmse',
    )

    assert evaluation.family == 'operator_model'
    assert evaluation.diagnostics['trajectory_transform']['resolved_window_size'] == 18


def test_neural_runtime_bridge_supports_stubbed_head_bridge(monkeypatch):
    class FakeNeuralBridge:
        def __init__(self, model_name, forecast_horizon, params=None):
            self.model_name = model_name
            self.forecast_horizon = forecast_horizon
            self.params = dict(params or {})

        def fit(self, time_series):
            self.series = np.asarray(time_series, dtype=float)
            return self

        def predict(self, time_series=None, forecast_horizon=None):
            del time_series, forecast_horizon
            return np.linspace(0.5, 0.5 + self.forecast_horizon - 1, num=self.forecast_horizon)

        def get_diagnostics(self):
            return {
                'model_family': 'neural_forecaster',
                'trajectory_transform': {'window_size': self.params.get('patch_len')},
                'decomposition': {},
                'rank_truncation': {},
                'forecast_head': {'head_type': self.model_name, 'epochs': self.params.get('epochs')},
            }

    monkeypatch.setattr(
        'fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_runtime.build_neural_forecast_head',
        lambda model_name, forecast_horizon, params=None: FakeNeuralBridge(
            model_name=model_name,
            forecast_horizon=forecast_horizon,
            params=params,
        ),
        raising=False,
    )

    evaluation = evaluate_forecasting_model_on_series(
        'patch_tst_model',
        time_series=_trend_with_oscillation(96),
        forecast_horizon=6,
        params={'patch_len': 12},
        metric_name='rmse',
    )

    assert evaluation.family == 'neural_forecaster'
    assert evaluation.diagnostics['forecast_head']['head_type'] == 'patch_tst_model'
