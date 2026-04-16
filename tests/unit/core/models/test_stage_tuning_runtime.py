import numpy as np
import pytest

pytest.importorskip('torch')

from fedot_ind.core.models.ts_forecasting.forecasting_runtime import ForecastingSplitSpec
from fedot_ind.core.models.ts_forecasting.low_rank_lagged_ridge_forecaster import (
    LowRankLaggedRidgeForecasterImplementation,
)
from fedot_ind.core.models.ts_forecasting.stage_tuning import ForecastingStageName
from fedot_ind.core.models.ts_forecasting.stage_tuning_runtime import (
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
    assert len(evaluation.forecast) == 8
    assert len(evaluation.target) == 8
    assert evaluation.split_metadata['train_length'] == len(series) - 8
    assert 'trajectory_transform' in evaluation.diagnostics


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


def test_implementation_run_stage_tuning_on_series_uses_runtime_bridge():
    implementation = LowRankLaggedRidgeForecasterImplementation(
        params={'window_size': 18, 'stride': 2, 'alpha': 1.0, 'explained_variance': 0.9}
    )
    series = _trend_with_oscillation(128)

    result = implementation.run_stage_tuning_on_series(
        series,
        forecast_horizon=8,
        metric_name='mae',
        split_spec=ForecastingSplitSpec(validation_horizon=8),
        stage_updates={
            ForecastingStageName.DECOMPOSITION_RANK.value: {'rank': 3},
            ForecastingStageName.FORECAST_HEAD.value: {'alpha': 0.5},
        },
        max_values_per_parameter=2,
        max_stage_candidates=4,
    )

    assert result['best_evaluation']['metric']['metric_name'] == 'mae'
    assert result['sequential_result']['stage_history'][-1]['stage'] == ForecastingStageName.FORECAST_HEAD.value


def test_okhs_runtime_bridge_supports_stubbed_backend(monkeypatch):
    class FakeInnerOKHSForecaster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, time_series, window_size=20):
            self.window_size = window_size
            self.series = np.asarray(time_series, dtype=float)
            return self

        def predict(self, time_series=None):
            del time_series
            horizon = int(self.kwargs['forecast_horizon'])
            return np.linspace(1.0, 1.0 + horizon - 1, num=horizon)

        def get_optimization_info(self):
            return {
                'resolved_window_size': int(self.window_size),
                'trajectory_representation_policy': 'projected',
                'trajectory_preprocessing': {'selected_rank': 4},
                'projection_metadata': {'decode_supported': True},
                'fdmd_prediction_diagnostics': {'anti_smoothing_diagnostics': {'collapse_detected': False}},
            }

    monkeypatch.setattr(
        'fedot_ind.core.models.ts_forecasting.okhs_fdmd_forecaster.OKHSForecaster',
        FakeInnerOKHSForecaster,
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
        'fedot_ind.core.models.ts_forecasting.stage_tuning_runtime.build_neural_forecast_head',
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
        params={'patch_len': 12, 'epochs': 2},
        metric_name='rmse',
    )

    assert evaluation.family == 'neural_forecaster'
    assert evaluation.diagnostics['forecast_head']['head_type'] == 'patch_tst_model'
