import numpy as np
import pytest

pytest.importorskip('torch')

from fedot_ind.core.models.ts_forecasting.hybrid_ensemble_forecaster import HybridEnsembleForecaster
from fedot_ind.core.models.ts_forecasting.lagged_ridge_forecaster import LaggedRidgeForecaster
from fedot_ind.core.models.ts_forecasting.low_rank_lagged_ridge_forecaster import LowRankLaggedRidgeForecaster


def _trend_with_oscillation(length: int = 96) -> np.ndarray:
    time = np.arange(length, dtype=float)
    return 0.25 * time + 1.5 * np.sin(time / 6.0)


def test_lagged_ridge_forecaster_returns_horizon_vector_and_stage_diagnostics():
    series = _trend_with_oscillation()
    model = LaggedRidgeForecaster(forecast_horizon=8, window_size=18, stride=2, alpha=0.5)

    model.fit(series)
    forecast = model.predict(series)
    diagnostics = model.get_diagnostics()

    assert forecast.shape == (8,)
    assert 'trajectory_transform' in diagnostics
    assert 'forecast_head' in diagnostics
    assert diagnostics['model_family'] == 'lagged_linear'


def test_low_rank_lagged_ridge_forecaster_exposes_decomposition_and_rank_metadata():
    series = _trend_with_oscillation(120)
    model = LowRankLaggedRidgeForecaster(
        forecast_horizon=10,
        window_size=24,
        stride=2,
        alpha=1.0,
        explained_variance=0.92,
    )

    model.fit(series)
    forecast = model.predict(series)
    diagnostics = model.get_diagnostics()

    assert forecast.shape == (10,)
    assert diagnostics['model_family'] == 'low_rank_linear'
    assert diagnostics['rank_truncation']['selected_rank'] >= 2
    assert diagnostics['decomposition']['basis_shape'][0] == diagnostics['trajectory_transform']['features_shape'][1]


def test_hybrid_ensemble_forecaster_learns_normalized_branch_weights():
    series = _trend_with_oscillation(150)
    model = HybridEnsembleForecaster(
        forecast_horizon=8,
        complex_branch='havok',
        lagged_params={'window_size': 20, 'alpha': 0.5},
        low_rank_params={'window_size': 20, 'explained_variance': 0.9},
        complex_params={'window_size': 18, 'rank': 3},
    )

    model.fit(series)
    forecast = model.predict(series)
    diagnostics = model.get_diagnostics()
    weights = diagnostics['ensemble_head']['weights']

    assert forecast.shape == (8,)
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-6
    assert set(diagnostics['branch_names']) == {'lagged_linear', 'low_rank_linear', 'operator_model'}
