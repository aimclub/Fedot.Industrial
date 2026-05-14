from fedot_ind.core.models.ts_forecasting.dmd_models.okhs_fdmd_forecaster import OKHSFDMDForecaster
from fedot_ind.core.models.ts_forecasting.lagged_model.low_rank_lagged_ridge_forecaster import \
    LowRankLaggedRidgeForecaster
from fedot_ind.core.models.ts_forecasting.lagged_model.lagged_ridge_forecaster import LaggedRidgeForecaster
from fedot_ind.core.models.ts_forecasting.ensemble_models.hybrid_ensemble_forecaster import HybridEnsembleForecaster
import numpy as np
import pytest

pytest.importorskip('torch')


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


def test_okhs_fdmd_forecaster_exposes_stage_aware_diagnostics(monkeypatch):
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
            return np.linspace(10.0, 10.0 + horizon - 1, num=horizon)

        def get_optimization_info(self):
            return {
                'q': 0.7,
                'q_policy': 'fixed',
                'window_policy': 'adaptive_cycle_aware',
                'resolved_window_size': int(self.window_size),
                'trajectory_representation_policy': 'projected',
                'trajectory_rank_policy': 'explained_dispersion',
                'window_diagnostics': {'window_fraction': 0.2, 'expected_overlap_ratio': 0.95},
                'trajectory_preprocessing': {
                    'effective_stride': 2,
                    'dense_trajectory_count': 40,
                    'effective_trajectory_count': 20,
                    'trajectory_matrix_shape_before': (40, int(self.window_size)),
                    'trajectory_matrix_shape_after': (12, 16, 4),
                    'selected_rank': 4,
                    'raw_selected_rank': 3,
                    'requested_rank_floor': 4,
                    'applied_rank_floor': 4,
                    'rank_floor_applied': True,
                    'explained_variance_retained': 0.94,
                    'compression_ratio': 0.33,
                },
                'projection_metadata': {
                    'projected_shape': (20, 4),
                    'basis_shape': (int(self.window_size), 4),
                    'decode_supported': True,
                    'decode_reconstruction_error': 0.05,
                    'latent_window_size': 16,
                    'latent_stride': 2,
                    'latent_overlap_ratio': 0.875,
                },
                'mode_selection_policy': 'energy',
                'mode_energy_threshold': 0.95,
                'prediction_mode_selection_policy': 'adaptive_tail_energy',
                'max_prediction_modes': None,
                'min_prediction_modes': 4,
                'boundary_alignment_policy': 'tapered_offset',
                'boundary_alignment_decay': 4.0,
                'prediction_stability_threshold': 0.03,
                'fdmd_fit_diagnostics': {'resolved_n_modes': 4},
                'fdmd_prediction_diagnostics': {
                    'n_selected_prediction_modes': 4,
                    'anti_smoothing_diagnostics': {'collapse_detected': False, 'correction_applied': False},
                },
            }

    monkeypatch.setattr(
        'fedot_ind.core.models.ts_forecasting.okhs_fdmd_forecaster.OKHSForecaster',
        FakeInnerOKHSForecaster,
    )

    series = _trend_with_oscillation(96)
    model = OKHSFDMDForecaster(forecast_horizon=6, window_size=18)
    model.fit(series)
    forecast = model.predict(series)
    diagnostics = model.get_diagnostics()

    assert forecast.shape == (6,)
    assert diagnostics['model_family'] == 'operator_model'
    assert diagnostics['trajectory_transform']['resolved_window_size'] == 18
    assert diagnostics['decomposition']['decode_supported'] is True
    assert diagnostics['rank_truncation']['selected_rank'] == 4
    assert diagnostics['forecast_head']['prediction_diagnostics']['n_selected_prediction_modes'] == 4
