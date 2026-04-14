import numpy as np

from fedot_ind.core.models.kernel.okhs_runtime import (
    build_okhs_dmd_model,
    build_okhs_direct_model,
    build_dense_okhs_trajectories,
    build_okhs_fit_plan,
    build_okhs_optimization_info,
    build_okhs_prediction_plan,
    decode_okhs_projected_prediction,
    build_okhs_stage_diagnostics,
    execute_okhs_dmd_prediction_plan,
    postprocess_okhs_dmd_forecast,
    resolve_okhs_prediction_time_grid,
    run_okhs_direct_prediction,
    run_okhs_dmd_prediction,
)


class CountingQSelector:
    def __init__(self, value: float):
        self.value = value
        self.calls = 0

    def analyze_and_suggest_q(self, trajectories, labels=None, verbose=True):
        del trajectories, labels, verbose
        self.calls += 1
        return self.value


def test_build_dense_okhs_trajectories_preserves_legacy_trajectory_count():
    series = np.arange(12, dtype=float)
    trajectories = build_dense_okhs_trajectories(series, window_size=4)

    assert len(trajectories) == 8
    assert np.array_equal(trajectories[0], np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(trajectories[-1], np.array([7.0, 8.0, 9.0, 10.0]))


def test_build_okhs_fit_plan_for_direct_mode_returns_dense_trajectories_and_resolved_q():
    selector = CountingQSelector(0.61)
    series = np.arange(64, dtype=float)

    fit_plan = build_okhs_fit_plan(
        time_series=series,
        window_size=8,
        method='direct',
        forecast_horizon=4,
        q=0.7,
        q_policy='data_driven',
        q_selector=selector,
        window_policy='fixed',
    )

    assert fit_plan['resolved_window_size'] == 8
    assert fit_plan['trajectory_preprocessing'] is None
    assert fit_plan['projection_metadata'] is None
    assert len(fit_plan['trajectories']) == len(series) - 8
    assert fit_plan['resolved_q'] == 0.61
    assert selector.calls == 1


def test_build_okhs_fit_plan_for_dmd_mode_returns_projection_metadata():
    time = np.arange(180, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0)

    fit_plan = build_okhs_fit_plan(
        time_series=series,
        window_size=8,
        method='dmd',
        forecast_horizon=12,
        q=0.7,
        q_policy='fixed',
        window_policy='adaptive_cycle_aware',
        trajectory_representation_policy='projected',
    )

    assert fit_plan['trajectory_preprocessing'] is not None
    assert fit_plan['projection_metadata']['decode_supported'] is True
    assert np.asarray(fit_plan['trajectories']).ndim == 3


def test_build_okhs_stage_diagnostics_returns_primitive_vocabulary():
    optimization_info = {
        'q': 0.7,
        'q_policy': 'fixed',
        'window_policy': 'adaptive_cycle_aware',
        'resolved_window_size': 18,
        'trajectory_rank_policy': 'explained_dispersion',
        'trajectory_representation_policy': 'projected',
        'window_diagnostics': {'window_fraction': 0.2, 'expected_overlap_ratio': 0.95},
        'trajectory_preprocessing': {
            'effective_stride': 2,
            'dense_trajectory_count': 42,
            'effective_trajectory_count': 21,
            'trajectory_matrix_shape_before': (42, 18),
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
            'projected_shape': (21, 4),
            'basis_shape': (18, 4),
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
        'fdmd_prediction_diagnostics': {'n_selected_prediction_modes': 4},
    }

    stage_diagnostics = build_okhs_stage_diagnostics(optimization_info)

    assert stage_diagnostics['trajectory_transform']['resolved_window_size'] == 18
    assert stage_diagnostics['decomposition']['decode_supported'] is True
    assert stage_diagnostics['rank_truncation']['selected_rank'] == 4
    assert stage_diagnostics['forecast_head']['prediction_diagnostics']['n_selected_prediction_modes'] == 4


def test_run_okhs_dmd_prediction_returns_prediction_and_time_grid():
    class FakeModel:
        dt = 0.5

        def predict_with_diagnostics(self, initial_trajectory, future_times):
            del initial_trajectory
            return np.arange(len(future_times), dtype=float), {'boundary_discontinuity_abs_mean': 0.2}

    prediction, diagnostics = run_okhs_dmd_prediction(
        model=FakeModel(),
        initial_trajectory=np.array([1.0, 2.0, 3.0]),
        forecast_horizon=4,
    )

    assert prediction.shape == (4,)
    assert diagnostics['boundary_discontinuity_abs_mean'] == 0.2
    assert diagnostics['prediction_time_grid'] == resolve_okhs_prediction_time_grid(3, 4, 0.5).tolist()


def test_build_okhs_dmd_model_falls_back_when_factory_rejects_device():
    captured = {}

    class FakeDMD:
        def __init__(self, **kwargs):
            if 'device' in kwargs:
                raise TypeError('unexpected device')
            captured.update(kwargs)

    model = build_okhs_dmd_model(
        dmd_factory=FakeDMD,
        resolved_q=0.61,
        n_modes=4,
        mode_selection_policy='energy',
        mode_energy_threshold=0.95,
        prediction_mode_selection_policy='adaptive_tail_energy',
        max_prediction_modes=None,
        min_prediction_modes=3,
        boundary_alignment_policy='tapered_offset',
        boundary_alignment_decay=4.0,
        prediction_stability_threshold=0.03,
        device='cpu',
    )

    assert isinstance(model, FakeDMD)
    assert captured['q'] == 0.61
    assert 'device' not in captured


def test_build_okhs_direct_model_returns_kernel_weights_and_fit_diagnostics():
    class FakeKernel:
        def __init__(self, q):
            self.q = q

        def compute_gram_matrix(self, trajectories):
            size = len(trajectories)
            return np.eye(size, dtype=float)

    direct_model = build_okhs_direct_model(
        kernel_factory=FakeKernel,
        resolved_q=0.55,
        trajectories=[
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0]),
        ],
    )

    assert isinstance(direct_model['kernel'], FakeKernel)
    assert direct_model['kernel'].q == 0.55
    assert direct_model['gram_matrix'].shape == (2, 2)
    assert direct_model['weights'].shape == (2,)
    assert direct_model['fit_diagnostics']['n_reference_trajectories'] == 2
    assert direct_model['fit_diagnostics']['weights_shape'] == (2,)


def test_run_okhs_direct_prediction_rolls_last_trajectory_and_returns_diagnostics():
    class FakeKernel:
        @staticmethod
        def _compute_trajectory_kernel(current_trajectory, train_trajectory):
            del train_trajectory
            return float(np.asarray(current_trajectory, dtype=float).reshape(-1)[-1])

    forecast, diagnostics = run_okhs_direct_prediction(
        kernel=FakeKernel(),
        reference_trajectories=[
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        ],
        last_trajectory=np.array([2.0, 3.0, 4.0]),
        weights=np.array([0.25, 0.75]),
        forecast_horizon=3,
    )

    assert forecast.tolist() == [4.0, 4.0, 4.0]
    assert diagnostics['reference_trajectory_count'] == 2
    assert diagnostics['forecast_horizon'] == 3
    assert diagnostics['last_trajectory_length'] == 3


def test_decode_okhs_projected_prediction_returns_forecast_and_decode_metadata():
    forecast, diagnostics = decode_okhs_projected_prediction(
        initial_trajectory=np.array([[1.0, 0.0], [2.0, 0.0]]),
        decoded_initial_trajectory=np.array([[1.0, 0.0], [2.0, 0.0]]),
        latent_prediction=np.array([[3.0, 0.0], [4.0, 0.0]]),
        projection_runtime={
            'basis': np.eye(2),
            'latent_state_matrix': np.array([[1.0, 0.0], [2.0, 0.0]]),
        },
    )

    assert forecast.tolist() == [0.0, 0.0]
    assert diagnostics['decode_supported'] is True
    assert diagnostics['basis_shape'] == (2, 2)
    assert diagnostics['latent_prediction_shape'] == (2, 2)


def test_build_okhs_prediction_plan_uses_projected_runtime_when_available():
    plan = build_okhs_prediction_plan(
        trajectories=[np.array([1.0, 2.0, 3.0])],
        time_series=None,
        train_series=np.arange(10, dtype=float),
        window_size=3,
        trajectory_preprocessing={'effective_stride': 1},
        projection_metadata={'representation_policy': 'projected', 'decode_supported': True},
        projection_runtime={
            'latent_window_size': 2,
            'latent_state_matrix': np.array([[1.0, 0.0], [2.0, 0.0]]),
            'sampled_matrix': np.array([[10.0, 11.0], [12.0, 13.0]]),
            'basis': np.eye(2),
        },
    )

    assert plan['representation_policy'] == 'projected'
    assert plan['initial_trajectory'].shape == (2, 2)
    assert plan['decoded_initial_trajectory'].shape == (2, 2)


def test_execute_okhs_dmd_prediction_plan_decodes_projected_predictions():
    class FakeModel:
        dt = 1.0

        def predict_with_diagnostics(self, initial_trajectory, future_times):
            del initial_trajectory
            return np.vstack([future_times, np.zeros_like(future_times)]).T, {'n_selected_prediction_modes': 2}

    forecast, diagnostics = execute_okhs_dmd_prediction_plan(
        model=FakeModel(),
        prediction_plan={
            'representation_policy': 'projected',
            'initial_trajectory': np.array([[1.0, 0.0], [2.0, 0.0]]),
            'decoded_initial_trajectory': np.array([[1.0, 0.0], [2.0, 0.0]]),
        },
        forecast_horizon=3,
        projection_runtime={
            'basis': np.eye(2),
            'latent_state_matrix': np.array([[1.0, 0.0], [2.0, 0.0]]),
        },
    )

    assert forecast.tolist() == [0.0, 0.0, 0.0]
    assert diagnostics['decode_supported'] is True
    assert diagnostics['n_selected_prediction_modes'] == 2


def test_postprocess_okhs_dmd_forecast_merges_anti_smoothing_diagnostics():
    time = np.arange(80, dtype=float)
    train = np.sin(2 * np.pi * time / 8.0)
    forecast = np.linspace(0.2, 0.05, num=10)

    corrected, merged = postprocess_okhs_dmd_forecast(
        train_series=train,
        forecast=forecast,
        forecast_horizon=10,
        prediction_diagnostics={'boundary_discontinuity_abs_mean': 0.01},
        anti_smoothing_policy='residual_bridge',
        anti_smoothing_tail_window=16,
    )

    assert corrected.shape == (10,)
    assert merged['boundary_discontinuity_abs_mean'] == 0.01
    assert merged['anti_smoothing_diagnostics']['collapse_detected'] is True
    assert merged['anti_smoothing_diagnostics']['correction_applied'] is True


def test_build_okhs_optimization_info_collects_stage_diagnostics_and_optional_fields():
    class FakeModel:
        def get_fit_diagnostics_summary(self):
            return {'resolved_n_modes': 3}

    info = build_okhs_optimization_info(
        method_name='fdmd',
        resolved_q=0.7,
        q_policy='fixed',
        forecast_horizon=6,
        window_policy='adaptive_cycle_aware',
        trajectory_sampling_policy='dense',
        trajectory_rank_policy='explained_dispersion',
        trajectory_rank_value=None,
        trajectory_representation_policy='projected',
        latent_trajectory_stride_policy='adaptive',
        latent_trajectory_stride=2,
        resolved_window_size=18,
        window_diagnostics={'window_fraction': 0.2},
        trajectory_preprocessing={'selected_rank': 4},
        projection_metadata={'decode_supported': True},
        mode_selection_policy='energy',
        mode_energy_threshold=0.95,
        prediction_mode_selection_policy='adaptive_tail_energy',
        max_prediction_modes=None,
        min_prediction_modes=4,
        boundary_alignment_policy='tapered_offset',
        boundary_alignment_decay=4.0,
        prediction_stability_threshold=0.03,
        anti_smoothing_policy='residual_bridge',
        anti_smoothing_tail_window=12,
        anti_smoothing_amplitude_ratio=0.35,
        anti_smoothing_monotone_ratio=0.9,
        anti_smoothing_oscillation_floor=0.25,
        anti_smoothing_decay=2.5,
        anti_smoothing_target_amplitude_ratio=0.8,
        model=FakeModel(),
        dmd_prediction_diagnostics={'boundary_discontinuity_abs_mean': 0.1},
        weights=np.array([1.0, 2.0, 3.0]),
    )

    assert info['fdmd_fit_diagnostics']['resolved_n_modes'] == 3
    assert info['weights_norm'] > 0
    assert info['head_runtime'] == 'fdmd'
    assert info['stage_diagnostics']['forecast_head']['prediction_diagnostics'][
               'boundary_discontinuity_abs_mean'] == 0.1


def test_build_okhs_optimization_info_supports_direct_runtime_diagnostics():
    info = build_okhs_optimization_info(
        method_name='direct',
        resolved_q=0.7,
        q_policy='fixed',
        forecast_horizon=4,
        window_policy='fixed',
        trajectory_sampling_policy='dense',
        trajectory_rank_policy='none',
        trajectory_rank_value=None,
        trajectory_representation_policy='none',
        latent_trajectory_stride_policy='none',
        latent_trajectory_stride=None,
        resolved_window_size=12,
        window_diagnostics={'window_fraction': 0.25},
        trajectory_preprocessing=None,
        projection_metadata=None,
        mode_selection_policy='fixed',
        mode_energy_threshold=0.95,
        prediction_mode_selection_policy='fixed',
        max_prediction_modes=None,
        min_prediction_modes=4,
        boundary_alignment_policy='none',
        boundary_alignment_decay=1.0,
        prediction_stability_threshold=None,
        anti_smoothing_policy='none',
        anti_smoothing_tail_window=None,
        anti_smoothing_amplitude_ratio=0.35,
        anti_smoothing_monotone_ratio=0.9,
        anti_smoothing_oscillation_floor=0.25,
        anti_smoothing_decay=2.5,
        anti_smoothing_target_amplitude_ratio=0.8,
        direct_fit_diagnostics={'gram_matrix_shape': (4, 4)},
        direct_prediction_diagnostics={'reference_trajectory_count': 4},
        weights=np.array([1.0, 2.0]),
    )

    assert info['head_runtime'] == 'direct_okhs'
    assert info['direct_fit_diagnostics']['gram_matrix_shape'] == (4, 4)
    assert info['direct_prediction_diagnostics']['reference_trajectory_count'] == 4
    assert info['stage_diagnostics']['forecast_head']['fit_diagnostics']['gram_matrix_shape'] == (4, 4)
