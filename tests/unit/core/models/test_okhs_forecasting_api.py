from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd as legacy_dmd_module
from fedot_ind.core.models.kernel.okhs_common import OKHSMethod, analyze_okhs_trajectory_preprocessing, \
    analyze_okhs_window_size, build_okhs_trajectory_matrix, build_okhs_trajectory_representation, \
    normalize_okhs_method, resolve_okhs_q, resolve_okhs_window_size
from fedot_ind.core.models.kernel.okhs_forecasting import OKHSForecaster
from fedot_ind.core.models.kernel.okhs_forecasting_torch import OKHSForecasterTorch
from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd_forecasting import DMDForecaster


class CountingQSelector:
    def __init__(self, value: float):
        self.value = value
        self.calls = 0
        self.last_verbose = None

    def analyze_and_suggest_q(self, trajectories, labels=None, verbose=True):
        self.calls += 1
        self.last_verbose = verbose
        return self.value


def test_normalize_okhs_method_supports_legacy_aliases():
    assert normalize_okhs_method("dmd") is OKHSMethod.DMD
    assert normalize_okhs_method("direct") is OKHSMethod.DIRECT
    assert normalize_okhs_method("occupation") is OKHSMethod.OCCUPATION


def test_resolve_okhs_q_fixed_policy_does_not_invoke_selector():
    selector = CountingQSelector(0.33)

    resolved_q = resolve_okhs_q(q=0.7, q_policy="fixed", trajectories=[np.arange(5)], q_selector=selector)

    assert resolved_q == 0.7
    assert selector.calls == 0


def test_resolve_okhs_window_size_fixed_policy_preserves_explicit_value():
    resolved_window = resolve_okhs_window_size(
        window_size=20,
        window_policy="fixed",
        time_series=np.arange(200, dtype=float),
        forecast_horizon=10,
    )

    assert resolved_window == 20


def test_resolve_okhs_window_size_adaptive_policy_respects_ratio_bounds():
    series = np.arange(700, dtype=float)

    resolved_window = resolve_okhs_window_size(
        window_size=8,
        window_policy="adaptive_heuristic",
        time_series=series,
        forecast_horizon=10,
    )

    assert 70 <= resolved_window <= 175


def test_analyze_okhs_window_size_cycle_aware_returns_diagnostics():
    time = np.arange(240, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0)

    diagnostics = analyze_okhs_window_size(
        window_size=8,
        window_policy="adaptive_cycle_aware",
        time_series=series,
        forecast_horizon=12,
    )

    assert diagnostics["window_policy"] == "adaptive_cycle_aware"
    assert diagnostics["dominant_period"] is not None
    assert 24 <= diagnostics["resolved_window_size"] <= 60
    assert diagnostics["trajectory_count"] == len(series) - diagnostics["resolved_window_size"]


def test_trajectory_preprocessing_disabled_for_fixed_window_policy():
    diagnostics = analyze_okhs_trajectory_preprocessing(
        time_series=np.arange(120, dtype=float),
        window_size=20,
        window_policy="fixed",
        forecast_horizon=8,
    )

    assert diagnostics["enabled"] is False
    assert diagnostics["trajectory_sampling_policy"] == "dense"
    assert diagnostics["trajectory_rank_policy"] == "none"
    assert diagnostics["effective_stride"] == 1


def test_build_okhs_trajectory_matrix_adaptive_policy_reduces_overlap_and_tracks_rank():
    time = np.arange(700, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0) + 0.05 * np.cos(2 * np.pi * time / 7.0)

    matrix, diagnostics = build_okhs_trajectory_matrix(
        time_series=series,
        window_size=8,
        window_policy="adaptive_cycle_aware",
        forecast_horizon=12,
    )

    assert matrix.ndim == 2
    assert diagnostics["enabled"] is True
    assert diagnostics["effective_stride"] > 1
    assert diagnostics["effective_stride"] <= max(1, int(round(diagnostics["window_size"] * 0.07)))
    assert diagnostics["trajectory_rank_policy"] == "explained_dispersion"
    assert diagnostics["selected_rank"] >= diagnostics["recommended_min_selected_rank"]
    assert diagnostics["rank_floor_applied"] in {True, False}
    assert diagnostics["selected_rank"] <= min(diagnostics["trajectory_matrix_shape_before"])
    assert diagnostics["trajectory_matrix_shape_after"][0] == matrix.shape[0]
    assert diagnostics["trajectory_matrix_shape_after"][1] == matrix.shape[1]
    assert diagnostics["effective_trajectory_count"] == matrix.shape[0]
    assert diagnostics["explained_variance_retained"] > 0.0


def test_build_okhs_trajectory_representation_projected_returns_latent_trajectories_and_basis():
    time = np.arange(700, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0) + 0.05 * np.cos(2 * np.pi * time / 11.0)

    representation = build_okhs_trajectory_representation(
        time_series=series,
        window_size=8,
        window_policy="adaptive_cycle_aware",
        forecast_horizon=12,
        trajectory_representation_policy="projected",
    )

    training_matrix = np.asarray(representation["training_matrix"], dtype=float)
    diagnostics = representation["trajectory_preprocessing"]
    projection_metadata = representation["projection_metadata"]
    projection_runtime = representation["projection_runtime"]

    assert training_matrix.ndim == 3
    assert diagnostics["trajectory_representation_policy"] == "projected"
    assert diagnostics["representation_policy"] == "projected"
    assert diagnostics["decode_supported"] is True
    assert projection_metadata["basis_shape"] is not None
    assert projection_metadata["projected_shape"] is not None
    assert projection_metadata["latent_stride"] >= 1
    assert projection_metadata["effective_latent_trajectory_count"] <= projection_metadata[
        "dense_latent_trajectory_count"]
    assert projection_runtime is not None
    assert training_matrix.shape == diagnostics["trajectory_matrix_shape_after"]
    decoded_state = projection_runtime["latent_state_matrix"] @ projection_runtime["basis"].T
    assert decoded_state.shape == projection_runtime["sampled_matrix"].shape


def test_projected_representation_adaptive_latent_stride_reduces_latent_trajectory_count():
    time = np.arange(700, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0) + 0.1 * np.cos(2 * np.pi * time / 9.0)

    representation = build_okhs_trajectory_representation(
        time_series=series,
        window_size=8,
        window_policy="adaptive_cycle_aware",
        forecast_horizon=12,
        trajectory_representation_policy="projected",
        latent_trajectory_stride_policy="adaptive",
    )

    diagnostics = representation["trajectory_preprocessing"]

    assert diagnostics["latent_trajectory_stride_policy"] == "adaptive"
    assert diagnostics["latent_stride"] >= 1
    assert diagnostics["effective_latent_trajectory_count"] <= diagnostics["dense_latent_trajectory_count"]
    assert diagnostics["latent_overlap_ratio"] <= 1.0


def test_okhs_forecaster_direct_fit_invokes_data_driven_selector_once(monkeypatch):
    selector = CountingQSelector(0.42)
    model = OKHSForecaster(
        q=0.7,
        forecast_horizon=3,
        method="direct",
        q_policy="data_driven",
        q_selector=selector,
    )

    monkeypatch.setattr(model, "_fit_direct_okhs", lambda time_series: None)

    model.fit(np.arange(12, dtype=float), window_size=4)

    assert model.method is OKHSMethod.DIRECT
    assert model.method_name_ == "direct"
    assert selector.calls == 1
    assert selector.last_verbose is False
    assert model.resolved_q_ == 0.42


def test_okhs_forecaster_fixed_policy_preserves_explicit_q(monkeypatch):
    selector = CountingQSelector(0.25)
    model = OKHSForecaster(
        q=0.8,
        forecast_horizon=3,
        method="direct",
        q_policy="fixed",
        q_selector=selector,
    )

    monkeypatch.setattr(model, "_fit_direct_okhs", lambda time_series: None)

    model.fit(np.arange(12, dtype=float), window_size=4)

    assert selector.calls == 0
    assert model.resolved_q_ == 0.8


def test_okhs_forecaster_adaptive_window_policy_overrides_too_small_window(monkeypatch):
    model = OKHSForecaster(
        q=0.8,
        forecast_horizon=10,
        method="direct",
        window_policy="adaptive_heuristic",
    )

    monkeypatch.setattr(model, "_fit_direct_okhs", lambda time_series: None)

    model.fit(np.arange(700, dtype=float), window_size=8)

    assert 70 <= model.resolved_window_size_ <= 175
    assert model.window_size_ == model.resolved_window_size_
    assert model.window_diagnostics_["resolved_window_size"] == model.resolved_window_size_


def test_okhs_forecaster_dmd_path_uses_wrapped_fractional_dmd(monkeypatch):
    state = {}

    class FakeWrappedFractionalDMD:
        def __init__(
                self,
                q,
                n_modes,
                mode_selection_policy='fixed',
                mode_energy_threshold=0.95,
                prediction_mode_selection_policy='adaptive_tail_energy',
                max_prediction_modes=None,
                min_prediction_modes=4,
                boundary_alignment_policy='tapered_offset',
                boundary_alignment_decay=4.0,
                prediction_stability_threshold=0.03,
        ):
            state["init"] = {
                "q": q,
                "n_modes": n_modes,
                "mode_selection_policy": mode_selection_policy,
                "mode_energy_threshold": mode_energy_threshold,
                "prediction_mode_selection_policy": prediction_mode_selection_policy,
                "max_prediction_modes": max_prediction_modes,
                "min_prediction_modes": min_prediction_modes,
                "boundary_alignment_policy": boundary_alignment_policy,
                "boundary_alignment_decay": boundary_alignment_decay,
                "prediction_stability_threshold": prediction_stability_threshold,
            }

        def fit(self, trajectories):
            state["fit"] = trajectories

        def predict_with_diagnostics(self, last_trajectory, future_times):
            state["predict"] = {"last_trajectory": last_trajectory, "future_times": future_times}
            return np.array([0.1, 0.2, 0.3]), {"boundary_discontinuity_abs_mean": 0.12}

        def get_fit_diagnostics_summary(self):
            return {"resolved_n_modes": 3, "fit_total_modes": 4}

    monkeypatch.setattr("fedot_ind.core.models.kernel.okhs_forecasting.FractionalDMD", FakeWrappedFractionalDMD)

    model = OKHSForecaster(q=0.75, forecast_horizon=3, n_modes=4, method="dmd", window_policy="fixed")
    series = np.arange(12, dtype=float)

    model.fit(series, window_size=4)
    prediction = model.predict()

    assert state["init"] == {
        "q": 0.75,
        "n_modes": 4,
        "mode_selection_policy": "fixed",
        "mode_energy_threshold": 0.95,
        "prediction_mode_selection_policy": "adaptive_tail_energy",
        "max_prediction_modes": None,
        "min_prediction_modes": 4,
        "boundary_alignment_policy": "tapered_offset",
        "boundary_alignment_decay": 4.0,
        "prediction_stability_threshold": 0.03,
    }
    assert len(state["fit"]) == len(series) - 4
    assert np.array_equal(state["predict"]["future_times"], np.array([4.0, 5.0, 6.0]))
    assert prediction.tolist() == [0.1, 0.2, 0.3]
    optimization_info = model.get_optimization_info()
    assert optimization_info["fdmd_fit_diagnostics"]["resolved_n_modes"] == 3
    assert optimization_info["fdmd_prediction_diagnostics"]["boundary_discontinuity_abs_mean"] == 0.12


def test_okhs_forecaster_adaptive_dmd_path_stores_preprocessing_diagnostics(monkeypatch):
    state = {}

    class FakeWrappedFractionalDMD:
        def __init__(
                self,
                q,
                n_modes,
                mode_selection_policy='fixed',
                mode_energy_threshold=0.95,
                prediction_mode_selection_policy='adaptive_tail_energy',
                max_prediction_modes=None,
                min_prediction_modes=4,
                boundary_alignment_policy='tapered_offset',
                boundary_alignment_decay=4.0,
                prediction_stability_threshold=0.03,
        ):
            del q, n_modes
            state["mode_selection"] = {
                "mode_selection_policy": mode_selection_policy,
                "mode_energy_threshold": mode_energy_threshold,
                "prediction_mode_selection_policy": prediction_mode_selection_policy,
                "max_prediction_modes": max_prediction_modes,
                "min_prediction_modes": min_prediction_modes,
                "boundary_alignment_policy": boundary_alignment_policy,
                "boundary_alignment_decay": boundary_alignment_decay,
                "prediction_stability_threshold": prediction_stability_threshold,
            }

        def fit(self, trajectories):
            state["fit"] = np.asarray(trajectories, dtype=float)

        def predict_with_diagnostics(self, last_trajectory, future_times):
            del last_trajectory, future_times
            return np.array([0.1, 0.2]), {"boundary_discontinuity_abs_mean": 0.05}

        def get_fit_diagnostics_summary(self):
            return {"resolved_n_modes": 2, "fit_total_modes": 5}

    monkeypatch.setattr("fedot_ind.core.models.kernel.okhs_forecasting.FractionalDMD", FakeWrappedFractionalDMD)

    time = np.arange(700, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0)
    model = OKHSForecaster(
        q=0.75,
        forecast_horizon=2,
        n_modes=4,
        method="dmd",
        window_policy="adaptive_cycle_aware",
    )

    model.fit(series, window_size=8)

    assert model.trajectory_preprocessing_ is not None
    assert model.trajectory_preprocessing_["enabled"] is True
    assert model.trajectory_preprocessing_["effective_stride"] > 1
    assert model.trajectory_preprocessing_["selected_rank"] >= model.trajectory_preprocessing_[
        "recommended_min_selected_rank"
    ]
    assert state["fit"].shape == model.trajectory_preprocessing_["trajectory_matrix_shape_after"]
    assert state["mode_selection"] == {
        "mode_selection_policy": "fixed",
        "mode_energy_threshold": 0.95,
        "prediction_mode_selection_policy": "adaptive_tail_energy",
        "max_prediction_modes": None,
        "min_prediction_modes": 4,
        "boundary_alignment_policy": "tapered_offset",
        "boundary_alignment_decay": 4.0,
        "prediction_stability_threshold": 0.03,
    }


def test_okhs_forecaster_projected_path_decodes_latent_prediction(monkeypatch):
    state = {}

    class FakeWrappedFractionalDMD:
        def __init__(
                self,
                q,
                n_modes,
                mode_selection_policy='fixed',
                mode_energy_threshold=0.95,
                prediction_mode_selection_policy='adaptive_tail_energy',
                max_prediction_modes=None,
                min_prediction_modes=4,
                boundary_alignment_policy='tapered_offset',
                boundary_alignment_decay=4.0,
                prediction_stability_threshold=0.03,
        ):
            del q, n_modes, mode_selection_policy, mode_energy_threshold
            del prediction_mode_selection_policy, max_prediction_modes, min_prediction_modes
            del boundary_alignment_policy, boundary_alignment_decay, prediction_stability_threshold

        def fit(self, trajectories):
            state["fit_shape"] = np.asarray(trajectories, dtype=float).shape

        def predict_with_diagnostics(self, last_trajectory, future_times):
            state["predict_shape"] = np.asarray(last_trajectory, dtype=float).shape
            horizon = len(future_times)
            rank = np.asarray(last_trajectory, dtype=float).shape[1]
            latent_prediction = np.ones((horizon, rank), dtype=float)
            return latent_prediction, {"boundary_discontinuity_abs_mean": 0.01}

        def get_fit_diagnostics_summary(self):
            return {"resolved_n_modes": 2, "fit_total_modes": 5}

    monkeypatch.setattr("fedot_ind.core.models.kernel.okhs_forecasting.FractionalDMD", FakeWrappedFractionalDMD)

    time = np.arange(700, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0)
    model = OKHSForecaster(
        q=0.75,
        forecast_horizon=3,
        n_modes=4,
        method="dmd",
        window_policy="adaptive_cycle_aware",
        trajectory_representation_policy="projected",
    )

    model.fit(series, window_size=8)
    prediction = model.predict()

    assert len(state["fit_shape"]) == 3
    assert len(state["predict_shape"]) == 2
    assert prediction.shape == (3,)
    assert model.trajectory_preprocessing_["representation_policy"] == "projected"
    assert model.trajectory_preprocessing_["latent_trajectory_stride_policy"] == "adaptive"
    assert model.dmd_prediction_diagnostics_["decoded_prediction_shape"][0] == 3
    assert model.dmd_prediction_diagnostics_["decode_supported"] is True


def test_okhs_forecaster_torch_accepts_occupation_method_and_resolves_q_once(monkeypatch):
    selector = CountingQSelector(0.55)
    model = OKHSForecasterTorch(
        params={
            "method": "occupation",
            "q": 0.7,
            "q_policy": "data_driven",
            "q_selector": selector,
            "forecast_horizon": 2,
            "epochs": 1,
        }
    )

    monkeypatch.setattr(
        model,
        "_hankelize",
        lambda time_series, window_size: setattr(
            model,
            "hankel_matrix",
            SimpleNamespace(
                trajectory_matrix=np.arange(24, dtype=float).reshape(6, 4),
                window_length=window_size,
            ),
        ),
    )
    monkeypatch.setattr(model, "_fit_direct_okhs_torch", lambda: model)

    result = model.fit(np.arange(20, dtype=float), window_size=4)

    assert result is model
    assert model.method is OKHSMethod.OCCUPATION
    assert model.method_name_ == "occupation"
    assert selector.calls == 1
    assert selector.last_verbose is False
    assert model.resolved_q_ == 0.55


def test_okhs_forecaster_torch_accepts_legacy_kwargs_aliases(monkeypatch):
    selector = CountingQSelector(0.61)
    model = OKHSForecasterTorch(
        q=0.7,
        horizon=3,
        max_epochs=5,
        method="direct",
        q_policy="data_driven",
        q_selector=selector,
        device="cpu",
    )

    monkeypatch.setattr(
        model,
        "_hankelize",
        lambda time_series, window_size: setattr(
            model,
            "hankel_matrix",
            SimpleNamespace(
                trajectory_matrix=np.arange(20, dtype=float).reshape(5, 4),
                window_length=window_size,
            ),
        ),
    )
    monkeypatch.setattr(model, "_fit_direct_okhs_torch", lambda: model)

    result = model.fit(np.arange(16, dtype=float), window_size=4)

    assert result is model
    assert model.forecast_horizon == 3
    assert model.epochs == 5
    assert str(model.device) == "cpu"
    assert model.method_name_ == "direct"
    assert model.resolved_q_ == 0.61
    assert selector.calls == 1


def test_okhs_forecaster_torch_adaptive_window_policy_updates_hankel_size(monkeypatch):
    model = OKHSForecasterTorch(
        params={
            "method": "occupation",
            "forecast_horizon": 10,
            "window_policy": "adaptive_heuristic",
            "epochs": 1,
        }
    )
    state = {}

    def fake_hankelize(time_series, window_size):
        del time_series
        state["window_size"] = window_size
        model.hankel_matrix = SimpleNamespace(
            trajectory_matrix=np.arange(400, dtype=float).reshape(20, 20),
            window_length=window_size,
        )

    monkeypatch.setattr(
        model,
        "_hankelize",
        fake_hankelize,
    )
    monkeypatch.setattr(model, "_fit_direct_okhs_torch", lambda: model)

    result = model.fit(np.arange(700, dtype=float), window_size=8)

    assert result is model
    assert 70 <= model.resolved_window_size_ <= 175
    assert state["window_size"] == model.resolved_window_size_
    assert model.window_diagnostics_["resolved_window_size"] == model.resolved_window_size_


def test_okhs_forecaster_torch_adaptive_dmd_path_updates_hankel_matrix_with_preprocessing(monkeypatch):
    time = np.arange(700, dtype=float)
    series = np.sin(2 * np.pi * time / 24.0)
    model = OKHSForecasterTorch(
        params={
            "method": "dmd",
            "forecast_horizon": 4,
            "window_policy": "adaptive_cycle_aware",
            "device": "cpu",
            "epochs": 1,
        }
    )
    state = {}

    def fake_dmd_fit(trajectory_matrix, window_size):
        state["trajectory_matrix"] = np.asarray(trajectory_matrix, dtype=float)
        state["window_size"] = window_size
        return model.dmd_model

    monkeypatch.setattr(DMDForecaster, "fit", lambda self, trajectory_matrix, window_size: fake_dmd_fit(
        trajectory_matrix, window_size))

    result = model.fit(series, window_size=8)

    assert result is model.dmd_model
    assert model.trajectory_preprocessing_ is not None
    assert model.trajectory_preprocessing_["effective_stride"] > 1
    assert model.trajectory_preprocessing_["selected_rank"] >= model.trajectory_preprocessing_[
        "recommended_min_selected_rank"
    ]
    assert state["window_size"] == model.resolved_window_size_
    assert state["trajectory_matrix"].shape == model.trajectory_preprocessing_["trajectory_matrix_shape_after"]


def test_dmd_forecaster_records_training_history(monkeypatch):
    model = DMDForecaster(
        forecast_horizon=2,
        epochs=3,
        use_koopman=True,
        device="cpu",
    )

    def simple_setup(input_dim):
        model.n_modes = input_dim
        model.K = nn.Parameter(torch.eye(input_dim, device=model.device))
        model.encoder = nn.Identity()
        model.decoder = nn.Identity()

    monkeypatch.setattr(model, "_setup_koopman_model", simple_setup)

    trajectories = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
        ]
    )

    result = model.fit(trajectories, window_size=4)

    assert result is model
    assert len(model.training_history_) == 3
    assert all(loss >= 0.0 for loss in model.training_history_)


def test_legacy_fractional_dmd_wraps_current_okhs_pipeline(monkeypatch):
    state = {}

    class FakeOKHSTransformer:
        def __init__(self, kernel, q, n_quad_points, dt):
            state["okhs_init"] = {"q": q, "n_quad_points": n_quad_points, "dt": dt, "kernel": kernel}

        def fit(self, trajectories):
            state["okhs_fit"] = trajectories

    class FakeLiouvilleOperator:
        def __init__(self, okhs_transformer, n_quad_points):
            state["liouville_init"] = {"n_quad_points": n_quad_points, "okhs_transformer": okhs_transformer}
            self.eigenvalues_ = np.array([3.0 + 0.0j, 2.0 + 0.0j, 1.0 + 0.0j])
            self.eigenvectors_ = np.eye(3)

        def fit(self):
            state["liouville_fit"] = True

    class FakeCoreFDMD:
        def __init__(self, liouville_operator, n_quad_points, regularization):
            state["fdmd_init"] = {
                "n_quad_points": n_quad_points,
                "regularization": regularization,
                "liouville_operator": liouville_operator,
            }
            self.modes_ = np.array([[10.0], [20.0], [30.0]])

        def fit(self, trajectories):
            state["fdmd_fit"] = trajectories

        def predict_with_diagnostics(
                self,
                initial_trajectory,
                times,
                stability_threshold=None,
                prediction_mode_selection_policy="all_stable",
                max_prediction_modes=None,
                min_prediction_modes=4,
        ):
            state["fdmd_predict"] = {
                "initial_trajectory": initial_trajectory,
                "times": times,
                "stability_threshold": stability_threshold,
                "prediction_mode_selection_policy": prediction_mode_selection_policy,
                "max_prediction_modes": max_prediction_modes,
                "min_prediction_modes": min_prediction_modes,
            }
            return np.arange(len(times), dtype=float).reshape(-1, 1), {"boundary_discontinuity_abs_mean": 0.25}

    monkeypatch.setattr(legacy_dmd_module, "OKHSTransformer", FakeOKHSTransformer)
    monkeypatch.setattr(legacy_dmd_module, "FractionalLiouvilleOperator", FakeLiouvilleOperator)
    monkeypatch.setattr(legacy_dmd_module, "OKHSFractionalDMD", FakeCoreFDMD)

    model = legacy_dmd_module.FractionalDMD(q=0.8, n_modes=2, n_quad_points=4, dt=0.5, regularization=1e-6)
    trajectories = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0])]

    fit_result = model.fit(trajectories)
    prediction = model.predict(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    assert fit_result is model
    assert state["liouville_fit"] is True
    assert all(traj.ndim == 2 for traj in state["okhs_fit"])
    assert model.eigenvalues_.tolist() == [3.0 + 0.0j, 2.0 + 0.0j]
    assert model.modes_.shape == (2, 1)
    assert state["fdmd_predict"]["initial_trajectory"].shape == (2, 1)
    assert state["fdmd_predict"]["stability_threshold"] == 0.03
    assert state["fdmd_predict"]["prediction_mode_selection_policy"] == "adaptive_tail_energy"
    assert state["fdmd_predict"]["max_prediction_modes"] is None
    assert state["fdmd_predict"]["min_prediction_modes"] == 4
    assert np.allclose(prediction, np.array([2.0, 1.2706705664732254, 2.0366312777774684]))
    assert model.get_fit_diagnostics_summary()["resolved_n_modes"] == 2
    assert model.last_prediction_diagnostics_["boundary_discontinuity_abs_mean"] == 0.25
    assert model.last_prediction_diagnostics_["boundary_alignment_applied"] is True
    assert model.last_prediction_diagnostics_["boundary_alignment_offset"] == [2.0]
    assert np.allclose(model.last_prediction_diagnostics_["boundary_alignment_weights"],
                       [1.0, 0.1353352832366127, 0.01831563888873418])


def test_legacy_fractional_dmd_can_disable_boundary_alignment():
    model = legacy_dmd_module.FractionalDMD(boundary_alignment_policy="none")
    model.fdmd_ = SimpleNamespace(
        predict_with_diagnostics=lambda initial_trajectory, times: (
            np.arange(len(times), dtype=float).reshape(-1, 1),
            {"boundary_discontinuity_abs_mean": 1.0},
        )
    )
    model.eigenvalues_ = np.array([1.0 + 0.0j])
    model.eigenvectors_ = np.eye(1)
    model.modes_ = np.eye(1)
    model.okhs = SimpleNamespace(gram_condition_number_=1.0)
    model.resolved_n_modes_ = 1

    prediction = model.predict(np.array([1.0, 2.0]), 2)

    assert prediction.tolist() == [0.0, 1.0]
    assert model.last_prediction_diagnostics_["boundary_alignment_applied"] is False


def test_legacy_fractional_dmd_supports_energy_mode_selection():
    model = legacy_dmd_module.FractionalDMD(mode_selection_policy="energy", mode_energy_threshold=0.75)
    model.eigenvalues_ = np.array([5.0 + 0.0j, 3.0 + 0.0j, 1.0 + 0.0j])
    model.eigenvectors_ = np.eye(3)
    model.modes_ = np.eye(3)
    model.liouville_operator_ = SimpleNamespace(
        eigenvalues_=model.eigenvalues_.copy(),
        eigenvectors_=model.eigenvectors_.copy(),
    )
    model.fdmd_ = SimpleNamespace(modes_=model.modes_.copy())

    model._select_modes()

    assert model.resolved_n_modes_ == 2
    assert len(model.eigenvalues_) == 2


def test_legacy_fractional_dmd_scalar_horizon_uses_continuation_time_grid(monkeypatch):
    model = legacy_dmd_module.FractionalDMD(dt=0.5)
    model.fdmd_ = SimpleNamespace(
        predict_with_diagnostics=lambda initial_trajectory, times: (
            np.ones((len(times), 1)),
            {"times": np.asarray(times, dtype=float).tolist()},
        )
    )
    model.eigenvalues_ = np.array([1.0 + 0.0j])
    model.eigenvectors_ = np.eye(1)
    model.modes_ = np.eye(1)
    model.okhs = SimpleNamespace(gram_condition_number_=1.0)
    model.resolved_n_modes_ = 1

    prediction = model.predict(np.array([1.0, 2.0, 3.0, 4.0]), 3)

    assert np.allclose(prediction, np.array([4.0, 1.406005849709838, 1.0549469166662025]))
    assert model.last_prediction_diagnostics_["prediction_time_grid"] == [2.0, 2.5, 3.0]
    assert model.last_prediction_diagnostics_["boundary_alignment_applied"] is True
