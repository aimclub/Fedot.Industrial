from types import SimpleNamespace

import numpy as np
import pytest

import fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.okhs as okhs_module
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.okhs import (
    FractionalDMD,
    FractionalLiouvilleOperator,
    OKHSTransformer,
    RegularizationPolicy,
    StabilityPolicy,
    select_stable_modes,
    sort_eigendecomposition,
    validate_initial_coefficient_feasibility,
    validate_liouville_shapes,
)


class DotKernel:
    def _compute_single_kernel(self, x, y):
        left = np.atleast_1d(x).astype(float)
        right = np.atleast_1d(y).astype(float)
        return float(left @ right + 1.0)


class BatchDotKernel(DotKernel):
    def __init__(self):
        self.batch_calls = 0

    def _compute_batch_kernel(self, x, y):
        self.batch_calls += 1
        left = np.asarray(x, dtype=float)
        right = np.asarray(y, dtype=float)
        return np.sum(left * right, axis=-1) + 1.0


class FakeTqdmBar:
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total", 0)
        self.updated = 0
        self.postfix_calls = []
        self.closed = False

    def set_postfix(self, payload, refresh=False):
        self.postfix_calls.append((payload, refresh))

    def update(self, amount):
        self.updated += amount

    def close(self):
        self.closed = True


def _reference_gram_entry(transformer, trajectory_i, trajectory_j):
    T_i = transformer._get_trajectory_duration(trajectory_i)
    T_j = transformer._get_trajectory_duration(trajectory_j)
    nodes, weights = transformer._get_jacobi_rule()
    t_nodes_i = T_i * (nodes + 1) / 2.0
    t_nodes_j = T_j * (nodes + 1) / 2.0
    vals_i = np.array([transformer._evaluate_trajectory_at_time(trajectory_i, t, T_i) for t in t_nodes_i])
    vals_j = np.array([transformer._evaluate_trajectory_at_time(trajectory_j, t, T_j) for t in t_nodes_j])

    gram_entry = 0.0
    for k in range(transformer.n_quad_points):
        for m in range(transformer.n_quad_points):
            gram_entry += weights[k] * weights[m] * transformer.kernel._compute_single_kernel(vals_j[m], vals_i[k])
    return (transformer.C_q ** 2) * ((T_i / 2.0) ** transformer.q) * ((T_j / 2.0) ** transformer.q) * gram_entry


def _reference_liouville_entry(transformer, trajectory_i, trajectory_j):
    T_i = transformer._get_trajectory_duration(trajectory_i)
    T_j = transformer._get_trajectory_duration(trajectory_j)
    if T_i <= 1e-14 or T_j <= 1e-14:
        return 0.0

    nodes, weights = transformer._get_jacobi_rule()
    tau_nodes = T_j * (nodes + 1) / 2.0
    xi_i_T = transformer._evaluate_trajectory_at_time(trajectory_i, T_i, T_i)
    xi_i_0 = transformer._evaluate_trajectory_at_time(trajectory_i, 0.0, T_i)
    xi_j_vals = [transformer._evaluate_trajectory_at_time(trajectory_j, tau, T_j) for tau in tau_nodes]

    integral_sum = 0.0
    for k in range(transformer.n_quad_points):
        integral_sum += weights[k] * (
                transformer.kernel._compute_single_kernel(xi_j_vals[k], xi_i_T)
                - transformer.kernel._compute_single_kernel(xi_j_vals[k], xi_i_0)
        )
    return transformer.C_q * ((T_j / 2.0) ** transformer.q) * integral_sum


def test_gram_matrix_is_symmetric():
    trajectories = [
        np.array([[0.0], [0.5], [1.0]]),
        np.array([[1.0], [1.5], [2.0]]),
    ]
    transformer = OKHSTransformer(kernel=DotKernel(), q=0.8, n_quad_points=4, dt=1.0)

    transformer.fit(trajectories)

    assert np.allclose(transformer.gram_matrix_, transformer.gram_matrix_.T)


def test_cached_gram_matrix_matches_reference_jacobi_computation():
    trajectories = [
        np.array([[0.0], [0.25], [0.5], [0.75]]),
        np.array([[1.0], [1.25], [1.5], [1.75]]),
    ]
    transformer = OKHSTransformer(kernel=DotKernel(), q=0.8, n_quad_points=5, dt=1.0)

    transformer.fit(trajectories)

    expected = np.array(
        [
            [_reference_gram_entry(transformer, trajectories[0], trajectories[0]),
             _reference_gram_entry(transformer, trajectories[0], trajectories[1])],
            [_reference_gram_entry(transformer, trajectories[1], trajectories[0]),
             _reference_gram_entry(transformer, trajectories[1], trajectories[1])],
        ]
    )
    actual = transformer.gram_matrix_ - transformer.regularization_policy.base_jitter * np.eye(len(trajectories))

    assert np.allclose(actual, expected)


def test_gram_computation_uses_batch_kernel_when_available():
    trajectories = [
        np.array([[0.0], [0.5], [1.0]]),
        np.array([[1.0], [1.5], [2.0]]),
    ]
    kernel = BatchDotKernel()
    transformer = OKHSTransformer(kernel=kernel, q=0.8, n_quad_points=4, dt=1.0)

    transformer.fit(trajectories)

    assert kernel.batch_calls > 0


def test_blockwise_gram_computation_splits_batch_work():
    trajectories = [
        np.array([[0.0], [0.5], [1.0]]),
        np.array([[1.0], [1.5], [2.0]]),
        np.array([[2.0], [2.5], [3.0]]),
    ]
    kernel = BatchDotKernel()
    transformer = OKHSTransformer(kernel=kernel, q=0.8, n_quad_points=4, dt=1.0, pairwise_block_size=1)

    transformer.fit(trajectories)

    assert kernel.batch_calls > 3


def test_liouville_matrix_matches_reference_jacobi_computation():
    trajectories = [
        np.array([[0.0], [0.25], [0.5], [0.75]]),
        np.array([[1.0], [1.25], [1.5], [1.75]]),
    ]
    transformer = OKHSTransformer(kernel=DotKernel(), q=0.8, n_quad_points=5, dt=1.0)
    transformer.fit(trajectories)
    operator = FractionalLiouvilleOperator(okhs_transformer=transformer, n_quad_points=5)

    operator.fit(trajectories)

    expected = np.array(
        [
            [_reference_liouville_entry(transformer, trajectories[0], trajectories[0]),
             _reference_liouville_entry(transformer, trajectories[0], trajectories[1])],
            [_reference_liouville_entry(transformer, trajectories[1], trajectories[0]),
             _reference_liouville_entry(transformer, trajectories[1], trajectories[1])],
        ]
    )

    assert np.allclose(operator.liouville_matrix_, expected)


def test_liouville_computation_uses_batch_kernel_when_available():
    trajectories = [
        np.array([[0.0], [0.5], [1.0]]),
        np.array([[1.0], [1.5], [2.0]]),
    ]
    kernel = BatchDotKernel()
    transformer = OKHSTransformer(kernel=kernel, q=0.8, n_quad_points=4, dt=1.0)
    transformer.fit(trajectories)
    kernel.batch_calls = 0

    operator = FractionalLiouvilleOperator(okhs_transformer=transformer, n_quad_points=4)
    operator.fit(trajectories)

    assert kernel.batch_calls > 0


def test_blockwise_liouville_computation_splits_batch_work():
    trajectories = [
        np.array([[0.0], [0.5], [1.0]]),
        np.array([[1.0], [1.5], [2.0]]),
        np.array([[2.0], [2.5], [3.0]]),
    ]
    kernel = BatchDotKernel()
    transformer = OKHSTransformer(kernel=kernel, q=0.8, n_quad_points=4, dt=1.0, pairwise_block_size=1)
    transformer.fit(trajectories)
    kernel.batch_calls = 0

    operator = FractionalLiouvilleOperator(okhs_transformer=transformer, n_quad_points=4, pairwise_block_size=1)
    operator.fit(trajectories)

    assert kernel.batch_calls > 6


def test_gram_progress_monitor_uses_tqdm(monkeypatch):
    trajectories = [
        np.array([[0.0], [0.5], [1.0]]),
        np.array([[1.0], [1.5], [2.0]]),
        np.array([[2.0], [2.5], [3.0]]),
    ]
    fake_bar = None
    writes = []

    def fake_factory(*args, **kwargs):
        nonlocal fake_bar
        fake_bar = FakeTqdmBar(*args, **kwargs)
        return fake_bar

    monkeypatch.setattr(okhs_module, "TQDM_FACTORY", fake_factory)
    monkeypatch.setattr(okhs_module, "TQDM_WRITE", lambda message: writes.append(message))

    transformer = OKHSTransformer(
        kernel=BatchDotKernel(),
        q=0.8,
        n_quad_points=4,
        dt=1.0,
        pairwise_block_size=1,
        show_progress=True,
    )

    transformer.fit(trajectories)

    assert fake_bar is not None
    assert fake_bar.total == 6
    assert fake_bar.updated == 6
    assert fake_bar.closed is True
    assert fake_bar.postfix_calls
    assert any("building gram matrix" in message for message in writes)
    assert any(
        "finished gram matrix" in message and "elapsed_s=" in message and "blk_s=" in message for message in writes)
    assert any(
        "elapsed_s" in payload and "blk_s" in payload and "eta_s" in payload for payload, _ in fake_bar.postfix_calls)


def test_liouville_progress_monitor_uses_tqdm(monkeypatch):
    trajectories = [
        np.array([[0.0], [0.5], [1.0]]),
        np.array([[1.0], [1.5], [2.0]]),
        np.array([[2.0], [2.5], [3.0]]),
    ]
    bars = []
    writes = []

    def fake_factory(*args, **kwargs):
        bar = FakeTqdmBar(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(okhs_module, "TQDM_FACTORY", fake_factory)
    monkeypatch.setattr(okhs_module, "TQDM_WRITE", lambda message: writes.append(message))

    transformer = OKHSTransformer(
        kernel=BatchDotKernel(),
        q=0.8,
        n_quad_points=4,
        dt=1.0,
        pairwise_block_size=1,
        show_progress=True,
    )
    transformer.fit(trajectories)

    operator = FractionalLiouvilleOperator(
        okhs_transformer=transformer,
        n_quad_points=4,
        pairwise_block_size=1,
    )
    operator.fit(trajectories)

    assert len(bars) == 2
    liouville_bar = bars[1]
    assert liouville_bar.total == 9
    assert liouville_bar.updated == 9
    assert liouville_bar.closed is True
    assert liouville_bar.postfix_calls
    assert any("building liouville matrix" in message for message in writes)
    assert any("finished liouville matrix" in message and "elapsed_s=" in message and "blk_s=" in message for message in
               writes)
    assert any("elapsed_s" in payload and "blk_s" in payload and "eta_s" in payload for payload, _ in
               liouville_bar.postfix_calls)


def test_transform_uses_pinv_when_condition_threshold_is_exceeded(monkeypatch):
    transformer = OKHSTransformer(
        kernel=DotKernel(),
        q=0.8,
        regularization_policy=RegularizationPolicy(condition_threshold=0.0, fallback_solver="pinv"),
    )
    transformer.train_trajectories_ = [np.array([[0.0], [1.0]]), np.array([[1.0], [2.0]])]
    transformer.gram_matrix_ = np.eye(2)
    transformer.gram_condition_number_ = 1.0

    monkeypatch.setattr(transformer, "_compute_gram_entry_jacobi", lambda left, right: 1.0)

    state = {"pinv_calls": 0}

    def fake_pinv(matrix):
        state["pinv_calls"] += 1
        return np.eye(matrix.shape[0])

    monkeypatch.setattr(np.linalg, "pinv", fake_pinv)
    monkeypatch.setattr(np.linalg, "solve",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("solve should not be used")))

    coordinates = transformer.transform([np.array([[2.0], [3.0]])])

    assert state["pinv_calls"] == 1
    assert coordinates.shape == (1, 2)


def test_liouville_shape_validation_rejects_mismatch():
    with pytest.raises(ValueError):
        validate_liouville_shapes(np.eye(2), np.ones((2, 3)))


def test_initial_coefficient_validation_fails_when_segment_is_too_short():
    with pytest.raises(ValueError):
        validate_initial_coefficient_feasibility(np.ones((2, 1)), n_modes=3)


def test_select_stable_modes_respects_threshold_override():
    eigenvalues = np.array([-0.5 + 0.0j, 0.1 + 0.0j, 0.4 + 0.0j])
    policy = StabilityPolicy(threshold=0.0)

    mask = select_stable_modes(eigenvalues, stability_policy=policy, stability_threshold=0.2)

    assert np.array_equal(mask, np.array([True, True, False]))


def test_sort_eigendecomposition_supports_real_desc():
    eigenvalues = np.array([1.0 + 1.0j, 3.0 + 0.0j, -2.0 + 0.5j])
    eigenvectors = np.eye(3)

    sorted_values, sorted_vectors = sort_eigendecomposition(eigenvalues, eigenvectors, "real_desc")

    assert np.array_equal(sorted_values, np.array([3.0 + 0.0j, 1.0 + 1.0j, -2.0 + 0.5j]))
    assert sorted_vectors.shape == (3, 3)


def test_predict_does_not_call_plotting(monkeypatch):
    liouville = SimpleNamespace(
        okhs=SimpleNamespace(q=0.8, dt=1.0),
        eigenvalues_=np.array([-0.2 + 0.0j]),
    )
    fdmd = FractionalDMD(liouville_operator=liouville, stability_policy=StabilityPolicy())
    fdmd.modes_ = np.array([[1.0]])

    monkeypatch.setattr(fdmd, "fit_initial_coefficients", lambda initial_trajectory, Xi=None, eig=None: np.array([1.0]))
    monkeypatch.setattr(
        okhs_module,
        "plot_forecast_diagnostics",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("predict should not plot")),
    )

    prediction = fdmd.predict(
        initial_trajectory=np.array([[1.0], [0.8]]),
        t_span=np.array([0.0, 1.0, 2.0]),
    )

    assert prediction.shape == (3, 1)


def test_prediction_state_caps_modes_to_available_equations():
    liouville = SimpleNamespace(
        okhs=SimpleNamespace(q=0.9, dt=1.0),
        eigenvalues_=np.array(
            [-0.01 + 0.0j, -0.02 + 0.0j, -0.03 + 0.0j, -0.04 + 0.0j, -0.05 + 0.0j, -0.06 + 0.0j]
        ),
    )
    fdmd = FractionalDMD(liouville_operator=liouville, stability_policy=StabilityPolicy())
    fdmd.modes_ = np.ones((6, 1), dtype=float)

    state = fdmd._prepare_prediction_state(
        initial_trajectory=np.arange(4, dtype=float).reshape(-1, 1),
        t_span=np.array([4.0, 5.0]),
        stability_threshold=0.0,
    )
    diagnostics = fdmd._build_prediction_diagnostics(state)

    assert diagnostics["available_equations"] == 4
    assert diagnostics["prediction_mode_cap"] == 4
    assert diagnostics["prediction_mode_cap_applied"] is True
    assert diagnostics["n_selected_prediction_modes"] == 4


def test_adaptive_prediction_mode_selector_respects_budget():
    liouville = SimpleNamespace(
        okhs=SimpleNamespace(q=0.9, dt=1.0),
        eigenvalues_=np.array(
            [
                -0.01 + 0.02j,
                -0.01 - 0.02j,
                0.015 + 0.0j,
                -0.02 + 0.01j,
                -0.02 - 0.01j,
                0.005 + 0.0j,
            ]
        ),
    )
    fdmd = FractionalDMD(liouville_operator=liouville, stability_policy=StabilityPolicy())
    fdmd.modes_ = np.array([[3.0], [3.0], [2.5], [1.5], [1.5], [0.5]], dtype=float)
    fdmd.fit_initial_coefficients = lambda initial_trajectory, Xi=None, eig=None: np.ones(len(eig), dtype=np.complex128)

    state = fdmd._prepare_prediction_state(
        initial_trajectory=np.arange(8, dtype=float).reshape(-1, 1),
        t_span=np.arange(8.0, 14.0),
        stability_threshold=0.03,
        prediction_mode_selection_policy="adaptive_tail_energy",
        max_prediction_modes=3,
        min_prediction_modes=2,
    )
    diagnostics = fdmd._build_prediction_diagnostics(state)

    assert diagnostics["prediction_mode_selection_policy"] == "adaptive_tail_energy"
    assert diagnostics["n_selected_prediction_modes"] <= 3
    assert diagnostics["prediction_preselection_count"] >= diagnostics["n_selected_prediction_modes"]
