import numpy as np
import pytest

from fedot_ind.core.kernel_learning import KernelMatrixBuilder, NystromApproximationPolicy


def test_kernel_matrix_builder_returns_psd_train_and_cross_kernel():
    train = np.array([[0.0], [1.0], [2.0]])
    test = np.array([[1.5], [3.0]])

    builder = KernelMatrixBuilder(kernel="rbf", normalize="trace", psd_correction="clip")
    bundle = builder.fit_transform(train, name="rbf")
    cross = builder.transform(test)

    assert bundle.train_kernel.shape == (3, 3)
    assert cross.shape == (2, 3)
    assert np.allclose(bundle.train_kernel, bundle.train_kernel.T)
    assert np.isclose(np.trace(bundle.train_kernel), 3.0)
    assert bundle.is_psd is True
    assert bundle.diagnostics["min_eigenvalue"] >= -1e-8


def test_kernel_matrix_builder_clips_non_psd_matrix():
    builder = KernelMatrixBuilder(psd_correction="clip")
    corrected, diagnostics = builder._validate_and_correct_psd(
        np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        )
    )

    assert diagnostics["psd_correction"] == "clip"
    assert diagnostics["is_psd"] is True
    assert np.min(np.linalg.eigvalsh(corrected)) >= -1e-8


def test_kernel_matrix_builder_rejects_distance_like_train_kernel():
    builder = KernelMatrixBuilder()
    builder.train_features_ = np.zeros((3, 1))

    with pytest.raises(ValueError, match="looks like a distance matrix"):
        builder._validate_train_kernel(
            np.array(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
            stage="raw",
        )


def test_kernel_matrix_builder_supports_nystrom_train_and_cross_shapes():
    train = np.array([[0.0], [1.0], [2.0], [3.0]])
    test = np.array([[1.5], [2.5]])

    builder = KernelMatrixBuilder(
        kernel="rbf",
        approximation="nystrom",
        nystrom_components=2,
        normalize="trace",
    )
    bundle = builder.fit_transform(train, name="nystrom_rbf")
    cross = builder.transform(test)

    assert bundle.train_kernel.shape == (4, 4)
    assert cross.shape == (2, 4)
    assert bundle.diagnostics["approximation"] == "nystrom"
    assert bundle.diagnostics["n_components"] == 2


def test_nystrom_policy_owns_default_component_heuristic():
    policy = NystromApproximationPolicy(default_max_components=3)

    assert policy.resolve_n_components(10) == 3
    assert policy.resolve_n_components(2) == 2

    with pytest.raises(ValueError, match="At least one sample"):
        policy.resolve_n_components(0)
