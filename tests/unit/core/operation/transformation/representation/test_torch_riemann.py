import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyriemann.utils.base import expm, invsqrtm, logm, sqrtm
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space as pyriemann_tangent_space
from pyriemann.utils.tangentspace import upper as pyriemann_upper

from fedot_ind.core.operation.transformation.representation.manifold.torch_tangent_space import (
    log_map_riemann,
    matrix_exp,
    matrix_invsqrt,
    matrix_log,
    matrix_sqrt,
    mean_euclid,
    mean_logeuclid,
    mean_riemann,
    tangent_space,
    upper,
)


def _spd_batch(n_matrices: int = 7, n_channels: int = 4, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n_matrices, n_channels, n_channels))
    return values @ values.transpose(0, 2, 1) + np.eye(n_channels) * 0.5


@pytest.mark.parametrize(
    ("torch_function", "pyriemann_function"),
    [
        (matrix_log, logm),
        (matrix_exp, expm),
        (matrix_sqrt, sqrtm),
        (matrix_invsqrt, invsqrtm),
    ],
)
def test_spectral_functions_match_pyriemann(torch_function, pyriemann_function):
    matrices = _spd_batch()
    actual = torch_function(torch.from_numpy(matrices)).numpy()
    expected = pyriemann_function(matrices)

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)


def test_upper_matches_pyriemann_order_and_weights():
    matrices = _spd_batch()
    actual = upper(torch.from_numpy(matrices)).numpy()
    expected = pyriemann_upper(matrices)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("metric", ["riemann", "logeuclid", "euclid"])
def test_tangent_space_matches_pyriemann(metric):
    matrices = _spd_batch()
    reference = mean_covariance(matrices, metric=metric)

    actual = tangent_space(
        torch.from_numpy(matrices), torch.from_numpy(reference), metric=metric
    ).numpy()
    expected = pyriemann_tangent_space(matrices, reference, metric=metric)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


def test_riemann_tangent_uses_cached_inverse_square_root():
    matrices = torch.from_numpy(_spd_batch())
    reference = mean_riemann(matrices)
    cached_invsqrt = matrix_invsqrt(reference)

    cached = log_map_riemann(matrices, reference, reference_invsqrt=cached_invsqrt)
    uncached = log_map_riemann(matrices, reference)

    torch.testing.assert_close(cached, uncached, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("torch_function", "metric"),
    [
        (mean_euclid, "euclid"),
        (mean_logeuclid, "logeuclid"),
        (mean_riemann, "riemann"),
    ],
)
def test_means_match_pyriemann(torch_function, metric):
    matrices = _spd_batch(n_matrices=5, n_channels=3)
    weights = np.array([1.0, 2.0, 3.0, 1.0, 4.0])

    actual = torch_function(
        torch.from_numpy(matrices), sample_weight=torch.from_numpy(weights)
    ).numpy()
    expected = mean_covariance(matrices, metric=metric, sample_weight=weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
