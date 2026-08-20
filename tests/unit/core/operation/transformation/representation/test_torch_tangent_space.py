"""Tests for separate Torch SPD centroid estimation and tangent projection."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space as pyriemann_tangent_space
from pyriemann.tangentspace import TangentSpace as PyRiemannTangentSpace

from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    DenseSPDBatch,
    DenseSPDReference,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_tangent_space import (
    TorchSPDCentroid,
    TorchTangentSpace,
)


def _spd_batch(n_samples: int = 7, n_channels: int = 4, seed: int = 42) -> np.ndarray:
    """Create a deterministic batch of symmetric positive-definite matrices."""
    generator = np.random.default_rng(seed)
    values = generator.standard_normal((n_samples, n_channels, n_channels))
    return values @ values.transpose(0, 2, 1) + np.eye(n_channels) * 0.5


@pytest.mark.parametrize("metric", ["riemann", "logeuclid", "euclid"])
def test_spd_centroid_matches_pyriemann(metric):
    """TorchSPDCentroid must estimate the same weighted reference as PyRiemann."""
    matrices = _spd_batch()
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0])
    estimator = TorchSPDCentroid(metric=metric)

    result = estimator.fit(
        DenseSPDBatch(torch.from_numpy(matrices)),
        y=np.arange(matrices.shape[0]),
        sample_weight=torch.from_numpy(weights),
    )
    expected = mean_covariance(matrices, metric=metric, sample_weight=weights)

    assert result is estimator
    assert isinstance(estimator.centroid_, DenseSPDReference)
    assert estimator.centroid_.matrices.device.type == "cpu"
    np.testing.assert_allclose(estimator.centroid_.matrices.numpy(), expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("metric", ["riemann", "logeuclid", "euclid"])
def test_tangent_space_projects_with_precomputed_centroid(metric):
    """Projection must match PyRiemann for an explicitly supplied centroid."""
    matrices = _spd_batch()
    centroid = mean_covariance(matrices, metric=metric)
    projector = TorchTangentSpace(
        centroid=DenseSPDReference(torch.from_numpy(centroid)), metric=metric
    )

    actual = projector.transform(DenseSPDBatch(torch.from_numpy(matrices)))
    expected = pyriemann_tangent_space(matrices, centroid, metric=metric)

    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("metric", ["riemann", "logeuclid", "euclid"])
def test_centroid_estimation_and_projection_match_pyriemann(metric):
    """The complete Torch centroid-to-projection pipeline must match PyRiemann."""
    matrices = _spd_batch()
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0])
    torch_matrices = torch.from_numpy(matrices)

    centroid = TorchSPDCentroid(metric=metric).fit(
        DenseSPDBatch(torch_matrices), sample_weight=torch.from_numpy(weights)
    ).centroid_
    actual = TorchTangentSpace(centroid=centroid, metric=metric).transform(
        DenseSPDBatch(torch_matrices)
    )

    expected = PyRiemannTangentSpace(metric=metric).fit_transform(
        matrices, sample_weight=weights
    )

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize(
    ("metric", "cache_name", "empty_cache_name"),
    [
        ("riemann", "centroid_invsqrt_", "centroid_log_"),
        ("logeuclid", "centroid_log_", "centroid_invsqrt_"),
    ],
)
def test_tangent_space_caches_only_operator_required_by_projection(
    metric, cache_name, empty_cache_name
):
    """The projector must prepare reusable spectral operators once per centroid."""
    centroid = DenseSPDReference(torch.from_numpy(_spd_batch(n_samples=1)[0]))
    projector = TorchTangentSpace(centroid=centroid, metric=metric)

    assert getattr(projector, cache_name) is not None
    assert getattr(projector, empty_cache_name) is None


def test_tangent_space_rejects_incompatible_matrix_size():
    """Projection must reject SPD matrices incompatible with the supplied centroid."""
    centroid = DenseSPDReference(torch.from_numpy(_spd_batch(n_channels=4, seed=1)[0]))
    projector = TorchTangentSpace(centroid=centroid, metric="riemann")

    with pytest.raises(ValueError, match="block layout"):
        projector.transform(DenseSPDBatch(torch.from_numpy(_spd_batch(n_channels=3))))


def test_tangent_space_rejects_metric_mapping():
    """The refactored projector must expose one metric rather than a metric mapping."""
    centroid = DenseSPDReference(torch.from_numpy(_spd_batch(n_samples=1)[0]))

    with pytest.raises(TypeError, match="metric"):
        TorchTangentSpace(centroid=centroid, metric={"mean": "riemann", "map": "riemann"})
