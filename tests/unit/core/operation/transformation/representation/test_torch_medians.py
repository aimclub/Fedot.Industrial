"""Parity tests for Torch geometric SPD medians supported by PyRiemann."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyriemann.utils.median import median_euclid, median_riemann

from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    DenseSPDBatch,
    RaggedBlockSPDBatch,
    UniformBlockSPDBatch,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_tangent_space import (
    TorchSPDCentroid,
)


def _spd_batch(n_samples: int = 6, n_channels: int = 3, seed: int = 42) -> np.ndarray:
    """Create a deterministic real SPD batch without duplicate matrices."""
    generator = np.random.default_rng(seed)
    values = generator.standard_normal((n_samples, n_channels, n_channels))
    return values @ values.transpose(0, 2, 1) + np.eye(n_channels) * 0.5


@pytest.mark.parametrize(
    ("metric", "pyriemann_function"),
    [("euclid", median_euclid), ("riemann", median_riemann)],
)
def test_torch_geometric_median_matches_pyriemann(metric, pyriemann_function):
    """Torch medians must reproduce PyRiemann's weighted geometric medians."""
    matrices = _spd_batch()
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0])
    spd = DenseSPDBatch(torch.from_numpy(matrices))

    actual = TorchSPDCentroid(
        metric=metric,
        centroid_type="median",
        median_tol=1e-5,
        median_max_iter=50,
    ).fit(spd, sample_weight=torch.from_numpy(weights)).centroid_.matrices
    expected = pyriemann_function(matrices, weights=weights, tol=1e-5, maxiter=50)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize(
    ("metric", "pyriemann_function"),
    [("euclid", median_euclid), ("riemann", median_riemann)],
)
@pytest.mark.parametrize("structure", ["uniform", "ragged"])
def test_structured_geometric_median_matches_full_block_diagonal_pyriemann(
    metric, pyriemann_function, structure
):
    """Block medians must retain the objective of the full block-diagonal problem."""
    blocks = [_spd_batch(n_channels=2, seed=seed) for seed in (1, 2, 3)]
    if structure == "uniform":
        spd = UniformBlockSPDBatch(torch.from_numpy(np.stack(blocks, axis=1)))
    else:
        spd = RaggedBlockSPDBatch((torch.from_numpy(blocks[0]), torch.from_numpy(_spd_batch(n_channels=3, seed=4))))

    actual = TorchSPDCentroid(
        metric=metric, centroid_type="median", median_tol=1e-5, median_max_iter=50
    ).fit(spd).centroid_.to_dense()
    expected = pyriemann_function(spd.to_dense().numpy(), tol=1e-5, maxiter=50)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)
