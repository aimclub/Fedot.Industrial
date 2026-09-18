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


def test_weighted_product_median_matches_scaled_euclidean_reference():
    """Block weights must define the metric of one coupled product median."""
    points = np.array([
        [1.0, 1.0],
        [2.0, 4.0],
        [4.0, 2.0],
        [6.0, 6.0],
        [3.0, 5.0],
        [7.0, 3.0],
    ])
    block_weights = np.array([4.0, 1.0])
    spd = RaggedBlockSPDBatch(tuple(
        torch.from_numpy(points[:, index, None, None])
        for index in range(points.shape[1])
    ))

    actual = TorchSPDCentroid(
        metric="euclid", centroid_type="median", median_tol=1e-8, median_max_iter=200
    ).fit(spd, block_weights=block_weights).centroid_.matrices

    scaled = np.stack([
        np.diag(np.sqrt(block_weights) * point)
        for point in points
    ])
    expected_scaled = median_euclid(scaled, tol=1e-8, maxiter=200)
    expected = np.diag(expected_scaled) / np.sqrt(block_weights)

    np.testing.assert_allclose(
        np.array([block.item() for block in actual]), expected, rtol=1e-7, atol=1e-8
    )


def test_product_mean_coordinates_do_not_depend_on_block_weights():
    """Positive product weights must not change separable Frechet means."""
    blocks = tuple(
        torch.from_numpy(_spd_batch(n_channels=size, seed=seed))
        for size, seed in ((2, 10), (3, 11))
    )
    spd = RaggedBlockSPDBatch(blocks)

    unweighted = TorchSPDCentroid(metric="riemann").fit(spd).centroid_.matrices
    weighted = TorchSPDCentroid(metric="riemann").fit(
        spd, block_weights=[0.25, 3.0]
    ).centroid_.matrices

    for actual, expected in zip(weighted, unweighted):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("block_weights", "error_type", "match"),
    [
        ([1.0], ValueError, "one value for every SPD block"),
        ([1.0, 0.0], ValueError, "strictly positive"),
        ([1.0, -1.0], ValueError, "strictly positive"),
        ([1.0, np.nan], ValueError, "finite"),
        ("invalid", TypeError, "one-dimensional array-like"),
    ],
)
def test_product_centroid_rejects_invalid_block_weights(block_weights, error_type, match):
    """Product metric weights must be finite, positive, and layout-compatible."""
    blocks = tuple(
        torch.from_numpy(_spd_batch(n_channels=2, seed=seed))
        for seed in (20, 21)
    )

    with pytest.raises(error_type, match=match):
        TorchSPDCentroid().fit(
            RaggedBlockSPDBatch(blocks), block_weights=block_weights
        )
