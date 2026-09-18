"""Parity tests for Torch MDM distances and class-wise SPD centroids."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyriemann.utils.distance import distance

from fedot_ind.core.operation.transformation.representation.manifold.torch_mdm import (
    TorchClassCentroids,
    TorchMDMDistances,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    DenseSPDBatch,
    RaggedBlockSPDBatch,
    UniformBlockSPDBatch,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_tangent_space import (
    TorchSPDCentroid,
)


def _spd_batch(n_samples: int = 6, n_channels: int = 3, seed: int = 42) -> np.ndarray:
    """Create a deterministic batch of real SPD matrices."""
    generator = np.random.default_rng(seed)
    values = generator.standard_normal((n_samples, n_channels, n_channels))
    return values @ values.transpose(0, 2, 1) + np.eye(n_channels) * 0.5


def _structured_batch(structure: str):
    """Create dense, uniform-block, or ragged-block SPD data."""
    if structure == "dense":
        return DenseSPDBatch(torch.from_numpy(_spd_batch()))
    if structure == "uniform":
        blocks = [_spd_batch(n_channels=2, seed=seed) for seed in (1, 2, 3)]
        return UniformBlockSPDBatch(torch.from_numpy(np.stack(blocks, axis=1)))
    return RaggedBlockSPDBatch(
        (torch.from_numpy(_spd_batch(n_channels=2, seed=1)), torch.from_numpy(_spd_batch(n_channels=3, seed=2)))
    )


@pytest.mark.parametrize("metric", ["riemann", "logeuclid", "euclid"])
@pytest.mark.parametrize("structure", ["dense", "uniform", "ragged"])
def test_mdm_distances_match_pyriemann_for_all_supported_structures(metric, structure):
    """Structured Torch distances must equal PyRiemann distances on dense equivalents."""
    spd = _structured_batch(structure)
    first_centroid = TorchSPDCentroid(metric=metric).fit(spd).centroid_
    second_centroid = TorchSPDCentroid(metric=metric).fit(
        _structured_batch(structure)
    ).centroid_

    actual = TorchMDMDistances(
        centroids=(first_centroid, second_centroid), metric=metric
    ).transform(spd)
    expected = np.column_stack(
        [
            distance(spd.to_dense().numpy(), centroid.to_dense().numpy(), metric=metric)
            for centroid in (first_centroid, second_centroid)
        ]
    )

    assert actual.shape == (spd.n_samples, 2)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)


def test_class_centroids_group_labels_and_produce_mdm_features_in_class_order():
    """Class-wise centroids must retain sorted labels and yield one distance per class."""
    matrices = _spd_batch(n_samples=6)
    labels = np.array([2, 1, 2, 1, 3, 3])
    spd = DenseSPDBatch(torch.from_numpy(matrices))

    class_centroids = TorchClassCentroids(metric="riemann").fit(spd, labels)
    actual = TorchMDMDistances(class_centroids.centroids_, metric="riemann").transform(spd)

    assert np.array_equal(class_centroids.classes_, np.array([1, 2, 3]))
    expected = np.column_stack(
        [
            distance(matrices, centroid.to_dense().numpy(), metric="riemann")
            for centroid in class_centroids.centroids_
        ]
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)


def test_class_centroids_forward_product_block_weights():
    """Every class median must use the configured product-manifold weights."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    spd = _structured_batch("ragged")
    block_weights = [0.5, 2.0]

    actual = TorchClassCentroids(
        metric="euclid", centroid_type="median", median_tol=1e-8, median_max_iter=200
    ).fit(spd, labels, block_weights=block_weights)

    for class_index, label in enumerate(actual.classes_):
        indices = torch.as_tensor(np.flatnonzero(labels == label))
        expected = TorchSPDCentroid(
            metric="euclid", centroid_type="median", median_tol=1e-8, median_max_iter=200
        ).fit(
            RaggedBlockSPDBatch(tuple(block[indices] for block in spd.matrices)),
            block_weights=block_weights,
        ).centroid_
        torch.testing.assert_close(
            actual.centroids_[class_index].to_dense(), expected.to_dense()
        )
