"""Parity tests for centroid estimation and tangent projection of SPD blocks."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space as pyriemann_tangent_space

from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    RaggedBlockSPDBatch,
    UniformBlockSPDBatch,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_tangent_space import (
    TorchSPDCentroid,
    TorchTangentSpace,
)


def _spd_blocks(block_sizes: tuple[int, ...], n_samples: int = 5, seed: int = 42):
    """Create independent deterministic SPD blocks for each sample."""
    generator = np.random.default_rng(seed)
    blocks = []
    for size in block_sizes:
        values = generator.standard_normal((n_samples, size, size))
        blocks.append(values @ values.transpose(0, 2, 1) + np.eye(size) * 0.5)
    return blocks


def _expected_compact_projection(blocks, metric: str) -> np.ndarray:
    """Build PyRiemann tangent features by independently projecting every block."""
    return np.concatenate(
        [
            pyriemann_tangent_space(
                block,
                mean_covariance(block, metric=metric),
                metric=metric,
            )
            for block in blocks
        ],
        axis=1,
    )


@pytest.mark.parametrize("metric", ["riemann", "logeuclid", "euclid"])
def test_ragged_block_centroid_and_projection_match_independent_pyriemann_blocks(metric):
    """Ragged blocks must have the same independent means and tangent features."""
    blocks = _spd_blocks((3, 2, 1))
    spd = RaggedBlockSPDBatch(tuple(torch.from_numpy(block) for block in blocks))

    centroid = TorchSPDCentroid(metric=metric).fit(spd).centroid_
    actual = TorchTangentSpace(centroid=centroid, metric=metric).transform(spd)
    expected = _expected_compact_projection(blocks, metric)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)
    expected_centroid = mean_covariance(spd.to_dense().numpy(), metric=metric)
    np.testing.assert_allclose(centroid.to_dense().numpy(), expected_centroid, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("metric", ["riemann", "logeuclid", "euclid"])
def test_uniform_block_centroid_and_projection_match_independent_pyriemann_blocks(metric):
    """Uniform blocks must be processed as one batched block axis without changing results."""
    blocks = _spd_blocks((2, 2, 2))
    spd = UniformBlockSPDBatch(torch.from_numpy(np.stack(blocks, axis=1)))

    centroid = TorchSPDCentroid(metric=metric).fit(spd).centroid_
    actual = TorchTangentSpace(centroid=centroid, metric=metric).transform(spd)
    expected = _expected_compact_projection(blocks, metric)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-7, atol=1e-8)


def test_tangent_space_rejects_different_block_layout_than_its_centroid():
    """Projection must reject an SPD batch with incompatible block dimensions."""
    centroid_blocks = _spd_blocks((3, 2))
    input_blocks = _spd_blocks((2, 3), seed=7)
    centroid = TorchSPDCentroid().fit(
        RaggedBlockSPDBatch(tuple(torch.from_numpy(block) for block in centroid_blocks))
    ).centroid_

    with pytest.raises(ValueError, match="block"):
        TorchTangentSpace(centroid=centroid).transform(
            RaggedBlockSPDBatch(tuple(torch.from_numpy(block) for block in input_blocks))
        )
