"""Parity tests for shrinkage of dense and structured Torch SPD batches."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyriemann.estimation import Shrinkage

from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    DenseSPDBatch,
    RaggedBlockSPDBatch,
    TorchShrinkage,
    UniformBlockSPDBatch,
)


def _spd_blocks(block_sizes: tuple[int, ...], n_samples: int = 4, seed: int = 42):
    """Create independent deterministic SPD blocks."""
    generator = np.random.default_rng(seed)
    blocks = []
    for size in block_sizes:
        values = generator.standard_normal((n_samples, size, size))
        blocks.append(values @ values.transpose(0, 2, 1) + np.eye(size) * 0.5)
    return blocks


@pytest.mark.parametrize("shrinkage", [0.0, 0.1, 0.7, 1.0])
def test_dense_shrinkage_matches_pyriemann_and_preserves_input(shrinkage):
    """Dense Torch shrinkage must reproduce PyRiemann without changing its input."""
    matrices = _spd_blocks((4,))[0]
    spd = DenseSPDBatch(torch.from_numpy(matrices))
    original = spd.matrices.clone()

    actual = TorchShrinkage(shrinkage=shrinkage).fit_transform(spd)
    expected = Shrinkage(shrinkage=shrinkage).fit_transform(matrices)

    assert isinstance(actual, DenseSPDBatch)
    torch.testing.assert_close(spd.matrices, original)
    np.testing.assert_allclose(actual.matrices.numpy(), expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("batch_type", ["uniform", "ragged"])
def test_structured_shrinkage_matches_full_block_diagonal_pyriemann(batch_type):
    """Global structured shrinkage must equal shrinking the dense block diagonal matrix."""
    blocks = _spd_blocks((3, 2, 1))
    if batch_type == "uniform":
        equal_blocks = _spd_blocks((2, 2, 2))
        spd = UniformBlockSPDBatch(torch.from_numpy(np.stack(equal_blocks, axis=1)))
    else:
        spd = RaggedBlockSPDBatch(tuple(torch.from_numpy(block) for block in blocks))

    actual = TorchShrinkage(shrinkage=0.1).fit_transform(spd)
    expected = Shrinkage(shrinkage=0.1).fit_transform(spd.to_dense().numpy())

    assert type(actual) is type(spd)
    np.testing.assert_allclose(actual.to_dense().numpy(), expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("shrinkage", [-0.1, 1.1])
def test_shrinkage_rejects_coefficient_outside_unit_interval(shrinkage):
    """Shrinkage coefficient must define a convex combination."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        TorchShrinkage(shrinkage=shrinkage)
