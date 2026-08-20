"""Tests for structured Torch SPD batches and their producing transformers."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    BaseTorchSPDBuilder,
    DenseSPDBatch,
    RaggedBlockSPDBatch,
    TorchBlockCovariances,
    TorchCoSpectra,
    TorchCovariances,
    UniformBlockSPDBatch,
)


def _signals(
    n_samples: int = 4,
    n_channels: int = 6,
    n_times: int = 128,
    seed: int = 42,
) -> torch.Tensor:
    """Create a deterministic real-valued signal batch."""
    values = np.random.default_rng(seed).standard_normal((n_samples, n_channels, n_times))
    return torch.from_numpy(values)


def test_spd_builders_share_structured_output_contract():
    """Every Torch SPD builder must implement the common structured API."""
    assert issubclass(TorchCovariances, BaseTorchSPDBuilder)
    assert issubclass(TorchBlockCovariances, BaseTorchSPDBuilder)
    assert issubclass(TorchCoSpectra, BaseTorchSPDBuilder)


def test_covariances_transform_spd_returns_dense_spd_batch():
    """Ordinary covariance estimation must expose a dense structured batch."""
    signals = _signals(n_channels=3)
    estimator = TorchCovariances(estimator="scm")

    structured = estimator.fit(signals).transform_spd(signals)
    dense = estimator.transform(signals)

    assert isinstance(structured, DenseSPDBatch)
    assert structured.structure == "dense"
    assert structured.n_samples == signals.shape[0]
    assert structured.block_shapes == ((signals.shape[1], signals.shape[1]),)
    torch.testing.assert_close(structured.to_dense(), dense)


def test_block_covariances_transform_spd_preserves_ragged_blocks():
    """Block covariance estimation must not require dense block-diagonal assembly."""
    signals = _signals(n_channels=6)
    estimator = TorchBlockCovariances(block_size=[3, 2, 1], estimator="scm")

    structured = estimator.fit(signals).transform_spd(signals)
    dense = estimator.transform(signals)

    assert isinstance(structured, RaggedBlockSPDBatch)
    assert structured.structure == "ragged_blocks"
    assert structured.n_samples == signals.shape[0]
    assert structured.block_shapes == ((3, 3), (2, 2), (1, 1))
    assert tuple(block.shape for block in structured.matrices) == (
        (signals.shape[0], 3, 3),
        (signals.shape[0], 2, 2),
        (signals.shape[0], 1, 1),
    )
    torch.testing.assert_close(structured.to_dense(), dense)


def test_cospectra_transform_spd_preserves_uniform_frequency_blocks():
    """Co-spectral estimation must store frequencies as a uniform block axis."""
    signals = _signals(n_channels=3)
    estimator = TorchCoSpectra(window=32, overlap=0.5, fs=128.0)

    structured = estimator.fit(signals).transform_spd(signals)
    pyriemann_layout = estimator.transform(signals)

    assert isinstance(structured, UniformBlockSPDBatch)
    assert structured.structure == "uniform_blocks"
    assert structured.n_samples == signals.shape[0]
    assert structured.matrices.shape == (
        signals.shape[0],
        pyriemann_layout.shape[-1],
        signals.shape[1],
        signals.shape[1],
    )
    assert structured.block_shapes == tuple(
        [(signals.shape[1], signals.shape[1])] * pyriemann_layout.shape[-1]
    )
    torch.testing.assert_close(
        structured.matrices,
        pyriemann_layout.permute(0, 3, 1, 2),
    )


@pytest.mark.parametrize(
    ("batch_class", "matrices", "error"),
    [
        (DenseSPDBatch, torch.ones(2, 3, 2), "square"),
        (UniformBlockSPDBatch, torch.ones(2, 3, 2, 3), "square"),
        (RaggedBlockSPDBatch, (torch.ones(2, 3, 3), torch.ones(3, 2, 2)), "same number"),
    ],
)
def test_structured_spd_batches_validate_matrix_shapes(batch_class, matrices, error):
    """Structured batches must reject malformed or incompatible SPD tensors."""
    with pytest.raises(ValueError, match=error):
        batch_class(matrices)
