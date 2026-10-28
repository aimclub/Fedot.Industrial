"""Reference and metamorphic checks for bounded-memory image kernels."""
import math

import pytest
import torch
from hypothesis import given, settings, strategies as st

from fedot_ind.core.operation.transformation.torch_backend.image.discretize import (
    _digitize_per_sample_bins_torch,
)
from fedot_ind.core.operation.transformation.torch_backend.image.mtf_transformation import (
    MTF, _aggregate_transition_field,
)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


def _block_reference(binned, transition, size, overlapping):
    """Direct float64 block means without integral images or histograms."""
    batch, length = binned.shape
    window = math.ceil(length / size)
    if overlapping and length % size:
        offsets = torch.linspace(0, size * window - length, size).long().tolist()
        starts = [i * window - offsets[i] for i in range(size)]
        ends = [s + window for s in starts]
    else:
        bounds = torch.linspace(0, length, size + 1).long().tolist()
        starts, ends = bounds[:-1], bounds[1:]
    result = torch.empty((batch, size, size), dtype=torch.float64)
    for b in range(batch):
        field = transition[b].double()[binned[b, :, None], binned[b, None, :]]
        for r, (r0, r1) in enumerate(zip(starts, ends)):
            for c, (c0, c1) in enumerate(zip(starts, ends)):
                result[b, r, c] = field[r0:r1, c0:c1].mean()
    return result


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length,size,bins", [(31, 7, 4), (32, 8, 4), (17, 17, 3), (7, 3, 9), (9, 1, 2)])
@pytest.mark.parametrize("overlapping", [False, True])
def test_mtf_block_means_match_independent_reference(device, length, size, bins, overlapping):
    gen = torch.Generator().manual_seed(734)
    binned = torch.randint(bins, (2, length), generator=gen)
    matrix = torch.rand((2, bins, bins), generator=gen)
    matrix[:, 0] = 0  # Unvisited states remain zero, not NaN.
    expected = _block_reference(binned, matrix, size, overlapping)
    actual = _aggregate_transition_field(binned.to(device), matrix.to(device), size, overlapping)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("strategy", ["quantile", "uniform", "normal"])
@pytest.mark.parametrize("flatten", [False, True])
def test_mtf_public_batch_and_device_parity(device, strategy, flatten):
    gen = torch.Generator().manual_seed(18)
    x = torch.randn((3, 2, 63), generator=gen)
    before = x.clone()
    params = {"image_size": 13, "n_bins": 8, "strategy": strategy,
              "overlapping": True, "flatten": flatten, "return_init_dim": not flatten}
    expected = MTF({**params, "torch_device": "cpu"}).transform(x)
    actual = MTF({**params, "torch_device": device}).transform(x)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert actual.device.type == device
    assert actual.dtype == torch.float32
    assert actual.shape == ((6, 169) if flatten else (3, 2, 13, 13))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", [0, 7, 8, 63])
def test_digitize_nan_inf_ties_unordered_and_noncontiguous(device, width):
    values = torch.tensor([[float("nan"), -float("inf"), -1., 0., 1., float("inf")]], device=device)
    edges = torch.tensor([[float("nan"), 1., -float("inf"), 0., 1., float("inf"), -1.]], device=device)
    edges = edges.repeat(1, math.ceil(width / 7))[:, :width]
    x = values.repeat(2, 2)[:, ::2]
    edges = edges.repeat(2, 1)
    expected = (edges[:, None, :] < x[:, :, None]).sum(-1)
    before = edges.clone()
    actual = _digitize_per_sample_bins_torch(x, edges)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(edges, before, equal_nan=True)
    assert actual.dtype == torch.int64


@given(batch=st.integers(1, 4), length=st.integers(1, 35), bins=st.integers(0, 25))
@settings(max_examples=25, deadline=None)
def test_digitize_counts_and_batch_partition_law(batch, length, bins):
    gen = torch.Generator().manual_seed(batch * 101 + length * 7 + bins)
    x = torch.randn((batch, length), generator=gen)
    edges = torch.randint(-2, 3, (batch, bins), generator=gen)
    expected = (edges[:, None, :] < x[:, :, None]).sum(-1)
    actual = _digitize_per_sample_bins_torch(x, edges)
    partitioned = torch.cat([_digitize_per_sample_bins_torch(row[None], edge[None])
                             for row, edge in zip(x, edges)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, partitioned, rtol=0, atol=0)


def test_histogram_aggregation_preserves_transition_gradient():
    binned = torch.tensor([[0, 1, 0, 2, 2, 1, 0]])
    matrix = torch.arange(9, dtype=torch.float64).reshape(1, 3, 3).requires_grad_()
    output = _aggregate_transition_field(binned, matrix, 3)
    output.sum().backward()
    actual_grad = matrix.grad.clone()
    reference_matrix = matrix.detach().clone().requires_grad_()
    field = reference_matrix[0][binned[0, :, None], binned[0, None, :]]
    bounds = [0, 2, 4, 7]
    reference = sum(field[a:b, c:d].mean()
                    for a, b in zip(bounds, bounds[1:]) for c, d in zip(bounds, bounds[1:]))
    reference.backward()
    torch.testing.assert_close(actual_grad, reference_matrix.grad, rtol=1e-12, atol=1e-12)
