import numpy as np
import pytest
import torch

from fedot_ind.core.operation.transformation.representation.statistical.stat_features import hurst_exponent
from fedot_ind.core.operation.transformation.torch_backend.statistical.stat_features import hurst_exponent_torch
from tests.unit.core.operation.transformation.representation.statistical.hurst_reference import (
    numpy_hurst, torch_hurst,
)


def signals(length, dtype):
    rng = np.random.default_rng(618)
    return [
        rng.normal(size=length).astype(dtype),
        np.linspace(-1, 1, length, dtype=dtype),
        np.sin(np.arange(length) * 0.3).astype(dtype),
        np.full(length, 1.1, dtype=dtype),
        (1e4 + rng.normal(size=length) * 0.01).astype(dtype),
        (rng.normal(size=length) * 1e-8).astype(dtype),
    ]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('length', [2, 5, 20, 28, 251, 512])
def test_numpy_hurst_preserves_reference(dtype, length):
    for signal in signals(length, dtype):
        before = signal.copy()
        np.testing.assert_allclose(hurst_exponent(signal), numpy_hurst(signal),
                                   rtol=1e-6, atol=1e-7, equal_nan=True)
        np.testing.assert_array_equal(signal, before)
    assert hurst_exponent(signal, axis=2) is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('length', [2, 5, 20, 28, 251, 512])
def test_torch_hurst_preserves_reference(dtype, length):
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    for signal in signals(length, np_dtype):
        x = torch.from_numpy(signal)
        before = x.clone()
        state = torch.random.get_rng_state().clone()
        observed = hurst_exponent_torch(x)
        expected = torch_hurst(x)
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7, equal_nan=True)
        assert isinstance(observed, float)
        assert torch.equal(x, before)
        assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize('shape,axis', [((3, 28), -1), ((28, 3), 0), ((2, 3, 28), -1)])
def test_torch_hurst_batch_shape_dtype_order_and_noncontiguous(shape, axis):
    x = torch.arange(np.prod(shape), dtype=torch.float64).reshape(shape).sin()
    x = x[..., ::2]
    actual = hurst_exponent_torch(x, axis)
    reference = torch_hurst(x, axis)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7, equal_nan=True)
    assert actual.dtype == reference.dtype
    assert actual.device == reference.device


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_hurst_nonfinite_behavior_preserved(value):
    x = np.arange(20, dtype=np.float64)
    x[8] = value
    np.testing.assert_allclose(hurst_exponent(x), numpy_hurst(x), equal_nan=True)
    tensor = torch.from_numpy(x)
    try:
        reference = torch_hurst(tensor)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            hurst_exponent_torch(tensor)
    else:
        np.testing.assert_allclose(hurst_exponent_torch(tensor), reference, equal_nan=True)
