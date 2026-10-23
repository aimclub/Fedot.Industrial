"""OLS oracle and repeatability laws for the finite Hurst regression."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from fedot_ind.core.operation.transformation.torch_backend.statistical.stat_features import (
    _hurst_regression_slope, hurst_exponent_torch,
)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('length', [0, 1, 2, 19, 27, 250])
def test_slope_matches_independent_svd_minimum_norm_oracle(dtype, length):
    n = torch.arange(2, length + 2, dtype=dtype).log()
    rhs = torch.from_numpy(np.random.default_rng(17).normal(size=(5, length))).to(dtype)
    a = np.column_stack((n.double().numpy(), np.ones(length)))
    expected = np.linalg.lstsq(a, rhs.double().numpy().T, rcond=None)[0][0]
    observed = _hurst_regression_slope(n, rhs)
    rtol, atol = (2e-12, 2e-13) if dtype == torch.float64 else (1e-6, 1e-7)
    np.testing.assert_allclose(observed.numpy(), expected, rtol=rtol, atol=atol)
    assert observed.dtype == dtype and observed.device == rhs.device


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_identical_solver_inputs_repeat_exactly_without_rng_or_lstsq(monkeypatch, dtype):
    n = torch.arange(2, 29, dtype=dtype).log()
    rhs = torch.arange(135, dtype=dtype).sin().reshape(5, 27)
    before_n, before_rhs = n.clone(), rhs.clone()
    state = torch.random.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError('finite Hurst must not call the unstable general solver')

    monkeypatch.setattr(torch.linalg, 'lstsq', forbidden)
    expected = _hurst_regression_slope(n, rhs)
    for _ in range(20):
        assert torch.equal(_hurst_regression_slope(n, rhs), expected)
    assert torch.equal(n, before_n) and torch.equal(rhs, before_rhs)
    assert torch.equal(torch.random.get_rng_state(), state)


@given(st.lists(st.floats(-30, 30, allow_nan=False, allow_infinity=False, width=32), min_size=2, max_size=40))
@settings(max_examples=40, deadline=None)
def test_finite_slope_is_invariant_to_intercept_and_affine_in_response(values):
    n = torch.arange(2, len(values) + 2, dtype=torch.float64).log()
    y = torch.tensor([values], dtype=torch.float64)
    slope = _hurst_regression_slope(n, y)
    shifted = _hurst_regression_slope(n, 2 * y + 7)
    torch.testing.assert_close(shifted, 2 * slope, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('length', [1, 4, 19, 27, 250])
def test_batch_partition_and_permutation_preserve_float32_slopes_exactly(length):
    n = torch.arange(2, length + 2, dtype=torch.float32).log()
    y = torch.from_numpy(np.random.default_rng(12).normal(size=(135, length * 2)).astype(np.float32))[:, ::2]
    expected = _hurst_regression_slope(n, y)
    parts = [_hurst_regression_slope(n, part) for part in (y[:1], y[1:128], y[128:])]
    assert torch.equal(torch.cat(parts), expected)
    permutation = torch.arange(134, -1, -1)
    assert torch.equal(_hurst_regression_slope(n, y[permutation]), expected[permutation])


@pytest.mark.parametrize('length', [2, 5, 20, 28, 251, 512])
def test_public_hurst_repeats_and_preserves_float32_batch_values(length):
    values = torch.from_numpy(np.random.default_rng(61).normal(size=(8, length)).astype(np.float32))
    actual = hurst_exponent_torch(values)
    rowwise = torch.tensor([hurst_exponent_torch(row) for row in values])
    assert torch.equal(actual, rowwise)
    for _ in range(4):
        assert torch.equal(hurst_exponent_torch(values), actual)


def test_nonfinite_response_still_uses_legacy_failure_path(monkeypatch):
    def expected_failure(*args, **kwargs):
        raise RuntimeError('legacy nonfinite failure')

    monkeypatch.setattr(torch.linalg, 'lstsq', expected_failure)
    with pytest.raises(RuntimeError, match='legacy nonfinite failure'):
        _hurst_regression_slope(torch.tensor([1., 2.]), torch.tensor([[float('nan'), 1.]]))


def test_mixed_rows_only_send_nonfinite_values_to_legacy_solver(monkeypatch):
    from types import SimpleNamespace
    n = torch.tensor([1., 2.], dtype=torch.float64)
    y = torch.tensor([[2., 3.], [float('nan'), 1.], [4., 8.],
                      [float('inf'), 2.], [-float('inf'), 3.]], dtype=torch.float64)
    seen = []

    def legacy(design, rhs):
        assert design.dtype == rhs.dtype == torch.float64
        torch.testing.assert_close(design, torch.stack((n, torch.ones_like(n)), dim=1))
        seen.append(rhs[:, 0].clone())
        return SimpleNamespace(solution=torch.tensor([[42.], [0.]], dtype=torch.float64))

    monkeypatch.setattr(torch.linalg, 'lstsq', legacy)
    actual = _hurst_regression_slope(n, y)
    torch.testing.assert_close(actual, torch.tensor([1., 42., 4., 42., 42.], dtype=torch.float64))
    torch.testing.assert_close(torch.stack(seen), y[[1, 3, 4]], equal_nan=True)


@pytest.mark.parametrize('length', [1, 2])
def test_public_short_signal_and_multichannel_shape(length):
    values = torch.arange(1, length + 1, dtype=torch.float64)
    assert isinstance(hurst_exponent_torch(values), float)
    multi = torch.stack((values, values + 1)).reshape(1, 2, length)
    actual = hurst_exponent_torch(multi)
    assert actual.shape == (1, 2)
