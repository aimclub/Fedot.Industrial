"""Batched NumPy Hurst must match the independent scalar implementation exactly."""

from types import SimpleNamespace

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.representation.statistical.hurst_batch import hurst_rows
from fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor
from fedot_ind.core.operation.transformation.representation.statistical.stat_features import hurst_exponent


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('length', [2, 9, 28, 144])
def test_prefix_reductions_and_solve_match_scalar(dtype, length):
    values = np.random.default_rng(17).normal(size=(131, length)).astype(dtype)
    values[1] = 0
    values[2] = 7
    values[3] = values[0] + 1e5
    original = values.copy()
    expected = np.array([hurst_exponent(row, 0) for row in values])
    np.testing.assert_array_equal(hurst_rows(values), expected)
    np.testing.assert_array_equal(values, original)
    permutation = np.random.default_rng(18).permutation(len(values))
    np.testing.assert_array_equal(hurst_rows(values[permutation]), expected[permutation])


@given(st.integers(2, 60), st.integers(1, 8), st.integers(0, 2**16))
@settings(max_examples=15, deadline=None)
def test_row_batching_does_not_change_formula(length, count, seed):
    values = np.random.default_rng(seed).normal(size=(count, length)).astype(np.float32)
    expected = np.array([hurst_exponent(row, 0) for row in values])
    np.testing.assert_array_equal(hurst_rows(values), expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_complete_feature_order_dtype_and_ownership_are_unchanged(dtype):
    values = np.random.default_rng(42).normal(size=(7, 24, 144)).astype(dtype)
    data = SimpleNamespace(features=values)
    params = OperationParameters(use_cache=False, window_size=0, use_sliding_window=False)
    expected = BaseExtractor._transform(QuantileExtractor(params), data)
    actual = QuantileExtractor(params)._transform(data)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
