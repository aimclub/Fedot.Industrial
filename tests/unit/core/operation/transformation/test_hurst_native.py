"""Check every prefix, not just an approximately matching final regression slope."""

from types import SimpleNamespace
import pickle
import random

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.representation.statistical import hurst_batch
from fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor


@pytest.fixture
def native():
    pytest.importorskip('numba')
    if np.__version__ != '1.26.4':
        pytest.skip('Native reduction contract is NumPy 1.26.4')
    from fedot_ind.core.operation.transformation.representation.statistical import native_statistics
    return native_statistics


def assert_prefixes(native, values):
    original = values.copy()
    times = np.arange(1, len(values) + 1)
    cumulative = np.cumsum(values)
    averages = cumulative / times
    deviations, ranges = native.prefixes(values, cumulative, averages)
    expected_std = [np.std(values[:count]) for count in times]
    expected_range = [np.ptp(cumulative[:count] - times[:count] * averages[count - 1]) for count in times]
    np.testing.assert_array_equal(deviations, expected_std)
    np.testing.assert_array_equal(ranges, expected_range)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize('length', [1, 2, 7, 8, 127, 128, 129, 512, 8193, 16385, 28536])
def test_numpy_buffer_and_pairwise_boundaries(native, length):
    assert_prefixes(native, np.random.default_rng(42).normal(size=length).astype(np.float32))


@pytest.mark.parametrize('stride', [2, -1, -3])
def test_native_prefixes_respect_noncontiguous_row_strides(native, stride):
    values = np.random.default_rng(52).normal(size=1025).astype(np.float32)[::stride]
    assert_prefixes(native, values)


@given(st.lists(st.floats(width=32, min_value=-1e5, max_value=1e5,
                          allow_nan=False, allow_infinity=False), min_size=8, max_size=257))
@settings(max_examples=30, deadline=None)
def test_simd_lanes_preserve_each_rounded_prefix(values):
    pytest.importorskip('numba')
    if np.__version__ != '1.26.4':
        pytest.skip('Native reduction contract is NumPy 1.26.4')
    from fedot_ind.core.operation.transformation.representation.statistical import native_statistics
    assert_prefixes(native_statistics, np.asarray(values, dtype=np.float32))


@pytest.mark.parametrize('kind', ['zeros', 'constant', 'ramp', 'offset', 'alternating', 'steps'])
def test_flat_and_near_collinear_curves(native, kind):
    values = np.random.default_rng(21).normal(size=700).astype(np.float32)
    if kind == 'zeros':
        values[:] = 0
    elif kind == 'constant':
        values[:] = np.float32(3.7)
    elif kind == 'ramp':
        values = np.linspace(-1, 1, len(values), dtype=np.float32)
    elif kind == 'offset':
        values += np.float32(1e5)
    elif kind == 'alternating':
        values[::2], values[1::2] = np.float32(1e8), np.float32(-1e8)
    else:
        values = np.round(values)
    assert_prefixes(native, values)


@given(st.integers(2, 1024), st.integers(0, 65535), st.integers(-15, 15))
@settings(max_examples=20, deadline=None)
def test_envelopes_preserve_rounded_extrema(length, seed, exponent):
    pytest.importorskip('numba')
    from fedot_ind.core.operation.transformation.representation.statistical.native_statistics import envelope_ranges
    values = np.random.default_rng(seed).normal(size=length).astype(np.float32) * np.float32(2.**exponent)
    times = np.arange(1, length + 1)
    cumulative = np.cumsum(values)
    averages = cumulative / times
    expected = [np.ptp(cumulative[:count] - times[:count] * averages[count - 1]) for count in times]
    np.testing.assert_array_equal(envelope_ranges(cumulative, averages), expected)


def test_long_table_matches_original_extractor_exactly(native):
    values = np.random.default_rng(15).normal(size=(4, 1025)).astype(np.float32)
    data = SimpleNamespace(features=values)
    params = OperationParameters(use_cache=False, use_sliding_window=False, window_size=0)
    expected = BaseExtractor._transform(QuantileExtractor(params), data)
    actual = QuantileExtractor(params)._transform(data)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert native.prefixes.targetoptions['fastmath'] is False
    assert not native.prefixes.targetoptions.get('parallel', False)


def test_missing_optional_runtime_keeps_reference(monkeypatch):
    monkeypatch.setattr(hurst_batch, 'find_spec', lambda name: None)
    values = np.random.default_rng(18).normal(size=(2, 600)).astype(np.float32)
    assert not hurst_batch.native_hurst_supported(values)
    data = SimpleNamespace(features=values)
    params = OperationParameters(use_cache=False, window_size=0)
    expected = BaseExtractor._transform(QuantileExtractor(params), data)
    np.testing.assert_array_equal(QuantileExtractor(params)._transform(data), expected)


def test_native_path_preserves_rng_and_torch_configuration(native):
    import torch

    values = np.random.default_rng(90).normal(size=(2, 600)).astype(np.float32)
    before = (pickle.dumps(np.random.get_state()), random.getstate(), torch.random.get_rng_state().clone(),
              torch.get_num_threads(), torch.get_num_interop_threads(), torch.are_deterministic_algorithms_enabled())
    hurst_batch.hurst_rows(values)
    assert pickle.dumps(np.random.get_state()) == before[0] and random.getstate() == before[1]
    assert torch.equal(torch.random.get_rng_state(), before[2])
    assert (torch.get_num_threads(), torch.get_num_interop_threads(),
            torch.are_deterministic_algorithms_enabled()) == before[3:]


@pytest.mark.parametrize('kind', ['random', 'repeated', 'zeros', 'offset', 'digit_boundary'])
def test_remaining_long_row_statistics_preserve_values(native, kind):
    from fedot_ind.core.operation.transformation.representation.statistical import long_row_features, stat_features

    values = np.random.default_rng(110).normal(size=1200).astype(np.float32)
    if kind == 'repeated':
        values = np.round(values)
    elif kind == 'zeros':
        values[:] = 0
    elif kind == 'offset':
        values += 1e5
    elif kind == 'digit_boundary':
        values = np.array([0, .99999994, 1, 1.0000001, 9.999999, 10, 10.000001, -100], dtype=np.float32)
    np.testing.assert_equal(long_row_features.benford_correlation(values), stat_features.ben_corr(values, 0))
    np.testing.assert_equal(long_row_features.hjorth_complexity(values), stat_features.hjorth_complexity(values, 0))
