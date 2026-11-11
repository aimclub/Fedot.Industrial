"""NumPy InputData batching must keep the old row-wise public result."""

import cProfile
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.torch_backend.enums import StatisticalFeature
from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor


def extractor(**kwargs):
    return TorchQuantileExtractor(OperationParameters(use_cache=False, use_sliding_window=False, **kwargs))


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32])
@pytest.mark.parametrize('shape', [(1, 20), (5, 28), (5, 1, 28), (5, 2, 28)])
@pytest.mark.parametrize('global_features', [False, True])
def test_numpy_input_matches_rowwise_reference(dtype, shape, global_features):
    values = (np.random.default_rng(123).normal(size=shape) * 5).astype(dtype)
    original = values.copy()
    data = SimpleNamespace(features=values)
    old, new = extractor(add_global_features=global_features), extractor(add_global_features=global_features)
    expected = BaseExtractor._transform(old, data)
    actual = new._transform(data)
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, original)
    assert new.predict is actual
    assert new.is_multichanel == old.is_multichanel


def test_eligible_input_uses_fewer_generator_calls():
    instance = extractor(window_size=0)
    values = np.random.default_rng(19).normal(size=(12, 1, 28)).astype(np.float32)
    profiler = cProfile.Profile()
    profiler.runcall(instance._transform, SimpleNamespace(features=values))
    code = TorchQuantileExtractor._generate_numpy_batch.__code__
    calls = sum(entry.callcount for entry in profiler.getstats() if entry.code is code)
    assert 0 < calls < len(values)


@pytest.mark.parametrize('rows', [129, 135])
def test_long_batch_is_bounded_and_does_not_reorder_rows(rows):
    values = np.random.default_rng(27).normal(size=(rows, 1, 9)).astype(np.float32)
    expected = BaseExtractor._transform(extractor(), SimpleNamespace(features=values))
    actual = extractor()._transform(SimpleNamespace(features=values))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('change', ['subclass', 'method', 'config', 'order', 'window', 'noncontiguous', 'channel'])
def test_non_audited_paths_keep_reference(change):
    class Custom(TorchQuantileExtractor):
        pass

    values = np.random.default_rng(17).normal(size=(3, 1, 28)).astype(np.float32)
    instance = Custom(OperationParameters(use_cache=False)) if change == 'subclass' else extractor()
    if change == 'method':
        instance._methods_local[StatisticalFeature.mean] = lambda ts, axis: ts.mean(dim=axis) + 3
    elif change == 'config':
        instance.config = {StatisticalFeature.mean: {}}
    elif change == 'order':
        instance.config = dict(reversed(list(instance.config.items())))
    elif change == 'window':
        instance.window_size, instance.stride, instance.use_sliding_window = 20, 2, True
    elif change == 'noncontiguous':
        values = values[..., ::2]
    elif change == 'channel':
        instance.channel_extraction = False
    old = deepcopy(instance)
    expected = BaseExtractor._transform(old, SimpleNamespace(features=values))
    actual = instance._transform(SimpleNamespace(features=values))
    np.testing.assert_array_equal(actual, expected)


def test_tensor_input_keeps_existing_output_and_device():
    values = torch.arange(112, dtype=torch.float64).reshape(4, 28).sin()
    expected = BaseExtractor._transform(extractor(), SimpleNamespace(features=values))
    actual = extractor()._transform(SimpleNamespace(features=values))
    assert isinstance(actual, torch.Tensor) and actual.device == values.device
    assert torch.equal(actual, expected)


def test_batch_does_not_change_rng_or_alias_outputs():
    values = np.random.default_rng(3).normal(size=(3, 1, 28)).astype(np.float32)
    state = torch.random.get_rng_state().clone()
    first, second = extractor(), extractor()
    a = first._transform(SimpleNamespace(features=values))
    b = second._transform(SimpleNamespace(features=values))
    assert torch.equal(torch.random.get_rng_state(), state)
    a[:] = 99
    assert not np.array_equal(a, b)
    np.testing.assert_allclose(second.predict, b)


def test_batch_composition_does_not_change_values():
    rng = np.random.default_rng(57)
    values = rng.normal(size=(6, 1, 28)).astype(np.float32)
    values[1] = np.round(values[1], 1)
    values[2] = np.round(values[2])
    values[3] = values[0]
    data = SimpleNamespace(features=values)
    expected = BaseExtractor._transform(extractor(), data)
    perm = np.array([4, 0, 5, 2, 3, 1])
    actual = extractor()._transform(SimpleNamespace(features=values[perm]))
    np.testing.assert_array_equal(actual[np.argsort(perm)], expected)
    parts = [extractor()._transform(SimpleNamespace(features=part))
             for part in (values[:1], values[1:3], values[3:])]
    np.testing.assert_array_equal(np.concatenate(parts), expected)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_conversion_precedes_statistics_for_large_offsets(dtype):
    values = (1e3 + np.random.default_rng(32).normal(size=(4, 1, 28))).astype(dtype)
    expected = BaseExtractor._transform(extractor(), SimpleNamespace(features=values))
    actual = extractor()._transform(SimpleNamespace(features=values))
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
