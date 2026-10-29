"""Fast paths preserve each backend's original values, not each other's formulas."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor
from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor


@pytest.mark.parametrize('factory', [QuantileExtractor, TorchQuantileExtractor])
@pytest.mark.parametrize('shape', [(1, 1, 20), (7, 3, 100), (45, 3, 9), (5, 6, 28)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_multichannel_matches_reference_exactly(factory, shape, dtype):
    values = np.random.default_rng(42).normal(size=shape).astype(dtype)
    original = values.copy()
    params = OperationParameters(use_cache=False, window_size=0, use_sliding_window=False)
    reference, candidate = factory(params), factory(params)
    data = SimpleNamespace(features=values)
    expected = BaseExtractor._transform(reference, data)
    state = torch.random.get_rng_state().clone()
    actual = candidate._transform(data)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(values, original)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize('factory', [QuantileExtractor, TorchQuantileExtractor])
def test_customized_statistic_keeps_reference_path(factory):
    values = np.random.default_rng(12).normal(size=(3, 2, 20)).astype(np.float32)
    instance = factory(OperationParameters(use_cache=False))
    instance._clean_predict = lambda array: np.asarray(array) + 7
    expected = BaseExtractor._transform(instance, SimpleNamespace(features=values))
    np.testing.assert_array_equal(instance._transform(SimpleNamespace(features=values)), expected)


def test_custom_dispatch_and_unhashable_callables_are_not_treated_as_audited():
    class CustomMean:
        __hash__ = None

        def __call__(self, values, axis=-1):
            return values.mean(dim=axis)

    instance = TorchQuantileExtractor(OperationParameters(use_cache=False))
    key = next(iter(instance._methods_local))
    instance._methods_local[key] = CustomMean()
    assert instance._select_methods(False)[0][1] == {}
    instance._supported_kwargs = lambda method, kwargs: {'custom': True}
    assert all(kwargs == {'custom': True} for method, kwargs in instance._select_methods(False))
