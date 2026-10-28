from types import SimpleNamespace

import numpy as np
import pytest

from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation


def _operation(use_cache):
    op = IndustrialCachableOperationImplementation(OperationParameters(use_cache=use_cache))
    op._transform = lambda data: data.features * 2
    op._convert_to_fedot_datatype = lambda data, features: features
    op.cacher.load_data_from_cache = lambda **kwargs: pytest.fail('Explicitly disabled cache was read')
    op.cacher.cache_data = lambda *args: pytest.fail('Explicitly disabled cache was written')
    return op


def test_declared_cache_opt_out_changes_same_shape_inputs_without_stale_reuse():
    op = _operation(False)
    first = SimpleNamespace(features=np.ones((3, 1, 4)))
    second = SimpleNamespace(features=np.full((3, 1, 4), 7.))
    np.testing.assert_array_equal(op.transform(first), first.features * 2)
    np.testing.assert_array_equal(op.transform(second), second.features * 2)


def test_explicit_argument_overrides_configured_cache():
    op = _operation(True)
    incoming = SimpleNamespace(features=np.ones((3, 1, 4)))
    np.testing.assert_array_equal(op.transform(incoming, use_cache=False), incoming.features * 2)


@pytest.mark.parametrize('value', ['false', 0, {}])
def test_cache_policy_rejects_ambiguous_values(value):
    op = _operation(value)
    with pytest.raises(ValueError, match='boolean'):
        op.transform(SimpleNamespace(features=np.ones((3, 1, 4))))
