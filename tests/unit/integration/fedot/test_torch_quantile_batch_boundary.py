"""Real deferred transform preserves its InputData and array boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor
from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
from fedot_ind.integration.fedot.extensions.factories import make_deferred_factory


@pytest.mark.parametrize('task', ['classification', 'regression'])
@pytest.mark.parametrize('shape', [(4, 28), (4, 1, 28)])
def test_deferred_torch_quantile_preserves_features_and_input_ownership(task, shape):
    declaration = load_industrial_extension_catalog().operation('quantile_extractor_torch')
    params = {'use_cache': False, 'use_sliding_window': False, 'window_size': 0, 'task_type': task}
    features = np.random.default_rng(34).normal(size=shape).astype(np.float64)
    target = np.arange(len(features)) % 2 if task == 'classification' else np.arange(len(features), dtype=float)
    original_features, original_target = features.copy(), target.copy()
    reference = TorchQuantileExtractor(OperationParameters(**params))
    expected = BaseExtractor._transform(reference, SimpleNamespace(features=features)).reshape(len(features), -1)
    runtime = make_deferred_factory(declaration)(params)
    runtime.fit(features, target)
    actual = runtime.transform(features)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(target, original_target)
    assert runtime.implementation.predict.shape == (len(features), 1, 28)
    before = deepcopy(runtime.implementation.params.to_dict())
    runtime.transform(features[:2])
    assert runtime.implementation.params.to_dict() == before
