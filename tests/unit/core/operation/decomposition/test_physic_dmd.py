from typing import Callable

import numpy as np
import pytest

from fedot_ind.core.operation.optimization.dmd.physic_dmd import piDMD


@pytest.fixture
def feature_target():
    return np.random.rand(10, 10), np.random.rand(10, 10)


@pytest.mark.parametrize('method', ('exact', 'orthogonal'))
def test_fit_exact(feature_target, method):
    decomposer = piDMD(method=method)
    features, target = feature_target

    fitted_linear_operator, eigenvals, eigenvectors = decomposer.fit(train_features=features,
                                                                     train_target=target)
    for i in [eigenvals, eigenvectors]:
        assert isinstance(i, np.ndarray)
    assert isinstance(fitted_linear_operator, Callable)
