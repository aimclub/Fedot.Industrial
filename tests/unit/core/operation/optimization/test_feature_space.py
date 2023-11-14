import numpy as np
import pandas as pd
import pytest

from fedot_ind.core.operation.optimization.FeatureSpace import VarianceSelector


@pytest.fixture
def model_data():
    return dict(quantile=np.random.rand(10, 10),
                signal=np.random.rand(10, 10),
                topological=np.random.rand(10, 10))


def test_get_best_model(model_data):
    selector = VarianceSelector(models=model_data)
    best_model = selector.get_best_model()
    assert isinstance(best_model, str)


def test_transform(model_data):
    selector = VarianceSelector(models=model_data)
    projected = selector.transform(model_data=model_data['quantile'],
                                   principal_components=np.random.rand(10, 2))
    assert isinstance(projected, np.ndarray)


def test_select_discriminative_features(model_data):
    selector = VarianceSelector(models=model_data)
    projected = selector.transform(model_data=model_data['quantile'],
                                   principal_components=np.random.rand(10, 2))

    discriminative_feature = selector.select_discriminative_features(model_data=pd.DataFrame(model_data['quantile']),
                                                                     projected_data=projected)

    assert isinstance(discriminative_feature, dict)
