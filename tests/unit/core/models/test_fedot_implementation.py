import numpy as np
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.api.utils.data import init_input_data
from fedot_ind.core.models.automl.fedot_implementation import FedotClassificationImplementation, \
    FedotRegressionImplementation
import pytest


@pytest.fixture(scope='session')
def clf_input_data():
    features = np.random.randn(100, 50)
    target = np.random.randint(0, 2, 100)
    return init_input_data(features, target, task='classification')


@pytest.fixture(scope='session')
def reg_input_data():
    features = np.random.randn(100, 50)
    target = np.random.randn(100)
    return init_input_data(features, target, task='regression')


def test_fedot_classification_implementation(clf_input_data):
    model = FedotClassificationImplementation(OperationParameters(timeout=0.1,
                                                                  problem='classification'))
    model.fit(clf_input_data)
    pred = model.predict(clf_input_data)
    assert model is not None


def test_fedot_regression_implementation(reg_input_data):
    model = FedotRegressionImplementation(OperationParameters(timeout=0.1,
                                                              problem='regression'))
    model.fit(reg_input_data)
    pred = model.predict(reg_input_data)
    assert model is not None
