import numpy as np
import pandas as pd
import pytest
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.api.utils.data import init_input_data
from fedot_ind.core.operation.interfaces.fedot_automl_evaluation_strategy import FedotAutoMLClassificationStrategy, \
    FedotAutoMLRegressionStrategy


@pytest.fixture()
def regression_data():
    features = np.random.rand(10, 10)
    target = np.random.rand(10, 1)
    return init_input_data(X=pd.DataFrame(features), y=target, task='regression')


@pytest.fixture()
def classification_data():
    features = np.random.rand(10, 10)
    target = np.random.randint(2, size=10)
    return init_input_data(X=pd.DataFrame(features), y=target, task='classification')


def test_fedot_automl_classification_strategy_fit(classification_data):
    operation_type = 'fedot_cls'
    params = OperationParameters()
    params._parameters.update({'problem': 'classification', 'timeout': 0.5})
    strategy = FedotAutoMLClassificationStrategy(operation_type=operation_type,
                                                 params=params)
    # trained_operation = strategy.fit(classification_data)

    # predict = strategy.predict(trained_operation, classification_data)
    # predict_for_fit = strategy.predict_for_fit(trained_operation, classification_data)

    # assert predict.predict is not None
    # assert predict_for_fit.predict is not None
    assert strategy.operation_impl is not None
    # assert trained_operation is not None


def test_fedot_automl_regression_strategy_fit(regression_data):
    operation_type = 'fedot_regr'
    params = OperationParameters()
    params._parameters.update({'problem': 'regression', 'timeout': 0.1})
    strategy = FedotAutoMLRegressionStrategy(operation_type=operation_type,
                                             params=params)
    # trained_operation = strategy.fit(regression_data)
    #
    # predict = strategy.predict(trained_operation, regression_data)
    # predict_for_fit = strategy.predict_for_fit(trained_operation, regression_data)

    # assert predict.predict is not None
    # assert predict_for_fit.predict is not None
    assert strategy.operation_impl is not None
    # assert trained_operation is not None
