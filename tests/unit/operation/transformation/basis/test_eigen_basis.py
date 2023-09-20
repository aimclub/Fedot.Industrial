import numpy as np
import pytest
from fedot.core.data.data import OutputData

from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


@pytest.fixture
def dataset():
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=20,
                                                                       max_ts_len=50,
                                                                       n_classes=2,
                                                                       test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


@pytest.fixture
def input_test(dataset):
    X_train, y_train, X_test, y_test = dataset
    input_test_data = init_input_data(X_test, y_test)
    return input_test_data


@pytest.fixture
def input_train(dataset):
    X_train, y_train, X_test, y_test = dataset
    input_train_data = init_input_data(X_train, y_train)
    return input_train_data


def test_transform(input_test):
    input_train_data = input_test
    basis = EigenBasisImplementation({'window_size': 30})
    train_features = basis.transform(input_data=input_train_data)
    assert isinstance(train_features, OutputData)
    assert train_features.features.shape[0] == input_train_data.features.shape[0]


def test_get_threshold(input_train):
    basis = EigenBasisImplementation({'window_size': 30})
    threshold = basis.get_threshold(input_train.features)
    assert isinstance(threshold, np.int64)
    assert threshold > 0
    assert threshold < input_train.features.shape[1]


def test_transform_one_sample(input_train):
    basis = EigenBasisImplementation({'window_size': 30})
    basis.SV_threshold = 3
    sample = input_train.features[0]
    transformed_sample = basis._transform_one_sample(sample)
    assert isinstance(transformed_sample, np.ndarray)
    assert transformed_sample.shape[0] == basis.SV_threshold
    assert transformed_sample.shape[1] == len(sample)
