import json

import numpy as np
import pytest
from fedot.core.data.data import InputData, OutputData

from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.api.utils.path_lib import PATH_TO_DEFAULT_PARAMS
from fedot_ind.core.models.recurrence.reccurence_extractor import RecurrenceExtractor
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def dataset(n_classes):
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=100,
                                                                       max_ts_len=24,
                                                                       n_classes=n_classes,
                                                                       test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


@pytest.fixture
def default_params():
    with open(PATH_TO_DEFAULT_PARAMS, 'r') as f:
        default_params = json.load(f)
    return default_params['recurrence_extractor']


@pytest.fixture
def input_data():
    n_classes = np.random.choice([2, 3])
    X_train, y_train, X_test, y_test = dataset(n_classes)
    input_train_data = init_input_data(X_train, y_train)
    return input_train_data


@pytest.fixture
def recurrence_extractor(default_params):
    return RecurrenceExtractor(default_params)


def test_transform(recurrence_extractor, input_data):
    train_features = recurrence_extractor.transform(input_data=input_data)
    assert train_features is not None
    assert isinstance(train_features, OutputData)
    assert train_features.predict.shape[1] == 15


def test_generate_recurrence_features_single(recurrence_extractor, input_data):
    sample = input_data.features[0]
    train_features = recurrence_extractor.generate_recurrence_features(sample)
    assert train_features is not None
    assert isinstance(train_features, InputData)


def test_generate_recurrence_features_multi(recurrence_extractor, input_data):
    samples = input_data.features[:3]
    train_features = recurrence_extractor.generate_recurrence_features(samples)
    assert train_features is not None
    assert isinstance(train_features, InputData)
