import json

import numpy as np
import pandas as pd
import pytest
from fedot.core.data.data import OutputData

from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.api.utils.path_lib import PATH_TO_DEFAULT_PARAMS
from fedot_ind.core.models.signal.signal_extractor import SignalExtractor
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def dataset(binary):
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=100,
                                                                       max_ts_len=24,
                                                                       binary=True,
                                                                       test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


@pytest.fixture
def default_params():
    with open(PATH_TO_DEFAULT_PARAMS, 'r') as f:
        default_params = json.load(f)
    return default_params['signal_extractor']


@pytest.fixture
def input_data():
    n_classes = np.random.choice([2, 3])
    X_train, y_train, X_test, y_test = dataset(n_classes)
    input_train_data = init_input_data(X_train, y_train)
    return input_train_data


@pytest.fixture
def signal_extractor(default_params):
    return SignalExtractor(default_params)


def test_transform(signal_extractor, input_data):
    train_features = signal_extractor.transform(input_data=input_data)
    assert train_features is not None
    assert isinstance(train_features, OutputData)


def test_extract_features(signal_extractor):
    X, y, _, _ = dataset(binary=True)
    train_features = signal_extractor.extract_features(X, y)
    assert train_features is not None
    assert isinstance(train_features, pd.DataFrame)
