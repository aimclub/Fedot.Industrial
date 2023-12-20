import numpy as np
import pandas as pd
import pytest
from fedot.core.data.data import OutputData

from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.models.quantile.stat_methods import stat_methods, stat_methods_global
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator

FEATURES = list(stat_methods.keys()) + list(stat_methods_global.keys())


def dataset(binary):
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=20,
                                                                       max_ts_len=50,
                                                                       binary=binary,
                                                                       test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


@pytest.fixture
def input_data():
    N_CLASSES = np.random.choice([True, False])
    X_train, y_train, X_test, y_test = dataset(N_CLASSES)
    input_train_data = init_input_data(X_train, y_train)
    return input_train_data


@pytest.fixture
def quantile_extractor():
    return QuantileExtractor({'window_size': 0})


@pytest.fixture
def quantile_extractor_window():
    return QuantileExtractor({'window_size': 20})


def test_transform(quantile_extractor, input_data):
    train_features = quantile_extractor.transform(input_data=input_data)
    assert train_features is not None
    assert isinstance(train_features, OutputData)
    assert len(FEATURES) == train_features.predict.shape[1]


def test_transform_window(quantile_extractor_window, input_data):
    train_features_window = quantile_extractor_window.transform(input_data=input_data)
    window = quantile_extractor_window.window_size
    len_ts = input_data.features.shape[1]
    assert train_features_window is not None
    assert isinstance(train_features_window, OutputData)


def test_extract_features(quantile_extractor):
    X, y, _, _ = dataset(binary=True)
    train_features = quantile_extractor.extract_features(X, y)
    assert train_features is not None
    assert isinstance(train_features, pd.DataFrame)
    assert len(FEATURES) == train_features.shape[1]
