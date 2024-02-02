import math

from fedot_ind.api.utils.data import init_input_data
from fedot_ind.core.architecture.settings.computational import backend_methods as np
import pandas as pd
import pytest
from fedot.core.data.data import OutputData


from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.repository.constanst_repository import STAT_METHODS_GLOBAL, STAT_METHODS
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


FEATURES = list(STAT_METHODS.keys()) + list(STAT_METHODS_GLOBAL.keys())


def dataset(n_classes):
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=20,
                                                                       max_ts_len=50,
                                                                       n_classes=n_classes,
                                                                       test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


@pytest.fixture
def input_data():
    N_CLASSES = np.random.choice([2, 3])
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
    #expected_n_features = len(stat_methods_global.keys()) + math.ceil(len_ts / (len_ts*window/100)) * len(stat_methods.keys())
    assert train_features_window is not None
    assert isinstance(train_features_window, OutputData)
    #assert expected_n_features == train_features_window.predict.shape[1]


def test_extract_features(quantile_extractor):
    X, y, _, _ = dataset(n_classes=2)
    train_features = quantile_extractor.extract_features(X, y)
    assert train_features is not None
    assert isinstance(train_features, pd.DataFrame)
    assert len(FEATURES) == train_features.shape[1]
