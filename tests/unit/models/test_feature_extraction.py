import json
import os

import pytest


from fedot_ind.api.utils.path_lib import PATH_TO_DEFAULT_PARAMS, PROJECT_PATH
from fedot_ind.core.models.recurrence.RecurrenceExtractor import RecurrenceExtractor
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.models.topological.TopologicalExtractor import TopologicalExtractor
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader


@pytest.fixture()
def feature_generators_list():
    return [QuantileExtractor, RecurrenceExtractor, TopologicalExtractor]


@pytest.fixture()
def get_multilabel_data():
    path_to_local_folder = os.path.join(PROJECT_PATH, 'tests', 'data', 'datasets', 'classification_multi')

    (train_features, train_target), (test_features, test_target) = DataLoader(dataset_name='ECG5000_small',
                                                                              folder=path_to_local_folder).load_data()
    return (train_features, train_target), (test_features, test_target)


@pytest.fixture()
def get_binary_data():
    path_to_local_folder = os.path.join(PROJECT_PATH, 'tests', 'data', 'datasets', 'classification_binary')

    (train_features, train_target), (test_features, test_target) = DataLoader(dataset_name='ECG200_small',
                                                                              folder=path_to_local_folder).load_data()
    return (train_features, train_target), (test_features, test_target)


def get_generator_params(generator_name: str):
    with open(PATH_TO_DEFAULT_PARAMS, 'r') as file:
        _feature_gen_params = json.load(file)
        params = _feature_gen_params[f'{generator_name}_extractor']
    return params


@pytest.fixture()
def get_topological_extractor():
    _params = get_generator_params('topological')
    return TopologicalExtractor(_params)


@pytest.fixture()
def get_quantile_extractor():
    _params = get_generator_params('quantile')
    return QuantileExtractor(_params)


@pytest.fixture()
def get_recurrence_extractor():
    _params = get_generator_params('recurrence')
    return RecurrenceExtractor(_params)


def test_topological_binary(get_binary_data, get_topological_extractor):
    (train_features, train_target), (test_features, test_target) = get_binary_data
    model = get_topological_extractor

    train_features = model.generate_features_from_ts(train_features)
    test_features = model.generate_features_from_ts(test_features)

    assert train_features is not None
    assert test_features is not None
    assert train_features.shape[0] == train_target.shape[0]
    assert test_features.shape[0] == test_target.shape[0]


def test_topological_multilabel(get_multilabel_data, get_topological_extractor):
    (train_features, train_target), (test_features, test_target) = get_multilabel_data
    model = get_topological_extractor

    train_features = model.generate_features_from_ts(train_features)
    test_features = model.generate_features_from_ts(test_features)

    assert train_features is not None
    assert test_features is not None
    assert train_features.shape[0] == train_target.shape[0]
    assert test_features.shape[0] == test_target.shape[0]


def test_stats_binary(get_binary_data, get_quantile_extractor):
    (train_features, train_target), (test_features, test_target) = get_binary_data
    model = get_quantile_extractor

    train_features = model.generate_features_from_ts(train_features)
    test_features = model.generate_features_from_ts(test_features)

    assert train_features is not None
    assert test_features is not None
    assert train_features.shape[0] == train_target.shape[0]
    assert test_features.shape[0] == test_target.shape[0]


def test_stats_multilabel(get_multilabel_data, get_quantile_extractor):
    (train_features, train_target), (test_features, test_target) = get_multilabel_data
    model = get_quantile_extractor

    train_features = model.generate_features_from_ts(train_features)
    test_features = model.generate_features_from_ts(test_features)

    assert train_features is not None
    assert test_features is not None
    assert train_features.shape[0] == train_target.shape[0]
    assert test_features.shape[0] == test_target.shape[0]


def test_recurrence_binary(get_binary_data, get_recurrence_extractor):
    (train_features, train_target), (test_features, test_target) = get_binary_data
    model = get_recurrence_extractor

    train_features = model.generate_features_from_ts(train_features)
    test_features = model.generate_features_from_ts(test_features)

    assert train_features is not None
    assert test_features is not None
    assert train_features.shape[0] == train_target.shape[0]
    assert test_features.shape[0] == test_target.shape[0]


def test_recurrence_multilabel(get_multilabel_data, get_recurrence_extractor):
    (train_features, train_target), (test_features, test_target) = get_multilabel_data
    model = get_recurrence_extractor

    train_features = model.generate_features_from_ts(train_features)
    test_features = model.generate_features_from_ts(test_features)

    assert train_features is not None
    assert test_features is not None
    assert train_features.shape[0] == train_target.shape[0]
    assert test_features.shape[0] == test_target.shape[0]
