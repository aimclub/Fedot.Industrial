import pytest

from fedot_ind.api.utils.hp_generator_collection import GeneratorParams
from fedot_ind.core.models.signal.RecurrenceExtractor import RecurrenceExtractor
from fedot_ind.core.models.statistical.StatsExtractor import StatsExtractor
from fedot_ind.core.models.topological.TopologicalExtractor import TopologicalExtractor


@pytest.fixture()
def feature_generators_list():
    return [StatsExtractor, RecurrenceExtractor, TopologicalExtractor]


@pytest.fixture()
def get_multilabel_data():
    from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
    (train_features, train_target), (test_features, test_target) = DataLoader('SmoothSubspace').load_data()
    return (train_features, train_target), (test_features, test_target)


@pytest.fixture()
def get_binary_data():
    from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
    (train_features, train_target), (test_features, test_target) = DataLoader('Chinatown').load_data()
    return (train_features, train_target), (test_features, test_target)


@pytest.fixture()
def get_topological_extractor():
    return TopologicalExtractor(GeneratorParams['topological'].value)


@pytest.fixture()
def get_stats_extractor():
    return StatsExtractor(GeneratorParams['statistical'].value)


@pytest.fixture()
def get_recurrence_extractor():
    return RecurrenceExtractor(GeneratorParams['recurrence'].value)


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


def test_stats_binary(get_binary_data, get_stats_extractor):
    (train_features, train_target), (test_features, test_target) = get_binary_data
    model = get_stats_extractor

    train_features = model.generate_features_from_ts(train_features)
    test_features = model.generate_features_from_ts(test_features)

    assert train_features is not None
    assert test_features is not None
    assert train_features.shape[0] == train_target.shape[0]
    assert test_features.shape[0] == test_target.shape[0]


def test_stats_multilabel(get_multilabel_data, get_stats_extractor):
    (train_features, train_target), (test_features, test_target) = get_multilabel_data
    model = get_stats_extractor

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
