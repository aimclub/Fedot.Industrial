from core.architecture.utils.Testing import ModelTestingModule
from core.models.statistical.StatsExtractor import *
from core.models.signal.SignalExtractor import *
from core.models.signal.RecurrenceExtractor import *
from core.models.topological.TopologicalExtractor import *
import pytest


@pytest.fixture()
def feature_generators_list():
    return [StatsExtractor, SignalExtractor, ReccurenceFeaturesExtractor, TopologicalExtractor]


@pytest.fixture()
def window_feature_generators_list():
    return [StatsExtractor]


def test_get_features(feature_generators_list):
    for feature_generator in feature_generators_list:
        model = feature_generator()
        TestModule = ModelTestingModule(model=model)
        train_feats_earthquakes, test_feats_earthquakes = TestModule.extract_from_binary(dataset_name='Earthquakes')
        train_feats_lightning7, test_feats_lightning7 = TestModule.extract_from_multi_class(dataset_name='Lightning7')
        assert train_feats_lightning7 is not None
        assert test_feats_lightning7 is not None

        assert train_feats_earthquakes is not None
        assert test_feats_earthquakes is not None


def test_get_features_window_mode(window_feature_generators_list):
    for feature_generator in window_feature_generators_list:
        model = feature_generator(window_mode=True)
        TestModule = ModelTestingModule(model=model)
        train_feats_earthquakes, test_feats_earthquakes = TestModule.extract_from_binary(dataset_name='Earthquakes')
        train_feats_lightning7, test_feats_lightning7 = TestModule.extract_from_multi_class(dataset_name='Lightning7')
        assert train_feats_lightning7 is not None
        assert test_feats_lightning7 is not None

        assert train_feats_earthquakes is not None
        assert test_feats_earthquakes is not None
