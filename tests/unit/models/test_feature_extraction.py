from core.architecture.utils.Testing import ModelTestingModule
from core.models.statistical.QuantileRunner import *
from core.models.spectral.SSARunner import *
from core.models.signal.SignalRunner import *
from core.models.signal.RecurrenceRunner import *
from core.models.topological.TopologicalRunner import *
import pytest


@pytest.fixture()
def feature_generators_list():
    return [StatsRunner, SSARunner, SignalRunner, RecurrenceRunner, TopologicalRunner]


@pytest.fixture()
def window_feature_generators_list():
    return [StatsRunner]


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
