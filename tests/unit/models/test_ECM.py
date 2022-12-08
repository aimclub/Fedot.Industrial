from core.utils import *
from core.models.cnn.sfp_models import *
from core.models.cnn.classification_models import *
import pytest


@pytest.fixture()
def clf_models_list():
    return CLF_MODELS


@pytest.fixture()
def sfp_models_list():
    return SFP_MODELS


def test_clf_models(clf_model_list):
    for clf_model in clf_model_list:
        model = clf_model()
        TestModule = ModelTestingModule(model=model)
        train_feats_earthquakes, test_feats_earthquakes = TestModule.extract_from_binary(dataset_name='Earthquakes')
        train_feats_lightning7, test_feats_lightning7 = TestModule.extract_from_multi_class(dataset_name='Lightning7')
        assert train_feats_lightning7 is not None
        assert test_feats_lightning7 is not None

        assert train_feats_earthquakes is not None
        assert test_feats_earthquakes is not None


def test_sfp_models(window_feature_generators_list):
    for sfp_model in window_feature_generators_list:
        model = sfp_model()
        TestModule = ModelTestingModule(model=model)
        train_feats_earthquakes, test_feats_earthquakes = TestModule.extract_from_binary(dataset_name='Earthquakes')
        train_feats_lightning7, test_feats_lightning7 = TestModule.extract_from_multi_class(dataset_name='Lightning7')
        assert train_feats_lightning7 is not None
        assert test_feats_lightning7 is not None

        assert train_feats_earthquakes is not None
        assert test_feats_earthquakes is not None

if __name__ == '__main__':
    test_clf_models(CLF_MODELS)
    test_sfp_models(SFP_MODELS)