import numpy as np
import pytest

from core.api.API import Industrial
from tests.unit.api.test_API_config import load_data


@pytest.fixture()
def basic_config_API():
    config_path = 'tests/data/config/Config_Classification.yaml'
    return config_path


@pytest.fixture()
def basic_API_class():
    ExperimentHelper = Industrial()
    return ExperimentHelper


def test_API_code_scenario(basic_API_class):
    train_data, test_data, n_classes = load_data('Lightning7')

    IndustrialModel, train_feats = basic_API_class.fit(model_name='wavelet',
                                                       task_type='ts_classification',
                                                       model_params={
                                                           'problem': 'classification',
                                                           'seed': 42,
                                                           'timeout': 1,
                                                           'max_depth': 3,
                                                           'max_arity': 1,
                                                           'cv_folds': 2,
                                                           'logging_level': 20,
                                                           'n_jobs': 2
                                                       },

                                                       train_features=train_data[0],
                                                       train_target=train_data[1])

    labels, test_feats = basic_API_class.predict(features=test_data[0])
    probs, test_feats = basic_API_class.predict_proba(features=test_data[0])

    metrics = basic_API_class.get_metrics(target=test_data[1],
                                          prediction_label=labels,
                                          prediction_proba=probs)
    assert IndustrialModel is not None
    assert basic_API_class.fitted_model is not None
    assert basic_API_class.train_features is not None
    assert type(train_feats) is np.ndarray
    assert type(test_feats) is np.ndarray
    assert type(probs) is np.ndarray
    assert type(labels) is np.ndarray
    assert type(metrics) is dict


def test_API_config_scenario(basic_API_class, basic_config_API):
    experiment_results = basic_API_class.run_experiment(config=basic_config_API, save_flag=False)
    dataset_list = basic_API_class.YAML.config_dict['datasets_list']
    fg_list = basic_API_class.YAML.config_dict['feature_generator']
    assert len(basic_API_class.YAML.config_dict['datasets_list']) == len(list(experiment_results.keys()))
    assert len(fg_list) == len(list(experiment_results[dataset_list[0]]['Original'].keys()))
