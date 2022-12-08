from core.models.topological.TopologicalRunner import *
import pytest
from core.api.API import Industrial


@pytest.fixture()
def basic_config_API():
    config_path = 'tests/data/config/Config_Classification.yaml'
    return config_path


@pytest.fixture()
def basic_API_class():
    ExperimentHelper = Industrial()
    return ExperimentHelper


def load_data(dataset_name):
    train_data, test_data, n_classes = Industrial().reader.read(dataset_name=dataset_name)
    return train_data, test_data, n_classes


def test_YAMl_reader(basic_API_class, basic_config_API):
    config_dict = basic_API_class.YAML.read_yaml_config(config_path=basic_config_API, return_dict=True)
    assert type(config_dict) is dict


def test_API_data_reader(basic_API_class, basic_config_API):
    exclude = ['topological', 'spectral', 'window_spectral']
    config_dict = basic_API_class.YAML.read_yaml_config(config_path=basic_config_API, return_dict=True)
    train_data, test_data, n_classes = load_data('Worms')
    filtred_dict = basic_API_class.exclude_generators(config_dict, train_data=train_data[0])
    assert type(train_data[0]) is pd.DataFrame
    assert type(train_data[1]) is np.ndarray
    assert type(n_classes) is int
    assert any(exclude) not in filtred_dict['feature_generator']


def test_API_code_scenario(basic_API_class):
    train_data, test_data, n_classes = load_data('Lightning7')

    IndustrialModel, train_feats = basic_API_class.fit(model_name='quantile',
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
                                                       feature_generator_params={'window_mode': False},
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
    basic_API_class.run_experiment(basic_config_API)
    _ = 1
    assert basic_API_class is not None
