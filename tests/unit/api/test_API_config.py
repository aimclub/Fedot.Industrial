import numpy as np
import pandas as pd
import pytest
from core.api.API import Industrial
from core.api.main import FedotIndustrial


@pytest.fixture()
def basic_config_API():
    config_path = 'tests/data/config/Config_Classification.yaml'
    return config_path


@pytest.fixture()
def basic_API_class():
    ExperimentHelper = Industrial()
    return ExperimentHelper


def load_data(dataset_name):
    config = dict(task='ts_classification',
                  dataset=dataset_name,
                  feature_generator='topological',
                  use_cache=False,
                  error_correction=False,
                  timeout=1,
                  n_jobs=2,
                  window_sizes='auto')

    industrial = FedotIndustrial(input_config=config, output_folder=None)
    train_data, test_data, n_classes = industrial.reader.read(dataset_name=dataset_name)
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
