import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier, TimeSeriesClassifierNN
from fedot_ind.core.architecture.experiment.TImeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.core.models.topological.TopologicalExtractor import TopologicalExtractor


@pytest.fixture()
def tsc_topo_config():
    config = dict(task='ts_classification',
                  dataset='Chinatown',
                  strategy='topological',
                  timeout=0.5,
                  logging_level=40,
                  use_cache=False)
    return config


@pytest.fixture()
def tsc_fedot_preset_config():
    config = dict(task='ts_classification',
                  dataset='Chinatown',
                  strategy='fedot_preset',
                  timeout=0.5,
                  logging_level=40,
                  use_cache=False)
    return config


@pytest.fixture()
def none_tsc_config():
    config = dict(task='ts_classification',
                  dataset='Chinatown',
                  strategy=None,
                  timeout=0.5,
                  logging_level=40,
                  use_cache=False)
    return config


def test_main_api_topo(tsc_topo_config):
    industrial = FedotIndustrial(**tsc_topo_config)

    assert type(industrial) is FedotIndustrial
    assert type(industrial.solver) is TimeSeriesClassifier
    assert industrial.solver.strategy == 'topological'
    assert industrial.config_dict['task'] == 'ts_classification'
    assert industrial.solver.dataset_name == 'Chinatown'
    assert type(industrial.solver.generator_runner) == TopologicalExtractor


def test_main_api_fedot_preset(tsc_fedot_preset_config):
    industrial = FedotIndustrial(**tsc_fedot_preset_config)

    assert type(industrial) is FedotIndustrial
    assert type(industrial.solver) is TimeSeriesClassifierPreset
    assert industrial.solver.generator_name == 'fedot_preset'
    assert industrial.config_dict['task'] == 'ts_classification'
    assert industrial.solver.dataset_name == 'Chinatown'


def test_main_api_none(none_tsc_config):
    industrial = FedotIndustrial(**none_tsc_config)

    assert type(industrial.solver) is TimeSeriesClassifierNN



