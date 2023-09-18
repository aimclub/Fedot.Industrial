import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.experiment.TimeSeriesAnomalyDetection import TimeSeriesAnomalyDetectionPreset
from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from fedot_ind.core.architecture.experiment.TimeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.core.models.topological.topological_extractor import TopologicalExtractor


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


@pytest.fixture()
def anomaly_detection_fedot_preset_config():
    config = dict(task='anomaly_detection',
                  dataset='custom_dataset',
                  strategy='fedot_preset',
                  use_cache=False,
                  timeout=0.5,
                  n_jobs=1,
                  logging_level=20,
                  output_folder='.')
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
    assert industrial.solver.extractors == ['quantile_extractor', 'quantile_extractor', 'quantile_extractor']
    assert industrial.solver.branch_nodes == ['data_driven_basis', 'fourier_basis', 'wavelet_basis']
    assert industrial.config_dict['task'] == 'ts_classification'
    assert industrial.solver.dataset_name == 'Chinatown'


def test_main_api_anomaly_detection_fedot_preset(anomaly_detection_fedot_preset_config):
    industrial = FedotIndustrial(**anomaly_detection_fedot_preset_config)
    assert type(industrial) is FedotIndustrial
    assert type(industrial.solver) is TimeSeriesAnomalyDetectionPreset
    assert industrial.solver.extractors == ['quantile_extractor', 'quantile_extractor', 'quantile_extractor']
    assert industrial.solver.branch_nodes == ['data_driven_basis', 'fourier_basis', 'wavelet_basis']
    assert industrial.config_dict['task'] == 'anomaly_detection'
