import os.path

import numpy as np
import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.experiment.TimeSeriesAnomalyDetection import TimeSeriesAnomalyDetectionPreset
from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from fedot_ind.core.architecture.experiment.TimeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.core.models.topological.topological_extractor import TopologicalExtractor
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


@pytest.fixture()
def tsc_topo_config():
    return dict(task='ts_classification',
                dataset='Chinatown',
                strategy='topological',
                timeout=0.1,
                logging_level=40,
                use_cache=False)


@pytest.fixture()
def tsc_fedot_preset_config():
    return dict(task='ts_classification',
                dataset='Chinatown',
                strategy='fedot_preset',
                timeout=0.5,
                logging_level=40,
                use_cache=False)


@pytest.fixture()
def none_tsc_config():
    return dict(task='ts_classification',
                dataset='Chinatown',
                strategy=None,
                timeout=0.5,
                logging_level=40,
                use_cache=False)


@pytest.fixture()
def anomaly_detection_fedot_preset_config():
    return dict(task='anomaly_detection',
                dataset='custom_dataset',
                strategy='fedot_preset',
                use_cache=False,
                timeout=0.5,
                n_jobs=1,
                logging_level=20)


@pytest.fixture()
def decomposition_config():
    return dict(task='anomaly_detection',
                dataset='custom_dataset',
                strategy='decomposition',
                use_cache=False,
                timeout=0.5,
                n_jobs=1,
                logging_level=20)


@pytest.fixture()
def ts_config():
    return dict(random_walk={'ts_type': 'random_walk',
                             'length': 1000,
                             'start_val': 36.6})


@pytest.fixture()
def anomaly_config():
    return {'dip': {'level': 20,
                    'number': 2,
                    'min_anomaly_length': 10,
                    'max_anomaly_length': 20}
            }


def test_main_api_topo(tsc_topo_config):
    industrial = FedotIndustrial(**tsc_topo_config)

    assert type(industrial.solver) is TimeSeriesClassifier
    assert industrial.solver.strategy == 'topological'
    assert industrial.config_dict['task'] == 'ts_classification'
    assert industrial.solver.dataset_name == 'Chinatown'
    assert type(industrial.solver.generator_runner) == TopologicalExtractor


def test_main_api_fedot_preset(tsc_fedot_preset_config):
    industrial = FedotIndustrial(**tsc_fedot_preset_config)

    assert type(industrial.solver) is TimeSeriesClassifierPreset
    assert industrial.solver.extractors == ['quantile_extractor', 'quantile_extractor', 'quantile_extractor']
    assert industrial.solver.branch_nodes == ['eigen_basis', 'fourier_basis', 'wavelet_basis']
    assert industrial.config_dict['task'] == 'ts_classification'
    assert industrial.solver.dataset_name == 'Chinatown'


def test_main_api_anomaly_detection_fedot_preset(anomaly_detection_fedot_preset_config):
    industrial = FedotIndustrial(**anomaly_detection_fedot_preset_config)

    assert type(industrial.solver) is TimeSeriesAnomalyDetectionPreset
    assert industrial.solver.extractors == ['quantile_extractor', 'quantile_extractor', 'quantile_extractor']
    assert industrial.solver.branch_nodes == ['eigen_basis', 'fourier_basis', 'wavelet_basis']
    assert industrial.config_dict['task'] == 'anomaly_detection'


def test_api_tsc(tsc_topo_config):
    tsc_topo_config.update({'output_folder': '.'})
    industrial = FedotIndustrial(**tsc_topo_config)
    train_data, test_data = TimeSeriesDatasetsGenerator(num_samples=50,
                                                        max_ts_len=30,
                                                        binary=True,
                                                        test_size=0.5).generate_data()
    model = industrial.fit(features=train_data[0], target=train_data[1])
    labels = industrial.predict(features=test_data[0], target=test_data[1])
    probs = industrial.predict_proba(features=test_data[0], target=test_data[1])
    metrics = industrial.get_metrics(target=test_data[1], metric_names=['roc_auc', 'accuracy'])

    for name, predict in zip(('labels', 'probs'), (labels, probs)):
        industrial.save_predict(predicted_data=predict, kind=name)
    industrial.save_metrics(metrics=metrics)

    expected_results_path = industrial.solver.saver.path

    for result in (model, labels, probs, metrics):
        assert result is not None
    for s in ('labels', 'probs', 'metrics'):
        filepath = expected_results_path + f'/{s}.csv'
        assert os.path.isfile(filepath)


def test_generate_ts(tsc_topo_config, ts_config):
    industrial = FedotIndustrial(**tsc_topo_config)
    ts = industrial.generate_ts(ts_config=ts_config)

    assert isinstance(ts, np.ndarray)
    assert ts.shape[0] == 1000


def test_generate_anomaly_ts(tsc_topo_config, ts_config, anomaly_config):
    industrial = FedotIndustrial(**tsc_topo_config)
    init_synth_ts, mod_synth_ts, synth_inters = industrial.generate_anomaly_ts(ts_data=ts_config,
                                                                               anomaly_config=anomaly_config)
    assert len(init_synth_ts) == len(mod_synth_ts)
    for anomaly_type in synth_inters:
        for interval in synth_inters[anomaly_type]:
            ts_range = range(len(init_synth_ts))
            assert interval[0] in ts_range and interval[1] in ts_range


def test_split_ts(tsc_topo_config):
    anomaly_dict = {'anomaly1': [[40, 50], [60, 80]],
                    'anomaly2': [[130, 170], [300, 320]]}
    industrial = FedotIndustrial(**tsc_topo_config)
    train_data, test_data = industrial.split_ts(time_series=np.random.rand(1000),
                                                anomaly_dict=anomaly_dict,
                                                plot=False)

    assert train_data is not None
    assert test_data is not None
