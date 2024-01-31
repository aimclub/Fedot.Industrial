import os.path

import numpy as np
import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.experiment.TimeSeriesAnomalyDetection import TimeSeriesAnomalyDetectionPreset
from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from fedot_ind.core.architecture.experiment.TimeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.core.models.topological.topological_extractor import TopologicalExtractor
from fedot_ind.core.optimizer import IndustrialEvoOptimizer
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


AVAILABLE_OPERATIONS = available_operations=['scaling', 'normalization', 'xgboost',
                                                       'rfr', 'rf', 'logit', 'mlp', 'knn',
                                                       'lgbm', 'pca']

@pytest.fixture()
def tsc_topo_config():
    return dict(problem='classification',
                dataset='Chinatown',
                strategy='topological',
                timeout=0.1,
                logging_level=40,
                use_cache=False,
                available_operations=AVAILABLE_OPERATIONS)


@pytest.fixture()
def tsc_config():
    experiment_setup = {'problem': 'classification',
                        'metric': 'accuracy',
                        'timeout': 0.5,
                        'num_of_generations': 15,
                        'pop_size': 20,
                        'logging_level': 10,
                        'available_operations': [
                            'eigen_basis',
                            'dimension_reduction',
                            'inception_model',
                            'logit',
                            'rf',
                            'xgboost',
                            'minirocket_extractor',
                            'normalization',
                            'omniscale_model',
                            'pca',
                            'mlp',
                            'quantile_extractor',
                            'scaling',
                            'signal_extractor',
                            'topological_features'
                        ],
                        'n_jobs': 2,
                        'initial_assumption': None,
                        'max_pipeline_fit_time': 10,
                        'with_tuning': False,
                        'early_stopping_iterations': 5,
                        'early_stopping_timeout': 60,
                        'optimizer': IndustrialEvoOptimizer}
    return experiment_setup


@pytest.fixture()
def tsc_fedot_preset_config():
    return dict(problem='classification',
                dataset='Chinatown',
                strategy='fedot_preset',
                timeout=0.5,
                logging_level=40,
                use_cache=False,
                available_operations=AVAILABLE_OPERATIONS)


@pytest.fixture()
def none_tsc_config():
    return dict(problem='classification',
                dataset='Chinatown',
                strategy=None,
                timeout=0.5,
                logging_level=40,
                use_cache=False,
                available_operations=AVAILABLE_OPERATIONS)


@pytest.fixture()
def anomaly_detection_fedot_preset_config():
    return dict(problem='anomaly_detection',
                dataset='custom_dataset',
                strategy='fedot_preset',
                use_cache=False,
                timeout=0.5,
                n_jobs=1,
                logging_level=20,
                available_operations=AVAILABLE_OPERATIONS)


@pytest.fixture()
def decomposition_config():
    return dict(problem='anomaly_detection',
                dataset='custom_dataset',
                strategy='decomposition',
                use_cache=False,
                timeout=0.5,
                n_jobs=1,
                logging_level=20,
                available_operations=AVAILABLE_OPERATIONS)


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

    assert industrial.config_dict['problem'] == 'classification'
    assert industrial.config_dict['dataset'] == 'Chinatown'
    assert industrial.config_dict['strategy'] == 'topological'


def test_main_api_fedot_preset(tsc_fedot_preset_config):
    industrial = FedotIndustrial(**tsc_fedot_preset_config)

    assert industrial.config_dict['problem'] == 'classification'
    assert industrial.config_dict['dataset'] == 'Chinatown'
    assert industrial.config_dict['strategy'] == 'fedot_preset'


def test_main_api_anomaly_detection_fedot_preset(anomaly_detection_fedot_preset_config):
    industrial = FedotIndustrial(**anomaly_detection_fedot_preset_config)

    assert industrial.config_dict['problem'] == 'anomaly_detection'
    assert industrial.config_dict['dataset'] == 'custom_dataset'
    assert industrial.config_dict['strategy'] == 'fedot_preset'


# def test_api_tsc(tsc_topo_config):
def test_api_tsc(tsc_config):
    # tsc_topo_config.update({'output_folder': '.'})
    industrial = FedotIndustrial(**tsc_config)
    train_data, test_data = TimeSeriesDatasetsGenerator(num_samples=50,
                                                        max_ts_len=30,
                                                        binary=True,
                                                        test_size=0.5).generate_data()
    model = industrial.fit(input_data=train_data)
    # model = industrial.fit(features=train_data[0], target=train_data[1])
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
