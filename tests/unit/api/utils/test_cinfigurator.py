import pytest
from fedot_ind.api.utils.configurator import Configurator, IndustrialConfigs
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor

TASKS = ['ts_classification', 'ts_regression', 'anomaly_detection']

QUANTILE = dict(task='ts_classification',
                dataset='dataset_name',
                strategy='quantile',
                use_cache=False,
                timeout=1,
                n_jobs=-1,
                window_size=20,
                output_folder='output_folder')

PRESET = dict(task='ts_classification',
              dataset='dataset_name',
              strategy='fedot_preset',
              branch_nodes=['wavelet_basis'],
              tuning_iters=5,
              tuning_timeout=15,
              use_cache=False,
              timeout=1,
              n_jobs=-1,
              output_folder='output_folder')


def test_base_config():
    configs = [Configurator()._base_config(task) for task in TASKS]
    assert all([isinstance(config, dict) for config in configs])


def test_init_experiment_setup_quantile():
    setup = Configurator().init_experiment_setup(**QUANTILE)
    generator = setup['generator_class']
    fedot_params = setup['model_params']
    assert fedot_params['timeout'] == 1
    assert fedot_params['n_jobs'] == -1
    assert isinstance(generator, QuantileExtractor)
    assert generator.window_size == 20
    assert isinstance(setup, dict)
    assert setup['task'] == 'ts_classification'
    assert setup['strategy'] == 'quantile'
    assert setup['use_cache'] is False
    assert setup['window_size'] == 20


def test_init_experiment_setup_preset():
    setup = Configurator().init_experiment_setup(**PRESET)
    fedot_params = setup['model_params']
    assert fedot_params['timeout'] == 1
    assert fedot_params['n_jobs'] == -1
    assert isinstance(setup, dict)
    assert setup['task'] == 'ts_classification'
    assert setup['strategy'] == 'fedot_preset'
    assert setup['use_cache'] is False
    assert setup['branch_nodes'] == ['wavelet_basis']
    assert setup['tuning_iters'] == 5
    assert setup['tuning_timeout'] == 15
    assert setup['output_folder'] == 'output_folder'


def test_get_generator_class():
    pass