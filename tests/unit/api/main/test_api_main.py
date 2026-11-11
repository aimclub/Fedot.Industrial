import warnings
from copy import deepcopy

import numpy as np
import pytest
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from matplotlib import get_backend, pyplot as plt

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.repository.config_repository import DEFAULT_CLF_AUTOML_CONFIG, DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_REG_AUTOML_CONFIG
from fedot_ind.tools.synthetic.synth_ts_data import SynthTimeSeriesData
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator
from fedot_ind.integration.fedot.extensions import industrial_extension_scope

from fedot.core.pipelines.pipeline import Pipeline


def univariate_clf_data():
    generator = TimeSeriesDatasetsGenerator(task='classification',
                                            num_samples=16, max_ts_len=32,
                                            binary=True,
                                            multivariate=False)
    train_data, test_data = generator.generate_data()

    return train_data


def univariate_regression_data():
    generator = TimeSeriesDatasetsGenerator(task='regression',
                                            num_samples=16, max_ts_len=32,
                                            binary=True,
                                            multivariate=False)
    train_data, test_data = generator.generate_data()

    return train_data


def multivariate_clf_data():
    generator = TimeSeriesDatasetsGenerator(task='classification',
                                            num_samples=16, max_ts_len=32,
                                            binary=True,
                                            multivariate=True)
    train_data, test_data = generator.generate_data()

    return train_data


def multivariate_regression_data():
    generator = TimeSeriesDatasetsGenerator(task='regression',
                                            num_samples=16, max_ts_len=32,
                                            binary=True,
                                            multivariate=True)
    train_data, test_data = generator.generate_data()

    return train_data


def _bounded_api(task, output_dir):
    model = 'logit' if task == 'classification' else 'ridge'
    metric = 'f1' if task == 'classification' else 'rmse'
    automl = deepcopy(DEFAULT_CLF_AUTOML_CONFIG if task ==
                      'classification' else DEFAULT_REG_AUTOML_CONFIG)
    with industrial_extension_scope():
        pipeline = (PipelineBuilder().add_node('quantile_extractor', params={
            'use_cache': False, 'use_sliding_window': False, 'window_size': 0,
        }).add_node(model).build())
    automl.update(initial_assumption=[pipeline], available_operations=[
                  'quantile_extractor', model])
    compute = deepcopy(DEFAULT_COMPUTE_CONFIG)
    compute.update(output_folder=str(output_dir), automl_folder={
        'optimisation_history': str(output_dir / 'history'),
        'composition_results': str(output_dir / 'models'),
    })
    learning = {'learning_strategy': 'from_scratch', 'learning_strategy_params': {
        'timeout': 0.1, 'logging_level': 50, 'pop_size': 2, 'num_of_generations': 1,
        'n_jobs': 1, 'with_tuning': False, 'seed': 42, 'cv_folds': 2,
        'use_operations_cache': False, 'use_predictions_cache': False,
    }, 'optimisation_loss': {'quality_loss': metric}}
    return FedotIndustrial(industrial_config={'problem': task, 'initial_assumption': [pipeline]},
                           automl_config=automl, learning_config=learning, compute_config=compute)


def fedot_industrial_classification(output_dir):
    return _bounded_api('classification', output_dir)


def fedot_industrial_regression(output_dir):
    return _bounded_api('regression', output_dir)


@pytest.mark.parametrize('metric_names, data_func, fedot_func', (
    [('f1', 'accuracy'), univariate_clf_data, fedot_industrial_classification],
    [('f1', 'accuracy'), multivariate_clf_data, fedot_industrial_classification],
    [('rmse', 'mae'), univariate_regression_data, fedot_industrial_regression],
    [('rmse', 'mae'), multivariate_regression_data, fedot_industrial_regression],
), ids=['clf_uni', 'clf_multi', 'reg_uni', 'reg_multi'])
def test_fit_predict_fedot_industrial(metric_names, data_func, fedot_func, tmp_path):
    data = data_func()
    fedot_industrial = fedot_func(tmp_path)
    fedot_industrial.fit(data)
    predict = fedot_industrial.predict(data)
    predict_proba = fedot_industrial.predict_proba(data)
    metrics = fedot_industrial.get_metrics(predict,
                                           predict_proba,
                                           target=data[1],
                                           metric_names=metric_names)

    fedot_industrial.save()

    for file in ['labels.csv', 'metrics.csv', 'optimization_history.json']:
        assert (tmp_path / file).is_file()

    # search for pipeline
    saved_pipelines = list(tmp_path.glob('*pipeline_saved*'))
    assert saved_pipelines
    loaded_ppl = fedot_industrial.load(str(saved_pipelines[0]))
    assert isinstance(loaded_ppl, Pipeline)

    assert predict.shape[0] == data[1].shape[0]
    assert predict_proba.shape[0] == data[1].shape[0]
    assert metrics is not None
    if len(data[1].shape) > 1:
        assert predict.shape[1] == data[1].shape[1]


@pytest.fixture()
def ts_config():
    return dict(random_walk={'ts_type': 'random_walk',
                             'length': 1000,
                             'start_val': 36.6})


def test_generate_ts(ts_config):
    ts = SynthTimeSeriesData(ts_config).generate_ts()

    assert isinstance(ts, np.ndarray)
    assert ts.shape[0] == 1000


@pytest.fixture()
def anomaly_config():
    return {'dip': {'level': 20,
                    'number': 2,
                    'min_anomaly_length': 10,
                    'max_anomaly_length': 20}
            }


def test_generate_anomaly_ts(ts_config, anomaly_config):
    init_synth_ts, mod_synth_ts, synth_inters = SynthTimeSeriesData(
        anomaly_config).generate_anomaly_ts(ts_config)
    assert len(init_synth_ts) == len(mod_synth_ts)
    for anomaly_type in synth_inters:
        for interval in synth_inters[anomaly_type]:
            ts_range = range(len(init_synth_ts))
            assert interval[0] in ts_range and interval[1] in ts_range


@pytest.mark.parametrize('data_func, fedot_func, node', (
    [univariate_clf_data, fedot_industrial_classification, 'pdl_clf'],
    [multivariate_clf_data, fedot_industrial_classification, 'pdl_clf'],
    [univariate_regression_data, fedot_industrial_regression, 'pdl_reg'],
    [multivariate_regression_data, fedot_industrial_regression, 'pdl_reg'],
), ids=['clf_uni', 'clf_multi', 'reg_uni', 'reg_multi'])
def test_finetune(data_func, fedot_func, node, tmp_path):
    data = data_func()
    fedot_industrial = fedot_func(tmp_path)
    downstream = 'dt' if node == 'pdl_clf' else 'dtreg'
    fedot_industrial.finetune(train_data=data,
                              model_to_tune=PipelineBuilder().add_node(node, params={
                                  'model': downstream, 'max_pairs': 128, 'random_state': 42,
                              }),
                              tuning_params={'tuning_timeout': 0.1, 'tuning_iterations': 1})
    assert fedot_industrial.manager.solver is not None


def test_plot_methods(tmp_path):
    industrial = fedot_industrial_classification(tmp_path)
    data = univariate_clf_data()
    industrial.fit(data)
    industrial.predict(data)
    industrial.predict_proba(data)

    # switch to non-Gui, preventing plots being displayed
    # suppress UserWarning that agg cannot show plots
    get_backend()
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", "Matplotlib is currently using agg")
    industrial.explain()
