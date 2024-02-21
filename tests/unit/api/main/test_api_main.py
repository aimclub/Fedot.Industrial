import warnings

import numpy as np
import pytest
from matplotlib import get_backend, pyplot as plt

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def univariate_clf_data():
    generator = TimeSeriesDatasetsGenerator(task='classification',
                                            binary=True,
                                            multivariate=False)
    train_data, test_data = generator.generate_data()

    return train_data


def univariate_regression_data():
    generator = TimeSeriesDatasetsGenerator(task='regression',
                                            binary=True,
                                            multivariate=False)
    train_data, test_data = generator.generate_data()

    return train_data


def multivariate_clf_data():
    generator = TimeSeriesDatasetsGenerator(task='classification',
                                            binary=True,
                                            multivariate=True)
    train_data, test_data = generator.generate_data()

    return train_data


def multivariate_regression_data():
    generator = TimeSeriesDatasetsGenerator(task='regression',
                                            binary=True,
                                            multivariate=True)
    train_data, test_data = generator.generate_data()

    return train_data


@pytest.fixture
def fedot_industrial_classification():
    return FedotIndustrial(problem='classification', timeout=0.1)


@pytest.fixture
def fedot_industrial_regression():
    return FedotIndustrial(problem='regression', timeout=0.1)


@pytest.mark.parametrize("data", (
    univariate_clf_data(),
    multivariate_clf_data()
))
def test_fit_predict_classification(fedot_industrial_classification, data):
    fedot_industrial_classification.fit(data)
    predict = fedot_industrial_classification.predict(data)
    predict_proba = fedot_industrial_classification.predict_proba(data)
    metrics = fedot_industrial_classification.get_metrics(target=data[1])
    num_unique = np.unique(data[1])

    assert predict.shape[0] == data[1].shape[0]
    assert predict_proba.shape[0] == data[1].shape[0]
    assert metrics is not None

    if len(num_unique) > 2:
        assert predict_proba.shape[1] == len(num_unique)
    else:
        assert len(predict_proba.shape) == 1


@pytest.mark.parametrize("data", (
    univariate_regression_data(),
    multivariate_regression_data()
))
def test_fit_predict_regression(fedot_industrial_regression, data):
    fedot_industrial_regression.fit(data)
    predict = fedot_industrial_regression.predict(data)

    assert predict.shape[0] == data[1].shape[0]
    if len(data[1].shape) > 1:
        assert predict.shape[1] == data[1].shape[1]
    else:
        assert len(predict.shape) == 1


@pytest.fixture()
def ts_config():
    return dict(random_walk={'ts_type': 'random_walk',
                             'length': 1000,
                             'start_val': 36.6})


def test_generate_ts(fedot_industrial_classification, ts_config):
    industrial = fedot_industrial_classification
    ts = industrial.generate_ts(ts_config=ts_config)

    assert isinstance(ts, np.ndarray)
    assert ts.shape[0] == 1000


@pytest.fixture()
def anomaly_config():
    return {'dip': {'level': 20,
                    'number': 2,
                    'min_anomaly_length': 10,
                    'max_anomaly_length': 20}
            }


def test_generate_anomaly_ts(fedot_industrial_classification, ts_config, anomaly_config):
    industrial = fedot_industrial_classification
    init_synth_ts, mod_synth_ts, synth_inters = industrial.generate_anomaly_ts(ts_data=ts_config,
                                                                               anomaly_config=anomaly_config)
    assert len(init_synth_ts) == len(mod_synth_ts)
    for anomaly_type in synth_inters:
        for interval in synth_inters[anomaly_type]:
            ts_range = range(len(init_synth_ts))
            assert interval[0] in ts_range and interval[1] in ts_range


def test_finetune(fedot_industrial_classification):
    industrial = fedot_industrial_classification
    data = univariate_clf_data()
    industrial.fit(data)
    industrial.finetune(data)
    assert industrial.solver is not None


def test_plot_methods(fedot_industrial_classification):
    industrial = fedot_industrial_classification
    data = univariate_clf_data()
    industrial.fit(data)
    labels = industrial.predict(data)
    probs = industrial.predict_proba(data)

    # switch to non-Gui, preventing plots being displayed
    # suppress UserWarning that agg cannot show plots
    curr_backend = get_backend()
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", "Matplotlib is currently using agg")
    fedot_industrial_classification.explain()
