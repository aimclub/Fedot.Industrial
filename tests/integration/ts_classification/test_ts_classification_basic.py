import numpy as np
import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader

# TODO: add tests for topological extractor
# def test_topological_extractor():
#     pass

ZERO_WINDOW = 0
NON_ZERO_WINDOW = 10


@pytest.mark.parametrize('strategy, window_size',
                         [('recurrence', ZERO_WINDOW), ('signal', ZERO_WINDOW), ('quantile', ZERO_WINDOW),
                          ('recurrence', NON_ZERO_WINDOW), ('signal', NON_ZERO_WINDOW), ('quantile', NON_ZERO_WINDOW)])
def test_ts_classification(strategy, window_size):
    train_data, test_data = DataLoader('Ham').load_data()
    industrial = FedotIndustrial(task='ts_classification',
                                 dataset='Ham',
                                 strategy=strategy,
                                 use_cache=False,
                                 timeout=1,
                                 n_jobs=-1,
                                 window_size=window_size)

    model = industrial.fit(features=train_data[0], target=train_data[1])
    labels = industrial.predict(features=test_data[0],
                                target=test_data[1])
    probs = industrial.predict_proba(features=test_data[0],
                                     target=test_data[1])
    metrics = industrial.get_metrics(target=test_data[1],
                                     metric_names=['f1', 'roc_auc', 'accuracy'])
    assert model is not None
    assert isinstance(labels, np.ndarray)
    assert isinstance(probs, np.ndarray)
    assert isinstance(metrics, dict)
    assert all([i in metrics.keys() for i in ['f1', 'roc_auc', 'accuracy']])


def test_recurrence_extractor():
    train_data, test_data = DataLoader('Ham').load_data()
    industrial = FedotIndustrial(task='ts_classification',
                                 dataset='Ham',
                                 strategy='recurrence',
                                 use_cache=False,
                                 timeout=1,
                                 n_jobs=-1)

    model = industrial.fit(features=train_data[0], target=train_data[1])
    labels = industrial.predict(features=test_data[0],
                                target=test_data[1])
    probs = industrial.predict_proba(features=test_data[0],
                                     target=test_data[1])
    metrics = industrial.get_metrics(target=test_data[1],
                                     metric_names=['f1', 'roc_auc', 'accuracy'])
    assert model is not None
    assert isinstance(labels, np.ndarray)
    assert isinstance(probs, np.ndarray)
    assert isinstance(metrics, dict)
    assert all([i in metrics.keys() for i in ['f1', 'roc_auc', 'accuracy']])


# signal extractor
def test_signal_extractor():
    train_data, test_data = DataLoader('Ham').load_data()
    industrial = FedotIndustrial(task='ts_classification',
                                 dataset='Ham',
                                 strategy='signal',
                                 use_cache=False,
                                 timeout=1,
                                 n_jobs=-1)

    model = industrial.fit(features=train_data[0], target=train_data[1])
    labels = industrial.predict(features=test_data[0],
                                target=test_data[1])
    probs = industrial.predict_proba(features=test_data[0],
                                     target=test_data[1])
    metrics = industrial.get_metrics(target=test_data[1],
                                     metric_names=['f1', 'roc_auc', 'accuracy'])
    assert model is not None
    assert isinstance(labels, np.ndarray)
    assert isinstance(probs, np.ndarray)
    assert isinstance(metrics, dict)
    assert all([i in metrics.keys() for i in ['f1', 'roc_auc', 'accuracy']])


# quantile extractor
def test_quantile_extractor():
    train_data, test_data = DataLoader('Ham').load_data()
    industrial = FedotIndustrial(task='ts_classification',
                                 dataset='Ham',
                                 strategy='quantile',
                                 use_cache=False,
                                 timeout=1,
                                 n_jobs=-1)

    model = industrial.fit(features=train_data[0], target=train_data[1])
    labels = industrial.predict(features=test_data[0],
                                target=test_data[1])
    probs = industrial.predict_proba(features=test_data[0],
                                     target=test_data[1])
    metrics = industrial.get_metrics(target=test_data[1],
                                     metric_names=['f1', 'roc_auc', 'accuracy'])
    assert model is not None
    assert isinstance(labels, np.ndarray)
    assert isinstance(probs, np.ndarray)
    assert isinstance(metrics, dict)
    assert all([i in metrics.keys() for i in ['f1', 'roc_auc', 'accuracy']])
