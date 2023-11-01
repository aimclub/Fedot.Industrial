import numpy as np
import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader


GENERATORS = ['signal', 'quantile', 'recurrence', 'topological']
WINDOWS = [0, 10]


def generator_window_combinations():
    return [(gen, win) for gen in GENERATORS for win in WINDOWS]


@pytest.mark.parametrize('strategy, window_size', generator_window_combinations())
def test_ts_classification(strategy, window_size):
    train_data, test_data = DataLoader('Ham').load_data()
    industrial = FedotIndustrial(task='ts_classification',
                                 dataset='Ham',
                                 strategy=strategy,
                                 use_cache=False,
                                 timeout=1,
                                 n_jobs=-1,
                                 window_size=window_size,
                                 available_operations=['scaling', 'normalization', 'xgboost',
                                                       'rfr', 'rf', 'logit', 'mlp', 'knn',
                                                       'lgbm', 'pca']
                                 )

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
