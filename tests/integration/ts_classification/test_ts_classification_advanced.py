import itertools

import numpy as np
import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader


@pytest.fixture
def multi_data():
    train_data, test_data = DataLoader(dataset_name='Epilepsy').load_data()
    return train_data, test_data


@pytest.fixture
def uni_data():
    train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
    return train_data, test_data


@pytest.mark.parametrize('data, strategy',
                         [itertools.combinations([uni_data,
                                                  multi_data],
                                                 ['federated_automl',
                                                  'kernel_automl',
                                                  'forecasting_assumptions',
                                                  'forecasting_exogenous'])])
def strategy_tsc_test(data, strategy):
    train_data, test_data = data

    industrial = FedotIndustrial(task='classification',
                                 timeout=2,
                                 n_jobs=-1,
                                 industrial_strategy=strategy)

    industrial.fit(train_data)
    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    assert labels is not None
    assert probs is not None
    assert np.mean(labels) > 0
    assert np.mean(probs) > 0
