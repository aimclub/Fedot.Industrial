import numpy as np

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader


def multi_data():
    train_data, test_data = DataLoader(dataset_name='Epilepsy').load_data()
    return train_data, test_data


def uni_data():
    train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
    return train_data, test_data


def combinations(data, strategy):
    return [[d, s] for d in data for s in strategy]


def test_kernel_automl_strategy_clf():

    dataset_name = 'Lightning7'
    api_config = dict(problem='classification',
                      metric='f1',
                      timeout=5,
                      n_jobs=2,
                      with_tuning=False,
                      industrial_strategy='kernel_automl',
                      industrial_strategy_params={},
                      logging_level=20)
    train_data, test_data = DataLoader(dataset_name).load_data()
    industrial = FedotIndustrial(**api_config)
    industrial.fit(train_data)
    labels = industrial.predict(test_data, 'ensemble')
    probs = industrial.predict_proba(test_data, 'ensemble')

    assert labels is not None
    assert probs is not None
    assert np.mean(probs) > 0


# ['federated_automl',
#  'kernel_automl',
#  'forecasting_assumptions',
#  'forecasting_exogenous']
