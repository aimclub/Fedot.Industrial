import numpy as np
import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.tools.loader import DataLoader


# def test_shit():
#     train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
#
#     industrial = FedotIndustrial(task='ts_classification',
#                                  dataset='Lightning7',
#                                  strategy='fedot_preset',
#                                  branch_nodes=['wavelet_basis'],
#                                  tuning_iterations=2,
#                                  tuning_timeout=2,
#                                  use_cache=False,
#                                  timeout=1,
#                                  n_jobs=-1)
#     model = industrial.fit(features=train_data[0], target=train_data[1])
#     labels = industrial.predict(features=test_data[0], target=test_data[1])
#     probs = industrial.predict_proba(features=test_data[0], target=test_data[1])
#
#     metrics = industrial.get_metrics(target=test_data[1], metric_names=['f1', 'roc_auc', 'accuracy'])
#
#     assert model is not None
#     assert type(labels) is np.ndarray
#     assert type(probs) is np.ndarray
#     assert type(metrics) is dict



@pytest.mark.parametrize('branch_nodes', [['eigen_basis'],
                                          ['wavelet_basis'],
                                          ['fourier_basis'],
                                          ['eigen_basis', 'wavelet_basis'],
                                          ['eigen_basis', 'fourier_basis'],
                                          ['wavelet_basis', 'fourier_basis'],
                                          ['eigen_basis', 'wavelet_basis', 'fourier_basis']]
                         )
def test_api_code_scenario(branch_nodes):
    train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()

    industrial = FedotIndustrial(task='ts_classification',
                                 dataset='Lightning7',
                                 strategy='fedot_preset',
                                 branch_nodes=branch_nodes,
                                 tuning_iterations=5,
                                 tuning_timeout=2,
                                 use_cache=False,
                                 timeout=1,
                                 n_jobs=-1)

    model = industrial.fit(features=train_data[0], target=train_data[1])
    labels = industrial.predict(features=test_data[0], target=test_data[1])
    probs = industrial.predict_proba(features=test_data[0], target=test_data[1])

    metrics = industrial.get_metrics(target=test_data[1], metric_names=['f1', 'roc_auc', 'accuracy'])

    assert model is not None
    assert type(labels) is np.ndarray
    assert type(probs) is np.ndarray
    assert type(metrics) is dict
