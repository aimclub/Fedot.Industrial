import pytest

from itertools import product
from tests.integration.integration_test_utils import data, launch_api

DATASETS = {
    'univariate': 'Lightning7',
    'multivariate': 'Epilepsy'
}
STRATEGIES = ['federated_automl', 'kernel_automl']


@pytest.mark.parametrize('type_,strategy', product(DATASETS.keys(), STRATEGIES))
def test_classification_advanced(type_, strategy):
    train_data, test_data = data(DATASETS[type_])
    launch_api('classification', strategy, train_data, test_data)
