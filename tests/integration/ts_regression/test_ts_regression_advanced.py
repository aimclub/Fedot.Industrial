import pytest

from itertools import product
from tests.integration.integration_test_utils import launch_api

TASK = 'regression'
DATASETS = {
    'univariate': 'ItalyPowerDemand',
    'multivariate': 'EthereumSentiment'
}
STRATEGIES = [
    'kernel_automl', 
    'federated_automl'
]

@pytest.mark.parametrize('type_,strategy', list((product(DATASETS.keys(), STRATEGIES))))
def test_regr_advanced(type_, strategy):
    launch_api(TASK, strategy, DATASETS[type_])

    
