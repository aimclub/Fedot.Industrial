import pytest

from itertools import product
from tests.integration.integration_test_utils import launch_api

TASK = 'classification'
DATASETS = {
    'univariate': 'Lightning7',
    'multivariate': 'Epilepsy'
}
STRATEGIES = ['federated_automl', 'kernel_automl', 'lora_strategy']


@pytest.mark.parametrize('type_,strategy', product(DATASETS.keys(), STRATEGIES))
def test_classification_advanced(type_, strategy):
    launch_api(TASK, strategy, DATASETS[type_])
