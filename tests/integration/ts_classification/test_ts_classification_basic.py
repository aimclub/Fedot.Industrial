import pytest

from tests.integration.integration_test_utils import data, basic_launch

TASK = 'classification'
DATASETS = {
    'univariate': 'Lightning7',
    'multivariate': 'Epilepsy'
}


@pytest.mark.parametrize('type_', ['univariate', 'multivariate'])
def test_basic_clf_test(type_):
    probs = basic_launch(TASK, *data(DATASETS[type_]))[1]
    assert probs.min() >= 0. and probs.max() <= 1
