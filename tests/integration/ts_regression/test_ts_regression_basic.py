import pytest

from tests.integration.integration_test_utils import data, basic_launch


DATASETS = {
    'univariate': 'ItalyPowerDemand',
    'multivariate': 'EthereumSentiment'
}


@pytest.mark.parametrize('type_', ['univariate', 'multivariate'])
def test_basic_reg_test(type_):
    train_data, test_data = data(DATASETS[type_])
    assert train_data is not None and test_data is not None
    basic_launch('regression', train_data, test_data)
