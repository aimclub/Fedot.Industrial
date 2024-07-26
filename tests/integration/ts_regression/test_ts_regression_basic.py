import pytest

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from tests.integration.integration_test_utils import data


DATASETS = {
    'univariate': 'ItalyPowerDemand',
    'multivariate': 'EthereumSentiment'
}

@pytest.mark.parametrize('type_', ['univariate', 'multivariate'])
def test_basic_reg_test(type_):
    train_data, test_data = data(DATASETS[type_])
    assert train_data is not None and test_data is not None

    industrial = FedotIndustrial(problem='regression',
                                 timeout=0.1,
                                 n_jobs=-1,
                                 )
    industrial.fit(train_data)
    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    assert labels is not None
    assert probs is not None
    assert probs.min() > 0
