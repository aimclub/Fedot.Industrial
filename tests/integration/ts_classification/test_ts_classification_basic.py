import pytest

from fedot_ind.api.main import FedotIndustrial
from tests.integration.integration_test_utils import data, basic_launch

TASK = 'classification'
DATASETS = {
    'univariate': 'Lightning7',
    'multivariate': 'Epilepsy'
}

@pytest.mark.parametrize('type_', ['univariate', 'multivariate'])
def test_basic_clf_test(type_):
    basic_launch(TASK, *data(DATASETS[type_]))


    # assert train_data is not None and test_data is not None

    # industrial = FedotIndustrial(problem=TASK,
    #                              timeout=0.1,
    #                              n_jobs=-1,
    #                              )

    # industrial.fit(train_data)
    # labels = industrial.predict(test_data)
    # probs = industrial.predict_proba(test_data)
    # assert labels is not None
    # assert probs is not None
    # assert probs.min() >= 0 and probs.max() <= 1
