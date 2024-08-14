import pytest

from our_approach.api.main import MainClass
from our_approach.core.architecture.settings.computational import backend_methods as np
from our_approach.tools.loader import DataLoader


def multi_data():
    train_data, test_data = DataLoader(dataset_name='Epilepsy').load_data()
    return train_data, test_data


def uni_data():
    train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
    return train_data, test_data


@pytest.mark.parametrize('data', [multi_data, uni_data])
def test_basic_tsc_test(data):
    train_data, test_data = data()

    industrial = MainClass(problem='classification',
                           timeout=0.1,
                           n_jobs=-1)

    industrial.fit(train_data)
    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    assert labels is not None
    assert probs is not None
    assert np.mean(probs) > 0
