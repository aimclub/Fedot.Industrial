import pytest

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.architecture.settings.computational import backend_methods as np


def input_data_with_nans():
    array = np.random.rand(10, 10)
    for i in range(10):
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 10)
        array[x, y] = np.nan
    target = np.random.rand(10)
    return array, target


def input_data_with_inf():
    array = np.random.rand(10, 10)
    for i in range(10):
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 10)
        array[x, y] = np.inf
    target = np.random.rand(10)
    return array, target


@pytest.mark.parametrize("input_data", [input_data_with_inf(), input_data_with_nans()])
def test_DataCheck(input_data):
    features, target = input_data
    data_check = DataCheck(input_data=(
        features, target), task='classification')
    clean_data = data_check.check_input_data()
    assert clean_data is not None
