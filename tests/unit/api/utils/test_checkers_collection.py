from fedot_ind.core.architecture.settings.computational import backend_methods as np

from fedot_ind.api.utils.checkers_collections import DataCheck
import pytest
import pandas as pd


# sample dataframe with nans
@pytest.fixture()
def df_with_nans():
    array = np.random.rand(10, 10)
    for i in range(10):
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 10)
        array[x, y] = np.nan
    return pd.DataFrame(array)


# sample dataframe with infs
@pytest.fixture()
def df_with_infs():
    array = np.random.rand(10, 10)
    for i in range(10):
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 10)
        array[x, y] = np.inf
    return pd.DataFrame(array)


def test_DataCheck(df_with_nans, df_with_infs):
    data_check = DataCheck()
    clean_nans = data_check.check_data(df_with_nans, return_df=True)
    clean_infs = data_check.check_data(df_with_infs, return_df=True)
    assert np.all(np.isfinite(clean_nans))
    assert np.all(np.isfinite(clean_infs))


def test_DataCheck_replace_inf_with_nans(df_with_infs):
    data_check = DataCheck()
    clean_infs = data_check._replace_inf_with_nans(df_with_infs.values)
    assert not np.any(np.isinf(clean_infs))


def test_DataCheck_check_for_nan(df_with_nans):
    data_check = DataCheck()
    clean_nans = data_check._check_for_nan(df_with_nans.values)
    assert not np.any(np.isnan(clean_nans))
