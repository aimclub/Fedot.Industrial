import os

import numpy as np
import pandas as pd
import pytest

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.tools.loader import DataLoader

EXAMPLES_DATA_PATH = os.path.join(PROJECT_PATH, 'examples', 'data')
MOCK_LOADER = DataLoader(dataset_name='mock')


def test_init_loader():
    ds_name = 'name'
    path = '.'
    loader = DataLoader(dataset_name=ds_name, folder=path)
    assert loader.dataset_name == ds_name
    assert loader.folder == path


def test_load_multivariate_data():
    train_data, test_data = DataLoader('Epilepsy').load_data()
    x_train, y_train = train_data
    x_test, y_test = test_data
    assert x_train.shape == (137, 3, 206)
    assert x_test.shape == (138, 3, 206)
    assert y_train.shape == (137,)
    assert y_test.shape == (138,)


def test_load_univariate_data():
    train_data, test_data = DataLoader('DodgerLoopDay').load_data()
    x_train, y_train = train_data
    x_test, y_test = test_data
    assert x_train.shape == (78, 288)
    assert x_test.shape == (80, 288)
    assert y_train.shape == (78,)
    assert y_test.shape == (80,)


def test_load_fake_data():
    with pytest.raises(FileNotFoundError):
        DataLoader('Fake').load_data()


@pytest.mark.parametrize('dataset_rel_path', (
    'AppliancesEnergy/AppliancesEnergy_TEST.ts'
))
def test__load_from_tsfile_to_dataframe(dataset_rel_path):
    path = os.path.join(EXAMPLES_DATA_PATH, dataset_rel_path)
    x, y = MOCK_LOADER._load_from_tsfile_to_dataframe(full_file_path_and_name=path,
                                                      return_separate_X_and_y=True)
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert x.shape[0] == y.shape[0]


def test_predict_encoding():
    full_path = os.path.join(EXAMPLES_DATA_PATH,
                             'ItalyPowerDemand_fake/ItalyPowerDemand_fake_TEST.ts')
    encoding = MOCK_LOADER.predict_encoding(file_path=full_path)
    assert encoding is not None


@pytest.mark.parametrize('func, dataset_name', [
    (MOCK_LOADER.read_txt_files, 'ItalyPowerDemand_fake'),
    (MOCK_LOADER.read_ts_files, 'ItalyPowerDemand_fake'),
    (MOCK_LOADER.read_arff_files, 'ItalyPowerDemand_fake'),
    (MOCK_LOADER.read_arff_files, 'DailyOilGasPrices'), # multivariate arff
    (MOCK_LOADER.read_train_test_files, 'ItalyPowerDemand_fake'),
    (MOCK_LOADER.read_tsv_or_csv, 'ItalyPowerDemand_fake'),
])
def test_read_train_test_files(func, dataset_name):
    data_path = EXAMPLES_DATA_PATH
    assert np.all(
        [attr is not None for attr in func(dataset_name=dataset_name, data_path=data_path)]
    )
