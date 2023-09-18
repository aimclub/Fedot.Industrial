from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader

import pytest


@pytest.fixture
def selector():
    return WindowSizeSelector()


@pytest.fixture
def single_ts_data():
    dataset_name = 'Ham'
    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    return train_data[0].values[0]


@pytest.fixture
def multiple_ts_data():
    dataset_name = 'Ham'
    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    return train_data[0].values


def test_dominant_fourier_frequency_single(single_ts_data, selector):
    ts = single_ts_data
    selector = selector()
    selected_window = selector.get_window_size(time_series=ts)

    assert selected_window > 0
