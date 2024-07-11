import pytest
import torch
from fedot.core.data.data import InputData, OutputData

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.models.nn.network_impl.deep_tcn import TCNModel
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def dataset():
    (X_train, y_train), (_, _) = TimeSeriesDatasetsGenerator(num_samples=1,
                                                             max_ts_len=50,
                                                             binary=False,
                                                             test_size=0.5,
                                                             task='regression').generate_data()
    return X_train, y_train, _, _


@pytest.fixture
def ts():
    task_params = {'forecast_length': 14}
    X_train, y_train, _, _ = dataset()
    train_data = (X_train, y_train)
    input_train = DataCheck(
        input_data=train_data,
        task='ts_forecasting',
        task_params=task_params).check_input_data()
    return input_train


@pytest.fixture
def tcn():
    return TCNModel({'epochs': 10})


def test_tcn_init(tcn):
    assert tcn is not None


def test_tcn_loader(ts, tcn):
    loader = tcn._TCNModel__create_torch_loader(ts)
    assert loader is not None
    assert isinstance(loader, torch.utils.data.dataloader.DataLoader)


def test_tcn_preprocess(ts, tcn):
    input_data = tcn._TCNModel__preprocess_for_fedot(ts)
    assert input_data is not None
    assert isinstance(input_data, InputData)


def test_tcn_prepare(ts, tcn):
    input_data = tcn._TCNModel__preprocess_for_fedot(ts)
    loader = tcn._prepare_data(
        input_data.features,
        patch_len=14,
        split_data=False)
    assert loader is not None
    assert isinstance(loader, torch.utils.data.dataloader.DataLoader)


def test_tcn_model_init(ts, tcn):
    ts = tcn._TCNModel__preprocess_for_fedot(ts)
    model, loss_fn, optimizer = tcn._init_model(ts=ts)
    assert model is not None
    assert model.input_chunk_length == ts.features.shape[0]
    assert loss_fn is not None
    assert optimizer is not None


def test_tcn_fit(ts, tcn):
    tcn.fit(ts)
    assert tcn.model_list is not None


def test_tcn_predict(ts, tcn):
    tcn.fit(ts)
    predict = tcn.predict(ts)
    assert predict is not None
    assert isinstance(predict, OutputData)
