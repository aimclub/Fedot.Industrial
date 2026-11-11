import numpy as np
import pytest
import torch
from fedot.core.data.input_data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.models.nn.network_impl.forecasting_model.deepar import DeepAR, DeepARModule


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('Real CUDA is unavailable')
    return torch.device(request.param)


@pytest.fixture
def series():
    values = 0.01 * np.arange(96) + np.sin(np.arange(96) / 5.0)
    return InputData(
        idx=np.arange(len(values)),
        features=values,
        target=values.copy(),
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=8)),
        data_type=DataTypesEnum.ts,
    )


def _params(device, cell_type='GRU'):
    return dict(device=str(device), cell_type=cell_type, epochs=1, patch_len=8,
                batch_size=16, hidden_size=4, prediction_averaging_factor=3)


def _state_tensors(state):
    return state if isinstance(state, tuple) else (state,)


@pytest.mark.parametrize('cell_type', ['GRU', 'LSTM'])
def test_deepar_fit_predict_preserves_input_and_history_device(series, device, cell_type):
    features, target = series.features, series.target
    original_features, original_target = features.copy(), target.copy()
    model = DeepAR(_params(device, cell_type)).fit(series)
    parameter = next(model.model.parameters())
    assert parameter.device.type == device.type
    assert np.isfinite(model.best_loss_)

    loader, _ = model._prepare_data(series, split_data=False, is_train=False)
    state = model.get_initial_hidden_state(loader)
    for tensor in _state_tensors(state):
        assert tensor.device == parameter.device
        assert tensor.dtype == parameter.dtype
        assert not tensor.requires_grad
    for prediction in (model.predict(series).predict, model.predict_for_fit(series).predict):
        assert prediction.shape == (8,)
        assert np.isfinite(prediction).all()

    assert series.features is features
    assert series.target is target
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(target, original_target)


@pytest.mark.parametrize('cell_type', ['GRU', 'LSTM'])
def test_deepar_prediction_uses_relocated_model_not_cached_device(series, device, cell_type):
    model = DeepAR(_params('cpu', cell_type)).fit(series)
    model.model.to(device=device, dtype=torch.float64)
    parameter = next(model.model.parameters())
    loader, _ = model._prepare_data(series, split_data=False, is_train=False)
    state = model.get_initial_hidden_state(loader)
    for tensor in _state_tensors(state):
        assert tensor.device == parameter.device
        assert tensor.dtype == parameter.dtype

    supplied_state = tuple(tensor.cpu().float() for tensor in _state_tensors(state))
    if cell_type != 'LSTM':
        supplied_state = supplied_state[0]
    originals = tuple(tensor.clone() for tensor in _state_tensors(supplied_state))
    forecast = model._predict(series, output_mode='predictions', hidden_state=supplied_state)
    assert forecast.device.type == 'cpu'
    assert forecast.dtype == torch.float64
    assert forecast.shape == (1, 8, 1)
    assert torch.isfinite(forecast).all()
    assert model.get_diagnostics()['device'] == str(parameter.device)
    for tensor, original in zip(_state_tensors(supplied_state), originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_deepar_training_and_validation_use_model_dtype_and_device(series, device):
    model = DeepAR(_params('cpu'))
    loader, _ = model._prepare_data(series, split_data=False)
    model._init_model(series)
    model.model.to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=model.learning_rate)
    trained = model._train_loop(
        model.model, train_loader=loader, val_loader=loader,
        loss_fn=model.loss_fn, optimizer=optimizer,
    )
    parameter = next(trained.parameters())
    assert parameter.device.type == device.type
    assert parameter.dtype == torch.float64
    assert model.best_epoch_ == 1
    assert np.isfinite(model.best_loss_)


@pytest.mark.parametrize('distribution', ['normal', 'cauchy'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_deepar_distribution_buffers_follow_model(device, dtype, distribution):
    quantiles = torch.tensor([0.1, 0.5, 0.9])
    model = DeepARModule(
        cell_type='GRU', input_size=4, hidden_size=3, rnn_layers=1,
        dropout=0.0, distribution=distribution, quantiles=quantiles,
        prediction_averaging_factor=3,
    ).eval()
    prefix = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    original_prefix = prefix.clone()
    model(prefix)
    model.to(device=device, dtype=dtype)
    parameter = next(model.parameters())
    buffers = dict(model.named_buffers())
    assert {'quantiles', 'scaler.means', 'scaler.factors'} <= buffers.keys()
    for tensor in buffers.values():
        assert tensor.device == parameter.device
        assert tensor.dtype == dtype
    raw, state = model(prefix, torch.zeros(1, 1, 3, dtype=torch.float64))
    assert raw.device == state.device == parameter.device
    assert raw.dtype == state.dtype == dtype
    custom_quantiles = torch.tensor([0.2, 0.8], dtype=torch.float64)
    for result in (model.to_quantiles(raw), model.to_quantiles(raw, custom_quantiles),
                   model.to_samples(raw, n_samples=3), model.to_predictions(raw)):
        assert result.device == parameter.device
        assert result.dtype == dtype
        assert torch.isfinite(result).all()
    forecast = model.forecast(prefix, forecast_length=2, output_mode='quantiles')
    assert forecast.device.type == 'cpu'
    assert forecast.dtype == dtype
    assert torch.isfinite(forecast).all()
    torch.testing.assert_close(prefix, original_prefix, rtol=0, atol=0)
    torch.testing.assert_close(quantiles, torch.tensor([0.1, 0.5, 0.9]), rtol=0, atol=0)
    assert custom_quantiles.device.type == 'cpu'
    assert custom_quantiles.dtype == torch.float64


def test_deepar_lagged_input_is_not_mutated(series, device):
    features = np.arange(12 * 8, dtype=float).reshape(12, 8) / 50.0
    target = np.arange(12, dtype=float)[:, None]
    original_features, original_target = features.copy(), target.copy()
    lagged = InputData(idx=np.arange(12), features=features, target=target,
                       task=series.task, data_type=DataTypesEnum.table)
    model = DeepAR({**_params(device), 'batch_size': 4}).fit(lagged)
    assert np.isfinite(model.predict(lagged).predict).all()
    assert lagged.features is features
    assert lagged.target is target
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(target, original_target)


@pytest.mark.parametrize('horizon', [1, 8])
def test_deepar_native_extension_fit_and_predict(series, horizon):
    fedot = pytest.importorskip('fedot')
    pytest.importorskip('fedot.extensions')
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope

    series.task.task_params.forecast_length = horizon
    tensor = fedot.create_data(
        series.features, target=series.target, task=series.task,
        ts_forecast_horizon=horizon, without_target=True,
    )
    features, target = tensor.features.clone(), tensor.target.clone()
    with industrial_extension_scope():
        params = _params('cpu')
        del params['device'], params['prediction_averaging_factor']
        pipeline = PipelineBuilder().add_node('deepar_model', params=params).build()
        fitted = pipeline.fit(tensor)
        prediction = pipeline.predict(tensor)
        implementation = pipeline.root_node.fitted_operation.implementation
        parameter = next(implementation.model_.model_.model.parameters())
        assert parameter.device.type == ('cuda' if torch.cuda.is_available() else 'cpu')
        assert parameter.dtype == torch.float32
        assert implementation.get_diagnostics()['forecast_head']['runtime']['device'] == str(parameter.device)
    assert fitted.predict.shape[-1] == horizon
    assert torch.isfinite(fitted.predict).all()
    assert prediction.predict.shape == (horizon,)
    assert torch.isfinite(prediction.predict).all()
    torch.testing.assert_close(tensor.features, features, rtol=0, atol=0)
    torch.testing.assert_close(tensor.target, target, rtol=0, atol=0)
