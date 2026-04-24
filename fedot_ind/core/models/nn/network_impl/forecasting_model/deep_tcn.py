import copy
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    transform_features_and_target_into_lagged
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from torch import nn, optim
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from fedot_ind.core.architecture.abstraction.decorators import convert_inputdata_to_torch_time_series_dataset
from fedot_ind.core.architecture.preprocessing.data_convertor import DataConverter
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot_ind.core.models.nn.network_impl.forecasting_model.common import (
    DEFAULT_FORECASTING_NN_BATCH_SIZE,
    DEFAULT_FORECASTING_NN_DEVICE,
    DEFAULT_FORECASTING_NN_EPOCHS,
    DEFAULT_FORECASTING_NN_LEARNING_RATE,
    build_plateau_scheduler,
    normalize_neural_forecasting_params,
    resolve_neural_forecasting_device,
    resolve_neural_patch_length,
)
from fedot_ind.core.models.nn.network_modules.activation import get_activation_fn
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.repository.constanst_repository import MSE

warnings.filterwarnings("ignore", category=UserWarning)


class _ResidualBlock(nn.Module):
    def __init__(self,
                 num_filters: int,
                 kernel_size: int,
                 dilation_base: int,
                 dropout_fn,
                 activation: str,
                 weight_norm: bool,
                 nr_blocks_below: int,
                 num_layers: int,
                 input_size: int,
                 target_size: int):
        super(_ResidualBlock, self).__init__()
        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below
        self.activation = get_activation_fn(activation)

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(input_dim,
                               num_filters,
                               kernel_size,
                               dilation=(dilation_base ** nr_blocks_below))
        self.conv2 = nn.Conv1d(num_filters,
                               output_dim,
                               kernel_size,
                               dilation=(dilation_base ** nr_blocks_below))
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(self.conv1), nn.utils.weight_norm(self.conv2)
        if nr_blocks_below == 0 or nr_blocks_below == num_layers - 1:
            self.conv3 = nn.Conv1d(input_dim,
                                   output_dim,
                                   kernel_size=1)

    def forward(self, x):
        residual = x

        # first step
        left_padding = (self.dilation_base ** self.nr_blocks_below) * (self.kernel_size - 1)
        x = F.pad(x, (left_padding, 0))
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout_fn(x)

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = self.activation(x)
        x = self.dropout_fn(x)

        # add residual
        if self.nr_blocks_below in {0, self.num_layers - 1}:
            residual = self.conv3(residual)
        x += residual
        return x


class TCNModule(nn.Module):
    def __init__(self,
                 input_size: int,
                 input_chunk_length: int,
                 kernel_size: int,
                 num_filters: int,
                 num_layers: Optional[int],
                 dilation_base: int,
                 weight_norm: bool,
                 target_size: int,
                 target_length: int,
                 dropout: float,
                 activation: str):
        super(TCNModule, self).__init__()

        # Defining parameters
        self.input_size = input_size
        self.input_chunk_length = input_chunk_length
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.target_length = target_length
        self.target_size = target_size
        self.dilation_base = dilation_base
        self.dropout = nn.Dropout(p=dropout)

        # If num_layers is not passed, compute number of layers needed for full
        # history coverage
        if num_layers is None and dilation_base > 1:
            num_layers = math.ceil(math.log(
                (input_chunk_length - 1) * (dilation_base - 1) / (kernel_size - 1) / 2 + 1, dilation_base))
        elif num_layers is None:
            num_layers = math.ceil(
                (input_chunk_length - 1) / (kernel_size - 1) / 2)
        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = _ResidualBlock(num_filters,
                                       kernel_size,
                                       dilation_base,
                                       self.dropout,
                                       activation,
                                       weight_norm,
                                       i,
                                       num_layers,
                                       self.input_size,
                                       target_size)
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)

    def forward(self, x):
        # data is of size (batch_size, input_chunk_length, input_size)
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        for res_block in self.res_blocks_list:
            x = res_block(x)
        x = x.transpose(1, 2)
        x = x.view(batch_size, self.input_chunk_length, self.target_size)
        return x


class TCNModel(BaseNeuralModel):
    def __init__(self, params: Optional[OperationParameters] = None):
        normalized_params = normalize_neural_forecasting_params(dict(params or {}))
        super().__init__(normalized_params)
        self.epochs = self.params.get("epochs", DEFAULT_FORECASTING_NN_EPOCHS)
        self.batch_size = self.params.get("batch_size", DEFAULT_FORECASTING_NN_BATCH_SIZE)
        self.activation = self.params.get('activation', 'ReLU')
        self.learning_rate = self.params.get("learning_rate", DEFAULT_FORECASTING_NN_LEARNING_RATE)
        self.device_name = str(self.params.get('device', DEFAULT_FORECASTING_NN_DEVICE))
        self.device = resolve_neural_forecasting_device(self.device_name)
        self.kernel_size = self.params.get("kernel_size", 3)
        self.num_filters = self.params.get("num_filters", 5)
        self.num_layers = self.params.get("num_layers", 3)
        self.dilation_base = self.params.get("dilation_base", 2)
        self.dropout = self.params.get("dropout", 0.2)
        self.weight_norm = self.params.get("weight_norm", False)
        self.split = self.params.get("split_data", False)
        self.patch_len = self.params.get("patch_len", None)
        self.horizon = self.params.get("horizon", None)
        self.train_log = self.params.get("train_log", False)
        self.scheduler_patience = int(self.params.get('scheduler_patience', 8))
        self.scheduler_factor = float(self.params.get('scheduler_factor', 0.5))
        self.scheduler_min_lr = float(self.params.get('scheduler_min_lr', 1e-5))
        self.early_stopping_patience = int(self.params.get('early_stopping_patience', 12))
        self.early_stopping_min_delta = float(self.params.get('early_stopping_min_delta', 1e-5))
        self.model_list = []

    def _build_model(self, ts: InputData):
        input_size = self.patch_len or ts.features.shape[-1]
        self.patch_len = input_size
        return TCNModule(input_size=input_size,
                         input_chunk_length=ts.features.shape[0],
                         target_size=self.horizon,
                         kernel_size=self.kernel_size,
                         num_filters=self.num_filters,
                         num_layers=self.num_layers,
                         dilation_base=self.dilation_base,
                         target_length=self.patch_len,
                         dropout=self.dropout,
                         activation=self.activation,
                         weight_norm=self.weight_norm).to(self.device)

    def _build_optimizer(self, model: TCNModule):
        return optim.Adam(model.parameters(), lr=self.learning_rate)

    def _build_loss_fn(self):
        return MSE()

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        return build_plateau_scheduler(
            optimizer,
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.scheduler_min_lr,
        )

    def _resolve_patch_len(self, input_data: InputData) -> int:
        return resolve_neural_patch_length(
            input_data.features,
            forecast_horizon=int(max(1, self.horizon)),
            requested_patch_len=self.patch_len,
        )

    def _fit_model(self, input_data: InputData, split_data: bool):
        self.patch_len = self._resolve_patch_len(input_data)
        train_loader = self._prepare_data(input_data.features, self.patch_len, split_data)
        self.test_patch_len = self.patch_len
        model = self._build_model(input_data)
        loss_fn = self._build_loss_fn()
        optimizer = self._build_optimizer(model)
        model = self._train_loop(model, train_loader, loss_fn, optimizer)
        self.model_list.append(model)

    def __preprocess_for_fedot(self, input_data: InputData):
        input_data.features = np.squeeze(input_data.features)
        if self.horizon is None:
            self.horizon = input_data.task.task_params.forecast_length
        if len(input_data.features.shape) == 1:
            input_data.features = input_data.features.reshape(1, -1)
        self.seq_len = input_data.features.shape[1]
        self.target = input_data.target
        self.task_type = input_data.task
        return input_data

    @convert_inputdata_to_torch_time_series_dataset
    def __create_torch_loader(self, train_data: tuple[np.ndarray]):
        return DataLoader(data.TensorDataset(
            train_data.x, train_data.y),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False)

    def fit(self, input_data: InputData):
        input_data = self.__preprocess_for_fedot(input_data)
        self._fit_model(input_data, split_data=self.split)

    def split_data(self, input_data):
        time_series = pd.DataFrame(input_data)
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=self.horizon))
        idx = pd.to_datetime(time_series['datetime'].values) if 'datetime' in time_series.columns \
            else time_series.columns.values
        time_series = time_series.values
        return InputData(idx=idx,
                         features=time_series.flatten(),
                         target=time_series.flatten(),
                         task=task,
                         data_type=DataTypesEnum.ts)

    def _prepare_data(self,
                      ts: np.ndarray,
                      patch_len: int,
                      split_data: bool,
                      validation_blocks: Optional[int] = None):
        train_data = self.split_data(ts)
        if split_data:
            validation_blocks = validation_blocks if validation_blocks is not None else 1
            train_data, val_data = train_test_data_setup(
                train_data, validation_blocks=validation_blocks)
            _, train_data.features, train_data.target = transform_features_and_target_into_lagged(
                train_data, self.horizon, patch_len)
            _, val_data.features, val_data.target = transform_features_and_target_into_lagged(
                val_data, self.horizon, patch_len)
        else:
            _, train_data.features, train_data.target = transform_features_and_target_into_lagged(
                train_data, self.horizon, patch_len)
        train_loader = self.__create_torch_loader(train_data)
        return train_loader

    def _train_loop(self,
                    model: TCNModule,
                    train_loader: DataLoader,
                    loss_fn: nn.MSELoss,
                    optimizer: torch.optim):
        scheduler = self._build_scheduler(optimizer)
        best_state_dict = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        patience_counter = 0
        for epoch in tqdm(range(self.epochs)):
            train_loss_values = []
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_values.append(float(loss.detach().cpu().item()))
            train_loss = float(np.average(train_loss_values)) if train_loss_values else 0.0
            scheduler.step(train_loss)
            if self.train_log:
                if epoch % 25 == 0:
                    print("Epoch: {0} | Train Loss: {1:.7f}".format(
                        epoch + 1, train_loss))
                    print('Updating learning rate to {}'.format(
                        scheduler.get_last_lr()[0]))
            if train_loss + self.early_stopping_min_delta < best_loss:
                best_loss = train_loss
                patience_counter = 0
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            if patience_counter >= self.early_stopping_patience:
                break
        model.load_state_dict(best_state_dict)
        return model

    def _predict(self, model: TCNModule, test_loader: DataLoader):
        model.eval()
        with torch.no_grad():
            if self.forecast_mode == 'in_sample':
                outputs = []
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    outputs.append(model(batch_x))
                return torch.cat(outputs).cpu().numpy()
            else:
                last_patch = test_loader.dataset[0][-1]
                c, s = last_patch.size()
                last_patch = last_patch.reshape(1, c, s).to(self.device)
                outputs = model(last_patch)
                return outputs.cpu().numpy()

    def _predict_model(self,
                       test_data: np.ndarray,
                       test_idx: str = None):
        y_pred = []
        self.forecast_mode = 'out_of_sample'
        for model in self.model_list:
            y_pred.append(self._predict_loop(model, test_data))
        y_pred = np.array(y_pred).squeeze()
        # Workaround for prediction starting point shift
        # TODO: find out what triggers prediction starting point shift
        start_point = self.target[-1]
        shift = y_pred[0] - start_point
        y_pred -= shift
        if test_idx is None:
            forecast_idx_predict = np.arange(
                start=test_data.idx.shape[0],
                stop=test_data.idx.shape[0] +
                self.horizon,
                step=1)
        else:
            forecast_idx_predict = test_idx
        return OutputData(
            idx=forecast_idx_predict,
            task=self.task_type,
            predict=y_pred,
            target=self.target,
            data_type=DataTypesEnum.table)

    def _predict_loop(self,
                      model: TCNModule,
                      test_data: np.ndarray):
        if isinstance(test_data, InputData):
            test_data = test_data.features
        if len(test_data.shape) == 1:
            test_data = test_data.reshape(1, -1)
        features = HankelMatrix(
            time_series=test_data,
            window_size=self.test_patch_len).trajectory_matrix
        features = torch.from_numpy(
            DataConverter(
                data=features).convert_to_torch_format()).float().permute(2, 1, 0)
        target = torch.from_numpy(
            DataConverter(
                data=features).convert_to_torch_format()).float()
        test_loader = DataLoader(
            data.TensorDataset(features, target),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False)
        return self._predict(model, test_loader)
