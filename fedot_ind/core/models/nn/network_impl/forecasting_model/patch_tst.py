import copy
import warnings
from typing import Optional

import pandas as pd
import torch
import torch.utils.data as data
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    transform_features_and_target_into_lagged
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from torch import nn, optim

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
from fedot_ind.core.models.nn.network_modules.layers.backbone import _PatchTST_backbone
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.repository.constanst_repository import EXPONENTIAL_WEIGHTED_LOSS

warnings.filterwarnings("ignore", category=UserWarning)


class PatchTST(nn.Module):
    def __init__(self,
                 input_dim,  # number of input channels
                 output_dim,  # used for compatibility
                 seq_len,  # input sequence length
                 pred_dim=None,  # prediction sequence length
                 n_layers=2,  # number of encoder layers
                 n_heads=8,  # number of heads
                 d_model=512,  # dimension of model
                 d_ff=2048,  # dimension of fully connected network (fcn)
                 dropout=0.05,  # dropout applied to all linear layers in the encoder
                 attn_dropout=0.,  # dropout applied to the attention scores
                 patch_len=16,  # patch_len
                 stride=8,  # stride
                 padding_patch=True,  # flag to indicate if padded is added if necessary
                 revin=True,  # RevIN
                 affine=False,  # RevIN affine
                 individual=False,  # individual head
                 subtract_last=False,  # subtract_last
                 # activation function of intermediate layer, relu or gelu.
                 activation="GELU",
                 norm='BatchNorm',  # type of normalization layer used in the encoder
                 # flag to indicate if normalization is applied as the first
                 # step in the sublayers
                 pre_norm=False,
                 res_attention=True,  # flag to indicate if Residual MultiheadAttention should be used
                 store_attn=True,
                 preprocess_to_lagged=False  # can be used to visualize attention weights
                 ):
        super().__init__()

        # model
        if pred_dim is None:
            pred_dim = seq_len

        self.model = _PatchTST_backbone(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_dim=pred_dim,
            patch_len=patch_len,
            stride=stride,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=activation,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            padding_patch=padding_patch,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            preprocess_to_lagged=preprocess_to_lagged)
        self.patch_num = self.model.patch_num

    def forward(self, x):
        """Args:
            x: rank 3 tensor with shape [batch size x features x sequence length]
        """
        x = self.model(x)
        return x


class PatchTSTModel(BaseNeuralModel):
    """Class responsible for PatchTST model implementation.
    The PatchTST model was proposed in `A Time Series is Worth 64 Words: Long-term Forecasting
    with Transformers by Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong and Jayant Kalagnanam.`

    At a high level the model vectorizes time series into patches of a given size and
    encodes the resulting sequence of vectors via a Transformer that then outputs the
    prediction length forecast via an appropriate head.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
            train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
            input_data = init_input_data(train_data[0], train_data[1])
            val_data = init_input_data(test_data[0], test_data[1])
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('inception_model', params={'epochs': 100,
                                                                                 'batch_size': 10}).build()
                pipeline.fit(input_data)
                target = pipeline.predict(val_data).predict
                metric = evaluate_metric(target=test_data[1], prediction=target)

    References:
        @article{nie2022time,
        title={A time series is worth 64 words: Long-term forecasting with transformers},
        author={Nie, Yuqi and Nguyen, Nam H and Sinthong, Phanwadee and Kalagnanam, Jayant},
        journal={arXiv preprint arXiv:2211.14730},
        year={2022}
        }

        Original paper: https://arxiv.org/pdf/2211.14730.pdf

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        normalized_params = normalize_neural_forecasting_params(dict(params or {}))
        super().__init__(normalized_params)
        self.epochs = self.params.get('epochs', DEFAULT_FORECASTING_NN_EPOCHS)
        self.batch_size = self.params.get('batch_size', DEFAULT_FORECASTING_NN_BATCH_SIZE)
        self.activation = self.params.get('activation', 'GELU')
        self.learning_rate = self.params.get('learning_rate', DEFAULT_FORECASTING_NN_LEARNING_RATE)
        self.device_name = str(self.params.get('device', DEFAULT_FORECASTING_NN_DEVICE))
        self.device = resolve_neural_forecasting_device(self.device_name)
        self.use_amp = self.params.get('use_amp', False)
        self.horizon = self.params.get('forecast_length', None)
        self.patch_len = self.params.get('patch_len', None)
        self.output_attention = self.params.get('output_attention', False)
        self.test_patch_len = self.patch_len
        self.preprocess_to_lagged = False
        self.forecast_mode = self.params.get('forecast_mode', 'out_of_sample')
        self.scheduler_patience = int(self.params.get('scheduler_patience', 8))
        self.scheduler_factor = float(self.params.get('scheduler_factor', 0.5))
        self.scheduler_min_lr = float(self.params.get('scheduler_min_lr', 1e-5))
        self.early_stopping_patience = int(self.params.get('early_stopping_patience', 12))
        self.early_stopping_min_delta = float(self.params.get('early_stopping_min_delta', 1e-5))
        self.model_list = []

    def _resolve_patch_len(self, input_data: InputData) -> int:
        if self.preprocess_to_lagged:
            return int(input_data.features.shape[1])
        return resolve_neural_patch_length(
            input_data.features,
            forecast_horizon=int(max(1, self.horizon)),
            requested_patch_len=self.patch_len,
            multiplier=2.0,
        )

    def _prepare_fit_loader(self, input_data: InputData):
        self.patch_len = self._resolve_patch_len(input_data)
        self.test_patch_len = self.patch_len
        if self.preprocess_to_lagged:
            return self.__create_torch_loader(input_data)
        return self._prepare_data(input_data.features, self.patch_len, False)

    def _build_model(self, ts: InputData):
        return PatchTST(
            input_dim=ts.features.shape[0],
            output_dim=None,
            seq_len=self.seq_len,
            pred_dim=self.horizon,
            patch_len=self.patch_len,
            preprocess_to_lagged=self.preprocess_to_lagged,
            activation=self.activation,
        ).to(self.device)

    def _build_optimizer(self, model: PatchTST):
        return optim.Adam(model.parameters(), lr=self.learning_rate)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        return build_plateau_scheduler(
            optimizer,
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.scheduler_min_lr,
        )

    def _build_loss_fn(self):
        patch_pred_len = round(self.horizon / 4)
        return EXPONENTIAL_WEIGHTED_LOSS(time_steps=patch_pred_len, tolerance=0.3)

    def _build_epoch_iterator(self):
        return tqdm(range(int(max(1, self.epochs))), desc='PatchTST fit', unit='epoch')

    def _run_training_epoch(self, model, train_loader, loss_fn, optimizer) -> float:
        model.train()
        epoch_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(epoch_losses)) if epoch_losses else 0.0

    def _train_loop(self, model, train_loader, loss_fn, optimizer):
        scheduler = self._build_scheduler(optimizer)
        best_state_dict = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch_index in self._build_epoch_iterator():
            train_loss = self._run_training_epoch(model, train_loader, loss_fn, optimizer)
            scheduler.step(train_loss)
            if train_loss + self.early_stopping_min_delta < best_loss:
                best_loss = train_loss
                best_epoch = int(epoch_index) + 1
                patience_counter = 0
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            if hasattr(self, 'train_log') and self.train_log and (epoch_index == 0 or (epoch_index + 1) % 25 == 0):
                print(
                    f'Epoch: {epoch_index + 1}, Train Loss: {train_loss:.7f}, '
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
            if patience_counter >= self.early_stopping_patience:
                break

        model.load_state_dict(best_state_dict)
        self.best_epoch_ = best_epoch
        self.best_loss_ = float(best_loss)
        return model

    def _fit_model(self, input_data: InputData, split_data: bool = True):
        train_loader = self._prepare_fit_loader(input_data)
        model = self._build_model(input_data)
        loss_fn = self._build_loss_fn()
        optimizer = self._build_optimizer(model)
        model = self._train_loop(model, train_loader, loss_fn, optimizer)
        self.model_list.append(model)

    def __preprocess_for_fedot(self, input_data):
        input_data.features = np.squeeze(input_data.features)
        if self.horizon is None:
            self.horizon = input_data.task.task_params.forecast_length
        if len(input_data.features.shape) == 1:
            input_data.features = input_data.features.reshape(1, -1)
        else:
            if input_data.features.shape[1] != 1:
                self.preprocess_to_lagged = True

        if self.preprocess_to_lagged:
            self.seq_len = input_data.features.shape[0] + \
                input_data.features.shape[1]
        else:
            self.seq_len = input_data.features.shape[1]
        self.target = input_data.target
        self.task_type = input_data.task
        return input_data

    def __create_torch_loader(self, train_data):

        train_dataset = self._create_dataset(train_data)
        train_loader = torch.utils.data.DataLoader(
            data.TensorDataset(train_dataset.x, train_dataset.y),
            batch_size=self.batch_size, shuffle=False)
        return train_loader

    def fit(self,
            input_data: InputData):
        """
        Method for feature generation for all series
        """
        input_data = self.__preprocess_for_fedot(input_data)
        self._fit_model(input_data)

    def split_data(self, input_data):

        time_series = pd.DataFrame(input_data)
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=self.horizon))
        if 'datetime' in time_series.columns:
            idx = pd.to_datetime(time_series['datetime'].values)
        else:
            idx = time_series.columns.values

        time_series = time_series.values
        train_input = InputData(idx=idx,
                                features=time_series.flatten(),
                                target=time_series.flatten(),
                                task=task,
                                data_type=DataTypesEnum.ts)

        return train_input

    @convert_inputdata_to_torch_time_series_dataset
    def _create_dataset(self,
                        ts: InputData,
                        flag: str = 'test',
                        batch_size: int = 16,
                        freq: int = 1):
        return ts

    def _prepare_data(self,
                      ts,
                      patch_len,
                      split_data: bool = True,
                      validation_blocks: int = None):
        train_data = self.split_data(ts)
        if split_data:
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

    def _predict(self, model, test_loader):
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
                return outputs.flatten().cpu().numpy()

    def _encoder_decoder_transition(
            self,
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark):
        # encoder - decoder
        if 'Linear' in self.model or 'TST' in self.model:
            outputs = self.model(batch_x)
        else:
            if self.output_attention:
                outputs = self.model(batch_x, batch_x_mark,
                                     dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark,
                                     dec_inp, batch_y_mark)
        return outputs

    def predict(self,
                test_data,
                output_mode: str = 'labels'):
        y_pred = []
        self.forecast_mode = 'out_of_sample'
        for model in self.model_list:
            y_pred.append(self._predict_loop(model, test_data))
        y_pred = np.array(y_pred)
        forecast_idx_predict = np.arange(start=test_data.idx[-self.horizon],
                                         stop=test_data.idx[-self.horizon] +
                                         y_pred.shape[1],
                                         step=1)
        predict = OutputData(
            idx=forecast_idx_predict,
            task=self.task_type,
            predict=y_pred.reshape(1, -1),
            target=self.target,
            data_type=DataTypesEnum.table)
        return predict

    def predict_for_fit(self,
                        test_data,
                        output_mode: str = 'labels'):
        y_pred = []
        self.forecast_mode = 'in_sample'
        for model in self.model_list:
            y_pred.append(self._predict_loop(model, test_data))
        y_pred = np.array(y_pred)
        y_pred = y_pred.squeeze()
        forecast_idx_predict = test_data.idx
        predict = OutputData(
            idx=forecast_idx_predict,
            task=self.task_type,
            predict=y_pred,
            target=self.target,
            data_type=DataTypesEnum.table)
        return predict

    def _predict_loop(self, model,
                      test_data):

        if len(test_data.features.shape) == 1:
            test_data.features = test_data.features.reshape(1, -1)

        if not self.preprocess_to_lagged:
            features = HankelMatrix(
                time_series=test_data.features,
                window_size=self.test_patch_len).trajectory_matrix
            features = torch.from_numpy(
                DataConverter(
                    data=features). convert_to_torch_format()).float().permute(
                2, 1, 0)
            target = torch.from_numpy(DataConverter(
                data=features).convert_to_torch_format()).float()

        else:
            features = test_data.features
            features = torch.from_numpy(DataConverter(data=features).
                                        convert_to_torch_format()).float()
            target = torch.from_numpy(DataConverter(
                data=features).convert_to_torch_format()).float()
        test_loader = torch.utils.data.DataLoader(data.TensorDataset(
            features, target), batch_size=self.batch_size, shuffle=False)
        return self._predict(model, test_loader)
