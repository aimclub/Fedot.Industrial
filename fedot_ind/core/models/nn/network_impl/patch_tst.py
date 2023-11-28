import warnings
from typing import Optional
import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    transform_features_and_target_into_lagged
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, TsForecastingParams, Task
from torch import nn, optim

from fedot_ind.core.architecture.abstraction.decorators import convert_inputdata_to_torch_time_series_dataset
from fedot_ind.core.architecture.preprocessing.data_convertor import DataConverter
from fedot_ind.core.architecture.settings.computational import default_device
from fedot_ind.core.architecture.settings.constanst_repository import MSE, SMAPE
from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot_ind.core.models.nn.network_modules.layers.backbone import _PatchTST_backbone
from fedot_ind.core.models.nn.network_modules.layers.special import SeriesDecomposition, \
    EarlyStopping, adjust_learning_rate
from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from torch.optim import lr_scheduler

from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector

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
                 decomposition=False,  # apply decomposition
                 kernel_size=25,  # decomposition kernel size
                 activation="gelu",  # activation function of intermediate layer, relu or gelu.
                 norm='BatchNorm',  # type of normalization layer used in the encoder
                 pre_norm=False,  # flag to indicate if normalization is applied as the first step in the sublayers
                 res_attention=True,  # flag to indicate if Residual MultiheadAttention should be used
                 store_attn=False,  # can be used to visualize attention weights
                 ):

        super().__init__()

        # model
        if pred_dim is None:
            pred_dim = seq_len

        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = SeriesDecomposition(kernel_size)
            self.model_trend = _PatchTST_backbone(input_dim=input_dim, seq_len=seq_len, pred_dim=pred_dim,
                                                  patch_len=patch_len, stride=stride, n_layers=n_layers,
                                                  d_model=d_model,
                                                  n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                                  dropout=dropout, act=activation, res_attention=res_attention,
                                                  pre_norm=pre_norm,
                                                  store_attn=store_attn, padding_patch=padding_patch,
                                                  individual=individual, revin=revin, affine=affine,
                                                  subtract_last=subtract_last)
            self.model_res = _PatchTST_backbone(input_dim=input_dim, seq_len=seq_len, pred_dim=pred_dim,
                                                patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model,
                                                n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                                dropout=dropout, act=activation, res_attention=res_attention,
                                                pre_norm=pre_norm,
                                                store_attn=store_attn, padding_patch=padding_patch,
                                                individual=individual, revin=revin, affine=affine,
                                                subtract_last=subtract_last)
            self.patch_num = self.model_trend.patch_num
        else:
            self.model = _PatchTST_backbone(input_dim=input_dim, seq_len=seq_len, pred_dim=pred_dim,
                                            patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model,
                                            n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                            dropout=dropout, act=activation, res_attention=res_attention,
                                            pre_norm=pre_norm,
                                            store_attn=store_attn, padding_patch=padding_patch,
                                            individual=individual, revin=revin, affine=affine,
                                            subtract_last=subtract_last)
            self.patch_num = self.model.patch_num

    def forward(self, x):
        """Args:
            x: rank 3 tensor with shape [batch size x features x sequence length]
        """
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
        else:
            x = self.model(x)
        return x


class PatchTSTModel(BaseNeuralModel):
    """Class responsible for InceptionTime model implementation.

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

    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.epochs = params.get('epochs', 100)
        self.batch_size = params.get('batch_size', 16)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.use_amp = params.get('use_amp', False)
        self.horizon = params.get('forecast_length', 30)
        self.patch_len = params.get('patch_len', None)
        self.output_attention = params.get('output_attention', False)
        self.test_patch_len = self.patch_len
        self.model_list = []

    def _init_model(self, ts):
        model = PatchTST(input_dim=1,
                         output_dim=None,
                         seq_len=ts.features.shape[0],
                         pred_dim=self.horizon).to(default_device())
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = SMAPE()
        return model, loss_fn, optimizer

    def _fit_model(self, ts: InputData, split_data: bool = True):
        if len(ts.features.shape) == 1:
            ts.features = ts.features.reshape(1, -1)

        for index, sequences in enumerate(ts.features):
            if self.patch_len is None:
                dominant_window_size = WindowSizeSelector(method='dff').get_window_size(sequences)
                patch_len = 2 * dominant_window_size
                self.test_patch_len = patch_len
                train_loader = self._prepare_data(sequences, patch_len, False)
                model, loss_fn, optimizer = self._init_model(ts)
                model = self._train_loop(model, train_loader, loss_fn, optimizer)
                self.model_list.append(model)

    def fit(self,
            input_data: InputData):
        """
        Method for feature generation for all series
        """
        input_data.features = np.squeeze(input_data.features)
        self.target = input_data.target
        self.task_type = input_data.task
        self._fit_model(input_data)

    def split_data(self, input_data):

        time_series = pd.DataFrame(input_data)
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=self.horizon))
        if 'datetime' in time_series.columns:
            idx = pd.to_datetime(time_series['datetime'].values)
        else:
            idx = time_series.index.values

        time_series = time_series.values
        train_input = InputData(idx=idx,
                                features=time_series,
                                target=time_series,
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

    def _prepare_data(self, ts,
                      patch_len,
                      split_data: bool = True,
                      validation_blocks: int = None):
        train_data = self.split_data(ts)
        if split_data:
            train_data, val_data = train_test_data_setup(train_data, validation_blocks=validation_blocks)
            _, train_data.features, train_data.target = transform_features_and_target_into_lagged(train_data,
                                                                                                  self.horizon,
                                                                                                  patch_len)
            _, val_data.features, val_data.target = transform_features_and_target_into_lagged(val_data,
                                                                                              self.horizon,
                                                                                              patch_len)
        else:
            _, train_data.features, train_data.target = transform_features_and_target_into_lagged(train_data,
                                                                                                  self.horizon,
                                                                                                  patch_len)

        train_dataset = self._create_dataset(train_data)
        train_dataset.x = train_dataset.x.permute(0, 2, 1)
        train_dataset.y = train_dataset.y.permute(0, 2, 1)
        train_loader = torch.utils.data.DataLoader(
            data.TensorDataset(train_dataset.x, train_dataset.y),
            batch_size=self.batch_size, shuffle=False)

        return train_loader

    def _train_loop(self, model,
                    train_loader,
                    loss_fn,
                    optimizer):
        train_steps = len(train_loader)
        early_stopping = EarlyStopping()
        model_optim = optimizer
        criterion = loss_fn
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            epochs=self.epochs,
                                            max_lr=self.learning_rate)
        args = {'lradj': 'type2'}

        for epoch in range(self.epochs):
            iter_count = 0
            train_loss = []

            model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(default_device())

                batch_y = batch_y.float().to(default_device())

                # decoder input
                dec_inp = torch.zeros_like(batch_y).float()
                dec_inp = torch.cat([batch_y, dec_inp], dim=1).float().to(default_device())
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        return model

    def _predict(self, model, test_data):
        features = HankelMatrix(time_series=test_data.T,
                                window_size=self.test_patch_len).trajectory_matrix[:, -1:]
        features = torch.from_numpy(DataConverter(data=features).convert_to_torch_format()).float().permute(2, 1, 0)
        target = torch.from_numpy(DataConverter(data=features).convert_to_torch_format()).float()
        test_loader = torch.utils.data.DataLoader(data.TensorDataset(features, target),
                                                  batch_size=1, shuffle=False)
        model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(default_device())
                batch_y = batch_y.float().to(default_device())
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, :, :]).float()
                dec_inp = torch.cat([batch_y[:, :, :], dec_inp], dim=1).float().to(default_device())
                # encoder - decoder
                outputs = model(batch_x)
        return outputs.flatten().cpu().numpy()

    def _encoder_decoder_transition(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        # encoder - decoder
        if 'Linear' in self.model or 'TST' in self.model:
            outputs = self.model(batch_x)
        else:
            if self.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    def predict(self,
                test_data,
                output_mode: str = 'labels'):
        y_pred = []
        test_data.features = test_data.features.squeeze()

        if len(test_data.features.shape) == 1:
            test_data.features = test_data.features.reshape(1, -1)

        for sequences, model in zip(test_data.features, self.model_list):
            if type(model) is np.ndarray:
                y_pred.append(model)
            else:
                y_pred.append(self._predict(model, sequences))

        y_pred = np.array(y_pred)
        predict = OutputData(
            idx=np.arange(len(y_pred)),
            task=self.task_type,
            predict=y_pred,
            target=self.target,
            data_type=DataTypesEnum.table)
        return predict
