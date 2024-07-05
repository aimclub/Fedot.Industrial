from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from typing import Optional, Callable, Any, List, Union
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data import InputData, OutputData
from fedot_ind.core.repository.constanst_repository import CROSS_ENTROPY, MULTI_CLASS_CROSS_ENTROPY, RMSE
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.architecture.abstraction.decorators import convert_to_3d_torch_array
import pandas as pd
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot_ind.core.models.nn.network_modules.layers.special import adjust_learning_rate, EarlyStopping
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot_ind.core.architecture.preprocessing.data_convertor import DataConverter
import torch.utils.data as data
from fedot_ind.core.architecture.settings.computational import default_device

class SqueezeExciteBlock(nn.Module):
        def __init__(self, input_channels, filters, reduce=4):
            super().__init__()
            self.filters = filters
            self.pool = nn.AvgPool1d(input_channels)
            self.bottleneck = max(self.filters // reduce, 4)
            self.fc1 = nn.Linear(self.filters, self.bottleneck, bias=False)
            self.fc2 = nn.Linear(self.bottleneck, self.filters, bias=False)
            torch.nn.init.kaiming_normal_(self.fc1.weight.data)
            torch.nn.init.kaiming_normal_(self.fc2.weight.data)

        def forward(self, x):
            input_x = x
            x = self.pool(x)
            x = F.relu(self.fc1(x.view(-1, 1, self.filters)))
            x = F.sigmoid(self.fc2(x))
            x = x.view(-1, self.filters, 1) * input_x
            return x

class MLSTM_module(nn.Module):
    def __init__(self, input_size, input_channels,
                 inner_size, inner_channels, 
                 output_size, num_layers, dropout=0.25):
        super().__init__()
        self.proj = nn.Linear(input_size * inner_channels + input_channels * inner_size, output_size)
        self.lstm = nn.LSTM(input_size, inner_size, num_layers,
                             batch_first=True, dropout=dropout)
        self.conv_branch = nn.Sequential(
            nn.Conv1d(input_channels, inner_channels,
                      padding='same',
                      kernel_size=9), 
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            SqueezeExciteBlock(input_size, inner_channels),
            nn.Conv1d(inner_channels, inner_channels * 2,
                      padding='same',
                      kernel_size=5,
                      ), # c x l | n x c x l
            nn.BatchNorm1d(inner_channels * 2), # n x c | n x c x l
            nn.ReLU(),
            SqueezeExciteBlock(input_size, inner_channels * 2),
            nn.Conv1d(inner_channels * 2, inner_channels,
                      padding='same',
                      kernel_size=3,
                      ), # c x l | n x c x l
            nn.BatchNorm1d(inner_channels), # n x c | n x c x l
            nn.ReLU(),
        )
        seq = next(iter(self.conv_branch.modules()))
        idx = [0, 4, 8]
        for i in idx:
            torch.nn.init.kaiming_uniform_(seq[i].weight.data)

    def forward(self, x):
        x_lstm, _ = self.lstm(x) # n x input_ch x inner_size
        x_conv = self.conv_branch(x) # n x inner_ch x len
        print(x_conv.size(), x_lstm.size())
        x = torch.cat([torch.flatten(x_lstm, start_dim=1), torch.flatten(x_conv, start_dim=1)], dim=-1)
        x = F.softmax(self.proj(x))
        return x


class MLSTM(BaseNeuralModel):
    def __init__(self, params: Optional[OperationParameters] = None):
        if params is None:
            params = {}    
        super().__init__()
        # self.num_classes = params.get('num_classes', None)
        # self.epochs = params.get('epochs', 100)
        # self.batch_size = params.get('batch_size', 16)
        # self.activation = params.get('activation', 'ReLU')
        # self.learning_rate = 0.001

        self.dropout = params.get('dropout', 0.25)
        self.hidden_size = params.get('hidden_size', 64)
        self.hidden_channels = params.get('hidden_channels', 32)
        self.num_layers = params.get('num_layers', 2)
        # self.target = None
        # self.task_type = None

    def _init_model(self, ts: InputData):
        _, input_channels, input_size = ts.features.shape
        self.model = MLSTM_module(input_size, input_channels,
                                  self.hidden_size, self.hidden_channels,
                                  self.num_classes, self.num_layers,
                                  self.dropout)
        self.model_for_inference = MLSTM_module(input_size, input_channels,
                                  self.hidden_size, self.hidden_channels,
                                  self.num_classes, self.num_layers,
                                  self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if ts.num_classes == 2:
            loss_fn = CROSS_ENTROPY()
        else:
            loss_fn = MULTI_CLASS_CROSS_ENTROPY()
        return loss_fn, optimizer
    
    @convert_to_3d_torch_array
    def _fit_model(self, ts: InputData):
        loss_fn, optimizer = self._init_model(ts)
        train_loader, val_loader = self._prepare_data(ts, split_data=True)
        self._train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )


from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from typing import Optional, Callable, Any, List, Union
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data import InputData, OutputData
from fedot_ind.core.repository.constanst_repository import CROSS_ENTROPY, MULTI_CLASS_CROSS_ENTROPY, RMSE
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.architecture.abstraction.decorators import convert_to_3d_torch_array
import pandas as pd
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot_ind.core.models.nn.network_modules.layers.special import adjust_learning_rate, EarlyStopping
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot_ind.core.architecture.preprocessing.data_convertor import DataConverter
import torch.utils.data as data
from fedot_ind.core.architecture.settings.computational import default_device

class SqueezeExciteBlock(nn.Module):
        def __init__(self, input_channels, filters, reduce=4):
            super().__init__()
            self.filters = filters
            self.pool = nn.AvgPool1d(input_channels)
            self.bottleneck = max(self.filters // reduce, 4)
            self.fc1 = nn.Linear(self.filters, self.bottleneck, bias=False)
            self.fc2 = nn.Linear(self.bottleneck, self.filters, bias=False)
            torch.nn.init.kaiming_normal_(self.fc1.weight.data)
            torch.nn.init.kaiming_normal_(self.fc2.weight.data)

        def forward(self, x):
            input_x = x
            x = self.pool(x)
            x = F.relu(self.fc1(x.view(-1, 1, self.filters)))
            x = F.sigmoid(self.fc2(x))
            x = x.view(-1, self.filters, 1) * input_x
            return x

class MLSTM_module(nn.Module):
    def __init__(self, input_size, input_channels,
                 inner_size, inner_channels, 
                 output_size, num_layers, dropout=0.25):
        super().__init__()
        self.proj = nn.Linear(input_size * inner_channels + input_channels * inner_size, output_size)
        self.lstm = nn.LSTM(input_size, inner_size, num_layers,
                             batch_first=True, dropout=dropout)
        self.conv_branch = nn.Sequential(
            nn.Conv1d(input_channels, inner_channels,
                      padding='same',
                      kernel_size=9), 
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            SqueezeExciteBlock(input_size, inner_channels),
            nn.Conv1d(inner_channels, inner_channels * 2,
                      padding='same',
                      kernel_size=5,
                      ), # c x l | n x c x l
            nn.BatchNorm1d(inner_channels * 2), # n x c | n x c x l
            nn.ReLU(),
            SqueezeExciteBlock(input_size, inner_channels * 2),
            nn.Conv1d(inner_channels * 2, inner_channels,
                      padding='same',
                      kernel_size=3,
                      ), # c x l | n x c x l
            nn.BatchNorm1d(inner_channels), # n x c | n x c x l
            nn.ReLU(),
        )
        seq = next(iter(self.conv_branch.modules()))
        idx = [0, 4, 8]
        for i in idx:
            torch.nn.init.kaiming_uniform_(seq[i].weight.data)

    def forward(self, x, hidden_state=None):
        x_lstm, hidden_state = self.lstm(x, hidden_state) # n x input_ch x inner_size
        x_conv = self.conv_branch(x) # n x inner_ch x len
        x = torch.cat([torch.flatten(x_lstm, start_dim=1), torch.flatten(x_conv, start_dim=1)], dim=-1)
        x = F.softmax(self.proj(x))
        return x#, hidden_state
    
    def augment_zero_padding(self, X: torch.Tensor):
        res = []
        for i in self.prediction_idx:
            zeroed_X = X[...]
            zeroed_X[..., i + 1:] = 0
            res.append(zeroed_X)
        res = torch.concat(res, 0)
        return res[torch.randperm(res.size(0)), ...]

class MLSTM(BaseNeuralModel):
    def __init__(self, params: Optional[OperationParameters] = None):
        if params is None:
            params = {}    
        super().__init__()
        # self.num_classes = params.get('num_classes', None)
        # self.epochs = params.get('epochs', 100)
        # self.batch_size = params.get('batch_size', 16)
        # self.activation = params.get('activation', 'ReLU')
        # self.learning_rate = 0.001

        self.dropout = params.get('dropout', 0.25)
        self.hidden_size = params.get('hidden_size', 64)
        self.hidden_channels = params.get('hidden_channels', 32)
        self.num_layers = params.get('num_layers', 2)
        # self.target = None
        # self.task_type = None
        self.interval_percentage = params.get('interval_percentage', 10)
        self.min_ts_length = params.get('min_ts_length', 5)

    def _compute_prediction_points(self, n_idx):
        interval_length = max(int(n_idx * self.interval_percentage / 100), self.min_ts_length)
        prediction_idx = np.arange(0, n_idx, interval_length)
        self.earliness = 1 - prediction_idx / n_idx # /n_idx because else the last hm score is always 0
        return prediction_idx

    def _init_model(self, ts: InputData):
        _, input_channels, input_size = ts.features.shape
        self.prediction_idx = self._compute_prediction_points(input_size)
        self.model = MLSTM_module(input_size, input_channels,
                                  self.hidden_size, self.hidden_channels,
                                  self.num_classes, self.num_layers,
                                  self.dropout)
        self.model_for_inference = MLSTM_module(input_size, input_channels,
                                  self.hidden_size, self.hidden_channels,
                                  self.num_classes, self.num_layers,
                                  self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if ts.num_classes == 2:
            loss_fn = CROSS_ENTROPY()
        else:
            loss_fn = MULTI_CLASS_CROSS_ENTROPY()
        return loss_fn, optimizer
    
    def _train_loop(self, train_loader, val_loader, loss_fn, optimizer):
        return super()._train_loop(train_loader, val_loader, loss_fn, optimizer)
    
    @convert_to_3d_torch_array
    def _fit_model(self, ts: InputData):
        if isinstance(ts, torch.Tensor):
            ts = self.augment_zero_padding(ts)
        else:
            print(type(ts))
        loss_fn, optimizer = self._init_model(ts)
        train_loader, val_loader = self._prepare_data(ts, split_data=True)
        self._train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )
    



