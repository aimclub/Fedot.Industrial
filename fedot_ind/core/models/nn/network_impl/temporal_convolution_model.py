import math
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from torch import nn
import torch.nn.functional as F

from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel


class _ResidualBlock(nn.Module):
    def __init__(self,
                 num_filters: int,
                 kernel_size: int,
                 dilation_base: int,
                 dropout_fn,
                 weight_norm: bool,
                 nr_blocks_below: int,
                 num_layers: int,
                 input_dim: int,
                 target_size: int):

        super(_ResidualBlock, self).__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_dim if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim, num_filters, kernel_size, dilation=(
                dilation_base ** nr_blocks_below))
        self.conv2 = nn.Conv1d(
            num_filters, output_dim, kernel_size, dilation=(
                dilation_base ** nr_blocks_below))
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(
                self.conv1), nn.utils.weight_norm(self.conv2)

        if nr_blocks_below == 0 or nr_blocks_below == num_layers - 1:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        residual = x

        # first step
        left_padding = (self.dilation_base **
                        self.nr_blocks_below) * (self.kernel_size - 1)
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(F.relu(self.conv1(x)))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)

        # add residual
        if self.nr_blocks_below in {0, self.num_layers - 1}:
            residual = self.conv3(residual)
        x += residual

        return x


class _TCNModule(nn.Module):
    def __init__(self,
                 input_size: int,
                 seq_len: int,
                 kernel_size: int,
                 num_filters: int,
                 num_layers: Optional[int],
                 dilation_base: int,
                 weight_norm: bool,
                 target_size: int,
                 target_length: int,
                 dropout: float):

        super(_TCNModule, self).__init__()

        if target_length is None:
            target_length = seq_len

        # Defining parameters
        self.input_size = input_size
        self.seq_len = seq_len
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
                (seq_len - 1) * (dilation_base - 1) / (kernel_size - 1) / 2 + 1, dilation_base))
        elif num_layers is None:
            num_layers = math.ceil((seq_len - 1) / (kernel_size - 1) / 2)
        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = _ResidualBlock(
                num_filters,
                kernel_size,
                dilation_base,
                self.dropout,
                weight_norm,
                i,
                num_layers,
                self.input_size,
                target_size)
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)

    def forward(self, x):
        # data is of size (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        for res_block in self.res_blocks_list:
            x = res_block(x)

        x = x.transpose(1, 2)
        x = x.view(batch_size, self.seq_len, self.target_size)

        return x


class TCNModel(BaseNeuralModel):
    def __init__(self, params: Optional[OperationParameters] = {}):
        self.epochs = params.get('epochs', 10)
        self.batch_size = params.get('batch_size', 16)
        self.seq_len = params.get('forecast_length', None)
        self.kernel_size = 3
        self.num_filters = 3
        self.num_layers = None
        self.dilation_base = 2
        self.dropout = 0.2
        self.weight_norm = False
        self.pred_dim = None

    def _init_model(self, ts):
        input_dim = ts.features.shape[1]
        target_size = None
        return _TCNModule(input_size=input_dim,
                          seq_len=self.seq_len,
                          target_size=target_size,
                          kernel_size=self.kernel_size,
                          num_filters=self.num_filters,
                          num_layers=self.num_layers,
                          dilation_base=self.dilation_base,
                          target_length=self.pred_dim,
                          dropout=self.dropout,
                          weight_norm=self.weight_norm)

    # def _build_train_dataset(self,
    #                         target: Sequence[TimeSeries],
    #                         covariates: Optional[Sequence[TimeSeries]]) -> ShiftedDataset:
    #    return ShiftedDataset(target_series=target,
    #                          covariates=covariates,
    #                          length=self.seq_len,
    #                          shift=self.pred_dim)

    # @random_method
    # def _produce_predict_output(self, input):
    #    if self.likelihood:
    #        output = self.model(input)
    #        return self.likelihood._sample(output)
    #    else:
    #        return self.model(input)

    @property
    def first_prediction_index(self) -> int:
        return -self.pred_dim
