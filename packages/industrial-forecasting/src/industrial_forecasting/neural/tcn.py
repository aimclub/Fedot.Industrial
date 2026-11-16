"""TCN layers shared by standalone PyTorch and the Industrial FEDOT adapter."""

import math
from typing import Optional

from torch import nn
import torch.nn.functional as F

from fedot_ind.neural.blocks.activation import get_activation_fn


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
        return x + residual


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
