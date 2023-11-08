import torch
import torch.nn.functional as F
from torch import nn, optim

from fedot_ind.core.architecture.abstraction.decorators import convert_to_torch_tensor
from fedot_ind.core.models.nn.network_modules.layers.conv_layers import  Conv1d
from fedot_ind.core.models.nn.network_modules.layers.linear_layers import BN1d, Concat, ConvBlock, Add, Noop
from fastcore.meta import delegates

from fastai.torch_core import Module


class SampaddingConv1D_BN(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        self.padding = nn.ConstantPad1d((int((kernel_size - 1) / 2), int(kernel_size / 2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = self.padding(x)
        x = self.conv1d(x)
        x = self.bn(x)
        return x


class ParameterizedLayer(Module):
    """
    formerly build_layer_with_layer_parameter
    """

    def __init__(self, layer_parameters):
        """
        layer_parameters format
            [in_channels, out_channels, kernel_size,
            in_channels, out_channels, kernel_size,
            ..., nlayers
            ]
        """
        self.conv_list = nn.ModuleList()

        for i in layer_parameters:
            # in_channels, out_channels, kernel_size
            conv = SampaddingConv1D_BN(i[0], i[1], i[2])
            self.conv_list.append(conv)

    def forward(self, x):

        conv_result_list = []
        for conv in self.conv_list:
            conv_result = conv(x)
            conv_result_list.append(conv_result)

        result = F.relu(torch.cat(tuple(conv_result_list), 1))
        return result


class InceptionModule(Module):
    def __init__(self,
                 input_dim,
                 number_of_filters,
                 ks=40,
                 bottleneck=True):
        ks = [ks // (2 ** i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if input_dim > 1 else False
        self.bottleneck = Conv1d(input_dim, number_of_filters, 1, bias=False) if bottleneck else Noop
        self.convs = nn.ModuleList([Conv1d(number_of_filters if bottleneck else input_dim,
                                           number_of_filters, k, bias=False) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1),
                                           Conv1d(input_dim, number_of_filters, 1, bias=False)])
        self.concat = Concat()
        self.batch_norm = BN1d(number_of_filters * 4)
        self.activation = nn.ReLU()

    @convert_to_torch_tensor
    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x.float()) for l in self.convs] + [self.maxconvpool(input_tensor.float())])
        return self.activation(self.batch_norm(x))


@delegates(InceptionModule.__init__)
class InceptionBlock(Module):
    def __init__(self,
                 input_dim,
                 number_of_filters=32,
                 residual=False,
                 depth=6,
                 **kwargs):
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(
                InceptionModule(input_dim if d == 0 else number_of_filters * 4, number_of_filters, **kwargs))
            if self.residual and d % 3 == 2:
                n_in, n_out = number_of_filters if d == 2 else number_of_filters * 4, number_of_filters * 4
                self.shortcut.append(BN1d(n_in) if n_in == n_out else ConvBlock(n_in, n_out, 1, act=None))
        self.add = Add()
        self.activation = nn.ReLU()

    @convert_to_torch_tensor
    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            try:
                if self.residual and d % 3 == 2:
                    res = x = self.activation(self.add(x, self.shortcut[d // 3](res)))
            except Exception:
                _ = 1
        return x