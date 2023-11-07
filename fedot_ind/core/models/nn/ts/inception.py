from typing import Optional
import numpy as np

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from torch import nn, optim

from fedot_ind.core.architecture.abstraction.decorators import convert_to_torch_tensor
from fedot_ind.core.architecture.settings.computational import default_device
from fedot_ind.core.models.nn.modules import BN1d, Concat, ConvBlock, Add, GAP1d, Noop, Conv1d

from examples.example_utils import evaluate_metric, init_input_data
from fastai.torch_core import Module
from fastcore.meta import delegates

from fedot_ind.core.models.nn.ts.base_nn_model import BaseNeuralModel
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.loader import DataLoader


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


@delegates(InceptionModule.__init__)
class InceptionTime(Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 seq_len=None,
                 number_of_filters=32,
                 nb_filters=None,
                 **kwargs):
        if number_of_filters is None:
            number_of_filters = nb_filters
        self.inceptionblock = InceptionBlock(input_dim, number_of_filters, **kwargs)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(number_of_filters * 4, output_dim)

    def forward(self, x):
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


class InceptionTimeModel(BaseNeuralModel):
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
        self.num_classes = params.get('num_classes', 1)
        self.epochs = params.get('epochs', 10)
        self.batch_size = params.get('batch_size', 20)

    def _init_model(self, ts):
        self.model = InceptionTime(input_dim=ts.features.shape[1],
                                   output_dim=self.num_classes).to(default_device())
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn, optimizer


if __name__ == "__main__":
    train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
    input_data = init_input_data(train_data[0], train_data[1])
    val_data = init_input_data(test_data[0], test_data[1])
    with IndustrialModels():
        pipeline = PipelineBuilder().add_node('inception_model', params={'epochs': 100,
                                                                         'batch_size': 10}).build()
        pipeline.fit(input_data)
        target = pipeline.predict(val_data).predict
        metric = evaluate_metric(target=test_data[1], prediction=target)
    _ = 1
