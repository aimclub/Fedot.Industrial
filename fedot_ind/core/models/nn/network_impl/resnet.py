from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastai.torch_core import Module
from fedot.core.data.data import OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from torch import nn, optim, Tensor
import torch.nn.functional as F
from examples.example_utils import evaluate_metric
from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.core.architecture.abstraction.decorators import convert_to_3d_torch_array
from fedot_ind.core.architecture.settings.computational import default_device
from fedot_ind.core.models.cnn.classification_models import CLF_MODELS
from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot_ind.core.models.nn.network_modules.layers.conv_layers import ConvBlock
from fedot_ind.core.models.nn.network_modules.layers.linear_layers import Squeeze, BN1d, Add
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.loader import DataLoader


class ResNet:
    def __init__(self,
                 input_dim,
                 output_dim,
                 model_name):
        nf = 64
        kss = [7, 5, 3]
        self.model = CLF_MODELS[model_name](num_classes=output_dim)
        # self.resblock1 = ResBlock(input_dim, nf, kss=kss)
        # self.resblock2 = ResBlock(nf, nf * 2, kss=kss)
        # self.resblock3 = ResBlock(nf * 2, nf * 2, kss=kss)
        # self.gap = nn.AdaptiveAvgPool1d(1)
        # self.squeeze = Squeeze(-1)
        # self.fc = nn.Linear(nf * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward method of the model and returns predictions."""
        x = x.to(default_device())
        return self.model(x)


class ResNetModel(BaseNeuralModel):
    """Class responsible for ResNet model implementation.

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
        self.epochs = params.get('epochs', 100)
        self.batch_size = params.get('batch_size', 32)
        self.model_name = params.get('model_name', 'ResNet18')

    def _init_model(self, ts):

        self.model = ResNet(input_dim=ts.features.shape[1],
                            output_dim=self.num_classes,
                            model_name=self.model_name)
        self.model = self.model.model
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if ts.num_classes == 2:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn, optimizer

    def _prepare_data(self, ts, split_data: bool = True):
        if split_data:
            train_data, val_data = train_test_data_setup(ts, shuffle_flag=True, split_ratio=0.7)
        train_dataset = self._create_dataset(train_data)
        train_dataset.x = train_dataset.x.permute(0, 3, 1, 2)
        val_dataset = self._create_dataset(val_data)
        val_dataset.x = val_dataset.x.permute(0, 3, 1, 2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, val_loader

    @convert_to_3d_torch_array
    def _predict_model(self, x_test):
        self.model.eval()
        x_test = Tensor(x_test).permute(0, 3, 1, 2).to(default_device())
        pred = self.model(x_test)
        return self._convert_predict(pred)


if __name__ == "__main__":
    dataset_list = [
        'Lightning7'
    ]
    result_dict = {}
    pipeline_dict = {'inception_model': PipelineBuilder().add_node('inception_model', params={'epochs': 100,
                                                                                              'batch_size': 32}),
                     'image_model': PipelineBuilder().add_node('recurrence_extractor', params={'image_mode': True}) \
                         .add_node('resnet_model', params={'epochs': 100,
                                                           'batch_size': 32}),
                     'rocket_model': PipelineBuilder().add_node('minirocket_extractor') \
                         .add_node('fedot_cls', params={'timeout': 10}),
                     'composed_model': PipelineBuilder().add_node('inception_model', params={'epochs': 100,
                                                                                             'batch_size': 32}) \
                         .add_node('recurrence_extractor', params={'image_mode': True}, branch_idx=1) \
                         .add_node('resnet_model', params={'epochs': 100,
                                                           'batch_size': 32}, branch_idx=1).add_node(
                         'minirocket_extractor', branch_idx=2) \
                         .add_node('fedot_cls', params={'timeout': 10}, branch_idx=2).join_branches('logit')}

    for dataset in dataset_list:
        train_data, test_data = DataLoader(dataset_name=dataset).load_data()
        input_data = init_input_data(train_data[0], train_data[1])
        val_data = init_input_data(test_data[0], test_data[1])
        metric_dict = {}
        for model in pipeline_dict:
            with IndustrialModels():
                pipeline = pipeline_dict[model].build()
                pipeline.fit(input_data)
                target = pipeline.predict(val_data).predict
                metric = evaluate_metric(target=test_data[1], prediction=target)
            metric_dict.update({model: metric})
        result_dict.update({dataset: metric_dict})
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv('./resnet_inception.csv')
    _ = 1
