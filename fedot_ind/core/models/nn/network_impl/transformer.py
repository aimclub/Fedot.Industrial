from typing import Optional

from fastai.torch_core import Module
from fedot.core.operations.operation_parameters import OperationParameters
from torch import nn, optim
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from fedot_ind.core.architecture.settings.computational import default_device
from .base_nn_model import BaseNeuralModel
from ..network_modules.layers.linear_layers import Max, Permute, Transpose


class TransformerModule(Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 d_model=64,
                 n_head=1,
                 d_ffn=128,
                 dropout=0.1,
                 activation="relu",
                 n_layers=1):
        """
        Args:
            input_dim: the number of features (aka variables, dimensions, channels) in the time series dataset
            output_dim: the number of target classes
            d_model: total dimension of the model.
            n_head: parallel attention layers.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            n_layers: the number of sub-encoder-layers in the encoder.

        Input shape:
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)

        """
        self.permute = Permute(2, 0, 1)
        self.inlinear = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU()
        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout,
                                                activation=activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, n_layers, norm=encoder_norm)
        self.transpose = Transpose(1, 0)
        self.max = Max(1)
        self.outlinear = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.permute(x)  # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x)  # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x)
        # seq_len x bs x d_model -> bs x seq_len x d_model
        x = self.transpose(x)
        x = self.max(x)
        x = self.relu(x)
        x = self.outlinear(x)
        return x


class TransformerModel(BaseNeuralModel):
    """Class responsible for Transformer model implementation.

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
                   pipeline = PipelineBuilder().add_node('tst_model', params={'epochs': 100,
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
        self.model = TransformerModule(input_dim=ts.features.shape[1],
                                       output_dim=self.num_classes).to(default_device())
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn, optimizer
