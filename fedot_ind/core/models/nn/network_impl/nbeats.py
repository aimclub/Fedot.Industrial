from typing import Optional

import torch
from torch import nn
from torch import optim

from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.architecture.settings.computational import default_device
from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel


class NBeatsModel(BaseNeuralModel):
    """Class responsible for NBeats model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
            train_data, test_data = DataLoader(dataset_name="Lightning7").load_data()
            input_data = init_input_data(train_data[0], train_data[1])
            val_data = init_input_data(test_data[0], test_data[1])
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node("tst_model",
                                                      params={"epochs": 100,
                                                              "batch_size": 10}
                                                     ) \
                                            .build()
                pipeline.fit(input_data)
                target = pipeline.predict(val_data).predict
                metric = evaluate_metric(target=test_data[1], prediction=target)
       """

    def __init__(self, params: Optional[OperationParameters] = {}):
        # self.epochs = params.get("epochs", 10)
        # self.batch_size = params.get("batch_size", 20)
        self.blocks = params.get("blocks", 20)

    def _init_model(self, ts):
        self.model = NBeats(
            input_dim=ts.features.shape[1],
            output_dim=self.num_classes).to(default_device())
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn, optimizer


class NBeats:
    """
    N-Beats Model proposed in https://arxiv.org/abs/1905.10437
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 is_generic_architecture: bool,
                 n_stacks: int,
                 n_blocks: int,
                 n_layers: int,
                 layer_size: int,
                 n_trend_blocks: int,
                 n_trend_layers: int,
                 trend_layer_size: int,
                 degree_of_polynomial: int,
                 n_seasonality_blocks: int,
                 n_seasonality_layers: int,
                 seasonality_layer_size: int,
                 n_of_harmonics: int,
                 ):
        """
        Args:
            input_dim: the number of features (aka variables, dimensions, channels) in the time series dataset
            output_dim: the number of target classes
            is_generic_architecture: indicating whether the generic architecture of N-BEATS is used.
                If not, the interpretable architecture outlined in the paper (consisting of one trend
                and one seasonality stack with appropriate waveform generator functions)
            n_stacks: the number of: The number of stacks that make up the whole model. Only used if `is_generic_architecture` is set to `True`.
                The interpretable architecture always uses two stacks - one for trend and one for seasonality.
            n_blocks: the number of blocks making up every stack.
            n_layers: the number of fully connected layers preceding the final forking layers in each block of every stack.
            layer_size: Determines the number of neurons that make up each fully connected layer in each block of every stack.
            n_trend_blocks:
                Used if `is_generic_architecture` is set to `False`
            n_trend_layers:
                Used if `is_generic_architecture` is set to `False`
            degree_of_polynomial:
                Used if `is_generic_architecture` is set to `False`
            n_seasonality_blocks
                Used if `is_generic_architecture` is set to `False`
            n_seasonality_layers
                Used if `is_generic_architecture` is set to `False`
            seasonality_layer_size:
                Used if `is_generic_architecture` is set to `False`
            n_of_harmonics:
                Used if `is_generic_architecture` is set to `False`
            dropout: probability to be used in fully connected layers.
            activation: the activation function of intermediate layer, relu or gelu.
        """

        self.stacks_list = list()
        self.stack = None

        if is_generic_architecture:
            self.stack = _NBeatsStack(
                input_dim=input_dim,
                output_dim=output_dim,
                is_generic_architecture=is_generic_architecture,
                n_blocks=n_blocks,
                n_layers=n_layers,
                layer_size=layer_size,
            )
        else:
            # The overall interpretable architecture consists of twostacks:
            # the trend stack is followed by the seasonality stack
            # TODO think how to refactor
            self.stack = _NBeatsStack(
                input_dim=input_dim,
                output_dim=output_dim,
                is_generic_architecture=is_generic_architecture,
                n_trend_blocks=n_trend_blocks,
                n_trend_layers=n_trend_layers,
                trend_layer_size=trend_layer_size,
                degree_of_polynomial=degree_of_polynomial,
                n_seasonality_blocks=n_seasonality_blocks,
                n_seasonality_layers=n_seasonality_layers,
                seasonality_layer_size=seasonality_layer_size,
                n_of_harmonics=n_of_harmonics,
            )

        for i in range(n_stacks):
            self.stacks_list.append(self.stack)


class _NBeatsStack(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            is_generic_architecture: bool,
            n_blocks: int,
            n_layers: int,
            layer_size: int,
            n_trend_blocks: int,
            n_trend_layers: int,
            trend_layer_size: int,
            degree_of_polynomial: int,
            n_seasonality_blocks: int,
            n_seasonality_layers: int,
            seasonality_layer_size: int,
            num_of_harmonics: int,
    ):
        self.block_list = list()
        self.block = None

        if is_generic_architecture:
            self.block = _NBeatsBlock(
            )
        else:

            self.block = _NBeatsBlock(
            )



class _NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
            self,
            input_size,
            output_dim,
            is_generic_architecture,
            n_layers,
            layer_size,
            degree_of_polynomial: int,
            n_seasonality_blocks: int,
            n_seasonality_layers: int,
            seasonality_layer_size: int,
            num_of_harmonics: int,
    ):
        # TODO: conditional creation of a block layers based on Generic | Interpretable architectures

        self.layers = None

        if is_generic_architecture:
            self.layers = _GenericBasis(
                input_size=input_size,
                theta_size=input_size + output_dim,
            )
        else:


class _GenericBasis(_NBeatsBlock):
    """
    Generic basis function.
    The generic architecture does not rely on TS-specific knowledge.
    Set g^b_l and g^f_l to be a linear projection of the previous layer output.
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class _TrendBasis(_NBeatsBlock):
    """
    Polynomial function to model trend.
    Trend model. A typical characteristic of trend is that most of the time it is a monotonic function,
    or at least a slowly varying function. In order to mimic this behaviour constrain g^b_sl, and g^f_sl,
    to be a polynomial of small degree p, a function slowly varying across forecast window.
    """
    # TODO:
    pass


class _SeasonalityBasis(_NBeatsBlock):
    """
    Harmonic functions to model seasonality.
    Seasonality model. Typical characteristic of seasonality is that it is a regular, cyclical, recurring fluctuation.
    Therefore, to model seasonality, constrain g^b_sl, and g^f_sl, to be long to the class of periodic functions,
    i.e. y_t = y_t-∆, where ∆ is a seasonality period.
    """
    # TODO:
    pass