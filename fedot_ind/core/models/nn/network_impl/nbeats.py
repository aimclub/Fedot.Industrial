from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch import optim

from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.architecture.settings.computational import default_device

from fedot_ind.core.models.nn.network_modules.losses import SMAPELoss

class NBeatsModel(ModelImplementation):
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

    References:
        @inproceedings{
            Oreshkin2020:N-BEATS,
            title={{N-BEATS}: Neural basis expansion analysis for interpretable time series forecasting},
            author={Boris N. Oreshkin and Dmitri Carpov and Nicolas Chapados and Yoshua Bengio},
            booktitle={International Conference on Learning Representations},
            year={2020},
            url={https://openreview.net/forum?id=r1ecqn4YwB}
        }
        Original paper: https://arxiv.org/abs/1905.10437
        Original code:  https://github.com/ServiceNow/N-BEATS
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.is_generic_architecture = params.get("is_generic_architecture", True)
        # self.loss_fn = params.get("loss_name", None)
        # self.learning_rate = 0.001

        self.n_stacks = params.get("n_stacks", 30)
        self.layers = params.get("layers", 4)
        self.layer_size = params.get("layer_size", 512)

        self.n_trend_blocks = params.get("n_trend_blocks", 3)
        self.n_trend_layers = params.get("n_trend_layers", 4)
        self.trend_layer_size = params.get("trend_layer_size", 2)
        self.degree_of_polynomial = params.get("degree_of_polynomial", 20)

        self.n_seasonality_blocks = params.get("n_seasonality_blocks", 3)
        self.n_seasonality_layers = params.get("n_seasonality_layers", 4)
        self.seasonality_layer_size = params.get("seasonality_layer_size", 2048)
        self.n_of_harmonics = params.get("n_of_harmonics", 1)

    def _init_model(self, ts):
        self.model = NBeats(
            input_dim=ts.features.shape[1],
            output_dim=self.num_classes).to(default_device())
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # In article, you could choose from: MAPE, MASE, SMAPE
        # https://github.com/ServiceNow/N-BEATS/blob/master/experiments/trainer.py#L79
        loss_fn = SMAPELoss
        return loss_fn, optimizer


class NBeats(nn.Module):
    """
    N-Beats Model proposed in https://arxiv.org/abs/1905.10437
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 is_generic_architecture: bool,
                 n_stacks: int,
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

        super().__init__()

        self.blocks = None

        if is_generic_architecture:
            self.stack = _NBeatsStack(
                input_dim=input_dim,
                output_dim=output_dim,
                is_generic_architecture=is_generic_architecture,
                n_stacks=n_stacks,
                # n_blocks=n_blocks,
                # n_layers=n_layers,
                # layer_size=layer_size,
            )
        else:
            # The overall interpretable architecture consists of two stacks:
            # the trend stack is followed by the seasonality stack
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

        self.blocks = nn.ModuleList(self.stacks)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast

        return forecast


class _NBeatsStack(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            is_generic_architecture: bool,
            n_stacks: int,
            n_trend_blocks: int,
            n_trend_layers: int,
            trend_layer_size: int,
            degree_of_polynomial: int,
            n_seasonality_blocks: int,
            n_seasonality_layers: int,
            seasonality_layer_size: int,
            n_of_harmonics: int,
    ):
        self.block = None

        if is_generic_architecture:
            self.block = _NBeatsBlock(
                input_size=input_dim,
                theta_size=input_dim + output_dim,
                basis_function=_GenericBasis(
                    backcast_size=input_dim,
                    forecast_size=output_dim
                )
            )
            self.blocks = [self.block for _ in range(n_stacks)]

        else:
            trend_block = _NBeatsBlock(
                input_size=input_dim,
                theta_size=2 * (degree_of_polynomial + 1),
                basis_function=_TrendBasis(
                    degree_of_polynomial=degree_of_polynomial,
                    backcast_size=input_dim,
                    forecast_size=output_dim),
                layers=n_trend_layers,
                layer_size=trend_layer_size,
            )

            seasonality_block = _NBeatsBlock(
                input_size=input_dim,
                theta_size=4 * int(np.ceil(n_of_harmonics / 2 * output_dim) - (n_of_harmonics - 1)),
                basis_function=_SeasonalityBasis(
                    harmonics=n_of_harmonics,
                    backcast_size=input_dim,
                    forecast_size=output_dim),
                    layers=n_seasonality_layers,
                    layer_size=seasonality_layer_size
            )

            self.blocks = [trend_block for _ in range(n_trend_blocks)] + [seasonality_block for _ in range(n_seasonality_blocks)]



class _NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
            self,
            input_size,
            theta_size: int,
            basis_function: nn.Module,
            layers: int,
            layer_size: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=input_size, out_features=layer_size)] +
            [nn.Linear(in_features=layer_size, out_features=layer_size) for _ in range(layers - 1)]
        )

        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)

        return self.basis_function(basis_parameters)


class _GenericBasis(nn.Module):
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


class _TrendBasis(nn.Module):
    """
    Polynomial function to model trend.
    Trend model. A typical characteristic of trend is that most of the time it is a monotonic function,
    or at least a slowly varying function. In order to mimic this behaviour constrain g^b_sl, and g^f_sl,
    to be a polynomial of small degree p, a function slowly varying across forecast window.
    """
    def __init__(
            self,
            degree_of_polynomial: int,
            backcast_size: int,
            forecast_size: int
    ):
        super().__init__()

        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = nn.Parameter(
            torch.tensor(np.concatenate(
                [np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :] for i in range(self.polynomial_size)]),
                dtype=torch.float32),
            requires_grad=False
        )

        self.forecast_time = nn.Parameter(
            torch.tensor(np.concatenate(
                [np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :] for i in range(self.polynomial_size)]),
                dtype=torch.float32),
            requires_grad=False
        )


    def forward(self, theta: torch.Tensor):
        backcast = torch.einsum(
            "bp,pt->bt",
            theta[:, self.polynomial_size:],
            self.backcast_time
        )

        forecast = torch.einsum(
            "bp,pt->bt",
            theta[:, :self.polynomial_size],
            self.forecast_time
        )

        return backcast, forecast


class _SeasonalityBasis(_NBeatsBlock):
    """
    Harmonic functions to model seasonality.
    Seasonality model. Typical characteristic of seasonality is that it is a regular, cyclical, recurring fluctuation.
    Therefore, to model seasonality, constrain g^b_sl, and g^f_sl, to be long to the class of periodic functions,
    i.e. y_t = y_t-∆, where ∆ is a seasonality period.
    """
    def __init__(
            self,
            harmonics: int,
            backcast_size: int,
            forecast_size: int
    ):
        super().__init__()

        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency

        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency

        self.backcast_cos_template = nn.Parameter(
            torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32),
            requires_grad=False
        )

        self.backcast_sin_template = nn.Parameter(
            torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32),
            requires_grad=False
        )

        self.forecast_cos_template = nn.Parameter(
            torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32),
            requires_grad=False
        )

        self.forecast_sin_template = nn.Parameter(
            torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32),
            requires_grad=False
        )

    def forward(self, theta: torch.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = torch.einsum(
            "bp,pt->bt",
            theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
            self.backcast_cos_template
        )

        backcast_harmonics_sin = torch.einsum(
            "bp,pt->bt",
            theta[:, 3 * params_per_harmonic:],
            self.backcast_sin_template
        )

        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = torch.einsum(
            "bp,pt->bt",
            theta[:, :params_per_harmonic],
            self.forecast_cos_template
        )

        forecast_harmonics_sin = torch.einsum(
            "bp,pt->bt",
            theta[:, params_per_harmonic:2 * params_per_harmonic],
            self.forecast_sin_template
        )

        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast