import torch
import torch.nn as nn

from fedot_ind.core.models.nn.models_rules import EncoderFamily
from fedot_ind.core.models.nn.network_impl.encoders.config import (
    EncoderConfig,
)
from fedot_ind.core.models.nn.network_modules.activation import get_activation_fn


def _activation() -> nn.Module:
    return get_activation_fn("GELU")


class CNNEncoder(nn.Module):
    """Config-driven CNN encoder for 1D and 2D inputs."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        if config.family is not EncoderFamily.cnn:
            raise ValueError("CNNEncoder requires family='cnn'.")

        self.config = config
        is_2d = config.input_rank == 4

        conv_cls = nn.Conv2d if is_2d else nn.Conv1d
        norm_cls = nn.BatchNorm2d if is_2d else nn.BatchNorm1d
        pool_cls = nn.MaxPool2d if is_2d else nn.MaxPool1d

        layers: list[nn.Module] = []
        current_channels = int(config.in_channels or 0)

        for block in config.conv_blocks:
            padding = block.kernel_size // 2
            layers.append(
                conv_cls(
                    in_channels=current_channels,
                    out_channels=block.out_channels,
                    kernel_size=block.kernel_size,
                    stride=block.stride,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(norm_cls(block.out_channels))
            layers.append(_activation())
            if block.use_pool:
                layers.append(
                    pool_cls(
                        kernel_size=block.pool_kernel_size,
                        stride=block.pool_stride,
                        ceil_mode=block.pool_ceil_mode,
                    )
                )
            layers.append(nn.Dropout(block.dropout))
            current_channels = block.out_channels

        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) if is_2d else nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(current_channels, config.d_model),
            nn.LayerNorm(config.d_model),
            _activation(),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.config.input_rank:
            raise ValueError(
                f"CNN encoder expected rank {self.config.input_rank}, got {x.ndim}."
            )
        z = self.cnn(x)
        pooled = self.global_pool(z)
        pooled = pooled.flatten(1)
        return self.projection(pooled)


class MLPEncoder(nn.Module):
    """Config-driven MLP encoder for tabular features."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        if config.family is not EncoderFamily.mlp:
            raise ValueError("MLPEncoder requires family='mlp'.")

        self.config = config
        in_features = int(config.in_features or 0)

        layers: list[nn.Module] = [nn.LayerNorm(in_features)]
        current_features = in_features
        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_features, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    _activation(),
                    nn.Dropout(config.dropout),
                ]
            )
            current_features = hidden_dim

        layers.extend(
            [
                nn.Linear(current_features, config.d_model),
                nn.LayerNorm(config.d_model),
                _activation(),
                nn.Dropout(config.dropout),
            ]
        )
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.config.input_rank:
            raise ValueError(
                f"MLP encoder expected rank {self.config.input_rank}, got {x.ndim}."
            )
        return self.mlp(x)
