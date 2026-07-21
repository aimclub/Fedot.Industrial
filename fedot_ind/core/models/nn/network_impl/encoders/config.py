"""Typed configuration contracts for FUTURE encoders."""

from dataclasses import dataclass

from fedot_ind.core.models.nn.models_rules import (
    EncoderFamily,
    normalize_encoder_family,
    validate_encoder_config,
    validate_conv_block_config,
)


@dataclass(frozen=True)
class ConvBlockConfig:
    """Single CNN block configuration."""

    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    use_pool: bool = True
    pool_kernel_size: int = 2
    pool_stride: int = 2
    pool_ceil_mode: bool = False
    dropout: float = 0.1

    def __post_init__(self) -> None:
        validate_conv_block_config(self)


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for a family-based encoder."""

    family: EncoderFamily | str
    d_model: int
    input_rank: int
    in_channels: int | None = None
    in_features: int | None = None
    conv_blocks: tuple[ConvBlockConfig, ...] = ()
    hidden_dims: tuple[int, ...] = ()
    dropout: float = 0.1

    def __post_init__(self) -> None:
        family = normalize_encoder_family(self.family)
        object.__setattr__(self, "family", family)
        validate_encoder_config(self)
