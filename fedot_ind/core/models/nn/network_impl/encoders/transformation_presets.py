"""Default preset configs for FUTURE encoders."""

from fedot_ind.core.models.nn.network_impl.encoders.config import (
    ConvBlockConfig,
    EncoderConfig,
    EncoderFamily,
)


def raw_encoder_config(
    in_channels: int,
    d_model: int = 128,
    hidden_channels: tuple[int, ...] = (64, 128, 128),
    kernel_size: int = 5,
    dropout: float = 0.1,
) -> EncoderConfig:
    conv_blocks = tuple(
        ConvBlockConfig(
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            use_pool=False,
            dropout=dropout,
        )
        for channels in hidden_channels
    )
    return EncoderConfig(
        family=EncoderFamily.cnn,
        d_model=d_model,
        input_rank=3,
        in_channels=in_channels,
        conv_blocks=conv_blocks,
        dropout=dropout,
    )


def gaf_encoder_config(
    in_channels: int = 1,
    d_model: int = 128,
    hidden_channels: tuple[int, ...] = (32, 64, 128),
    dropout: float = 0.1,
) -> EncoderConfig:
    conv_blocks = tuple(
        ConvBlockConfig(out_channels=channels, dropout=dropout)
        for channels in hidden_channels
    )
    return EncoderConfig(
        family=EncoderFamily.cnn,
        d_model=d_model,
        input_rank=4,
        in_channels=in_channels,
        conv_blocks=conv_blocks,
        dropout=dropout,
    )


def mtf_encoder_config(
    in_channels: int = 1,
    d_model: int = 128,
    hidden_channels: tuple[int, ...] = (32, 64, 128),
    dropout: float = 0.1,
) -> EncoderConfig:
    """Build an MTF image encoder config.

    The initial architecture mirrors GAF because both modalities are 2D image
    representations, but the preset stays separate so MTF can evolve without
    changing the GAF contract.
    """

    return gaf_encoder_config(
        in_channels=in_channels,
        d_model=d_model,
        hidden_channels=hidden_channels,
        dropout=dropout,
    )


def stft_encoder_config(
    in_channels: int = 1,
    d_model: int = 128,
    hidden_channels: tuple[int, ...] = (32, 64, 128),
    dropout: float = 0.1,
) -> EncoderConfig:
    conv_blocks = tuple(
        ConvBlockConfig(
            out_channels=channels,
            pool_ceil_mode=True,
            dropout=dropout,
        )
        for channels in hidden_channels
    )
    return EncoderConfig(
        family=EncoderFamily.cnn,
        d_model=d_model,
        input_rank=4,
        in_channels=in_channels,
        conv_blocks=conv_blocks,
        dropout=dropout,
    )


def stats_encoder_config(
    in_features: int,
    d_model: int = 128,
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.2,
) -> EncoderConfig:
    return EncoderConfig(
        family=EncoderFamily.mlp,
        d_model=d_model,
        input_rank=2,
        in_features=in_features,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
