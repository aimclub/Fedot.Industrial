"""Legacy wrapper for transformation encoders."""

from fedot_ind.core.models.nn.network_impl.encoders.builder import build_encoder
from fedot_ind.core.models.nn.network_impl.encoders.transformation_presets import (
    gaf_encoder_config,
    raw_encoder_config,
    stats_encoder_config,
    stft_encoder_config,
)


def GAFEncoder(
    in_channels: int = 1,
    d_model: int = 128,
    hidden_channels: tuple[int, ...] = (32, 64, 128),
    dropout: float = 0.1,
):
    """Build GAF encoder using family-based preset config."""

    config = gaf_encoder_config(
        in_channels=in_channels,
        d_model=d_model,
        hidden_channels=hidden_channels,
        dropout=dropout,
    )
    return build_encoder(config)


def MTFEncoder(
    in_channels: int = 1,
    d_model: int = 128,
    hidden_channels: tuple[int, ...] = (32, 64, 128),
    dropout: float = 0.1,
):
    """Build MTF encoder using GAF-like family-based preset config."""

    config = gaf_encoder_config(
        in_channels=in_channels,
        d_model=d_model,
        hidden_channels=hidden_channels,
        dropout=dropout,
    )
    return build_encoder(config)


def RawTimeSeriesEncoder(
    in_channels: int,
    d_model: int = 128,
    hidden_channels: tuple[int, ...] = (64, 128, 128),
    kernel_size: int = 5,
    dropout: float = 0.1,
):
    """Build raw encoder using family-based preset config."""

    config = raw_encoder_config(
        in_channels=in_channels,
        d_model=d_model,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    return build_encoder(config)


def StatisticalEncoder(
    in_features: int,
    d_model: int = 128,
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.2,
):
    """Build stats encoder using family-based preset config."""

    config = stats_encoder_config(
        in_features=in_features,
        d_model=d_model,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return build_encoder(config)


def STFTEncoder(
    in_channels: int = 1,
    d_model: int = 128,
    hidden_channels: tuple[int, ...] = (32, 64, 128),
    dropout: float = 0.1,
):
    """Build STFT encoder using family-based preset config."""

    config = stft_encoder_config(
        in_channels=in_channels,
        d_model=d_model,
        hidden_channels=hidden_channels,
        dropout=dropout,
    )
    return build_encoder(config)
