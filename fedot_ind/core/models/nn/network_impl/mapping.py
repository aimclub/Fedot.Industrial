"""Mappings for configurable NN components."""

from collections.abc import Callable
from typing import TypeAlias

from fedot_ind.core.models.nn.network_impl.encoders.config import EncoderConfig
from fedot_ind.core.models.nn.network_impl.encoders.transformation_presets import (
    gaf_encoder_config,
    raw_encoder_config,
    stats_encoder_config,
    stft_encoder_config,
)
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.models.nn.models_rules import EncoderFamily
from fedot_ind.core.models.nn.network_impl.encoders.encoders import CNNEncoder, MLPEncoder
from torch import nn

PresetBuilder: TypeAlias = Callable[..., EncoderConfig]


ENCODER_PRESET_BUILDERS: dict[MultimodalModality, tuple[PresetBuilder, str]] = {
    MultimodalModality.raw: (raw_encoder_config, "in_channels"),
    MultimodalModality.stats: (stats_encoder_config, "in_features"),
    MultimodalModality.gaf: (gaf_encoder_config, "in_channels"),
    MultimodalModality.stft: (stft_encoder_config, "in_channels"),
    MultimodalModality.mtf: (gaf_encoder_config, "in_channels"),
}


ENCODER_BUILDERS_BY_FAMILY: dict[EncoderFamily, Callable[[EncoderConfig], nn.Module]] = {
    EncoderFamily.cnn: CNNEncoder,
    EncoderFamily.mlp: MLPEncoder,
}
