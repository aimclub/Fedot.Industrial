"""Family-based encoder builders."""

from torch import nn
from fedot_ind.core.models.nn.network_impl.encoders.config import EncoderConfig
from fedot_ind.core.models.nn.network_impl.mapping import ENCODER_BUILDERS_BY_FAMILY


def build_encoder(config: EncoderConfig) -> nn.Module:
    """Build any supported family encoder from config."""

    builder = ENCODER_BUILDERS_BY_FAMILY.get(config.family)
    if builder is None:
        raise ValueError(f"Unsupported encoder family: {config.family}.")
    return builder(config)
