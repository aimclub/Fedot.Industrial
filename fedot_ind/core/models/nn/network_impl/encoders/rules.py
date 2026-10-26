"""Compatibility exports for encoder configuration rules."""

from fedot_ind.neural.encoders.rules import (
    normalize_encoder_family,
    validate_conv_block_config,
    validate_encoder_config,
)

__all__ = [
    "normalize_encoder_family", "validate_conv_block_config", "validate_encoder_config",
]
