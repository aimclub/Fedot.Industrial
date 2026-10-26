"""Compatibility exports sharing the neutral encoder registry."""

from fedot_ind.neural.encoders.mapping import (
    ENCODER_BUILDERS_BY_FAMILY,
    EncoderPresetEntry,
    EncoderShapeArg,
    PresetBuilder,
    normalize_encoder_shape_arg,
)

__all__ = [
    "ENCODER_BUILDERS_BY_FAMILY", "EncoderPresetEntry", "EncoderShapeArg",
    "PresetBuilder", "normalize_encoder_shape_arg",
]
