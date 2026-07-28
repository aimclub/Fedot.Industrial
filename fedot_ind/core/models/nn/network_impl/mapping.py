"""Mappings for configurable NN components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fedot_ind.core.models.nn.network_impl.encoders.config import EncoderConfig
from fedot_ind.core.models.nn.network_impl.encoders.transformation_presets import (
    gaf_encoder_config,
    mtf_encoder_config,
    raw_encoder_config,
    stats_encoder_config,
    stft_encoder_config,
)
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.models.nn.models_rules import EncoderFamily
from fedot_ind.core.models.nn.network_impl.encoders.encoders import CNNEncoder, MLPEncoder
from torch import nn

PresetBuilder = Callable[..., EncoderConfig]


class EncoderShapeArg(str, Enum):
    in_channels = "in_channels"
    in_features = "in_features"


def normalize_encoder_shape_arg(
    value: EncoderShapeArg | str,
) -> EncoderShapeArg:
    if isinstance(value, EncoderShapeArg):
        return value
    try:
        return EncoderShapeArg(str(value))
    except ValueError as exc:
        known = [item.value for item in EncoderShapeArg]
        raise ValueError(
            f"Unknown encoder shape argument {value!r}. Known values: {known}."
        ) from exc


@dataclass(frozen=True)
class EncoderPresetEntry:
    """Registry descriptor for one modality encoder preset."""

    builder: PresetBuilder
    shape_arg_name: EncoderShapeArg
    shape_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shape_arg_name",
            normalize_encoder_shape_arg(self.shape_arg_name),
        )

    def build_config(
        self,
        shape: tuple[int, ...],
        *,
        d_model: int,
        kwargs: dict[str, Any] | None = None,
    ) -> EncoderConfig:
        if len(shape) <= self.shape_index:
            raise ValueError(
                f"Cannot resolve {self.shape_arg_name.value} from shape={shape}."
            )
        return self.builder(
            d_model=d_model,
            **{self.shape_arg_name.value: int(shape[self.shape_index])},
            **dict(kwargs or {}),
        )


ENCODER_PRESET_BUILDERS: dict[MultimodalModality, EncoderPresetEntry] = {
    MultimodalModality.raw: EncoderPresetEntry(
        raw_encoder_config,
        EncoderShapeArg.in_channels,
        1,
    ),
    MultimodalModality.stats: EncoderPresetEntry(
        stats_encoder_config,
        EncoderShapeArg.in_features,
        1,
    ),
    MultimodalModality.gaf: EncoderPresetEntry(
        gaf_encoder_config,
        EncoderShapeArg.in_channels,
        1,
    ),
    MultimodalModality.stft: EncoderPresetEntry(
        stft_encoder_config,
        EncoderShapeArg.in_channels,
        1,
    ),
    MultimodalModality.mtf: EncoderPresetEntry(
        mtf_encoder_config,
        EncoderShapeArg.in_channels,
        1,
    ),
}


ENCODER_BUILDERS_BY_FAMILY: dict[EncoderFamily, Callable[[EncoderConfig], nn.Module]] = {
    EncoderFamily.cnn: CNNEncoder,
    EncoderFamily.mlp: MLPEncoder,
}
