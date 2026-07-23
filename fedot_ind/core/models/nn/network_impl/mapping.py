"""Mappings for configurable NN components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EncoderPresetEntry:
    """Registry descriptor for one modality encoder preset."""

    builder: PresetBuilder
    shape_arg_name: str
    shape_index: int
    output_kind: str

    def build_config(
        self,
        shape: tuple[int, ...],
        *,
        d_model: int,
        kwargs: dict[str, Any] | None = None,
    ) -> EncoderConfig:
        if len(shape) <= self.shape_index:
            raise ValueError(
                f"Cannot resolve {self.shape_arg_name} from shape={shape}."
            )
        return self.builder(
            d_model=d_model,
            **{self.shape_arg_name: int(shape[self.shape_index])},
            **dict(kwargs or {}),
        )


ENCODER_PRESET_BUILDERS: dict[MultimodalModality, EncoderPresetEntry] = {
    MultimodalModality.raw: EncoderPresetEntry(raw_encoder_config, "in_channels", 1, "raw_series"),
    MultimodalModality.stats: EncoderPresetEntry(stats_encoder_config, "in_features", 1, "tabular_features"),
    MultimodalModality.gaf: EncoderPresetEntry(gaf_encoder_config, "in_channels", 1, "image"),
    MultimodalModality.stft: EncoderPresetEntry(stft_encoder_config, "in_channels", 1, "spectrogram"),
    MultimodalModality.mtf: EncoderPresetEntry(mtf_encoder_config, "in_channels", 1, "image"),
}


ENCODER_BUILDERS_BY_FAMILY: dict[EncoderFamily, Callable[[EncoderConfig], nn.Module]] = {
    EncoderFamily.cnn: CNNEncoder,
    EncoderFamily.mlp: MLPEncoder,
}
