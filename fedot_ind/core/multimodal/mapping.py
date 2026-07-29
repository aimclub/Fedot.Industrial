from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fedot_ind.core.models.nn.network_impl.encoders.transformation_presets import (
    gaf_encoder_config,
    raw_encoder_config,
    stats_encoder_config,
    stft_encoder_config,
)
from fedot_ind.core.models.nn.network_impl.mapping import (
    EncoderPresetEntry,
    EncoderShapeArg,
)
from fedot_ind.core.multimodal.enums import (
    MultimodalModality,
    NormalizationStep,
)
from fedot_ind.core.multimodal.rules import (
    ModalitySpec,
    validate_normalization_handler_types,
)
from fedot_ind.core.multimodal.normalization import (
    AbstractNormalizer,
    FeatureStandardizationNormalizer,
    ImageStandardizationNormalizer,
    ImputationNormalizer,
    Log1pNormalizer,
)
from fedot_ind.core.operation.transformation.torch_backend.image.gaf_transformation import GAF
from fedot_ind.core.operation.transformation.torch_backend.image.mtf_transformation import MTF
from fedot_ind.core.operation.transformation.torch_backend.image.stft_transformation import (
    STFTSpectrogram,
)
from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import (
    DEFAULT_CONFIG,
    DEFAULT_GLOBAL_CONFIG,
    TorchQuantileExtractor,
)
from fedot_ind.core.operation.transformation.torch_backend.enums import STAT_FEATURE_CONFIG


DEFAULT_STAT_FEATURE_CONFIG: STAT_FEATURE_CONFIG = dict(DEFAULT_CONFIG)
DEFAULT_STAT_FEATURE_GLOBAL_CONFIG: STAT_FEATURE_CONFIG = dict(DEFAULT_GLOBAL_CONFIG)
DEFAULT_STAT_FEATURES = tuple(
    feature.value for feature in (
        *DEFAULT_STAT_FEATURE_CONFIG.keys(),
        *DEFAULT_STAT_FEATURE_GLOBAL_CONFIG.keys(),
    )
)

NORMALIZATION_HANDLERS = {
    NormalizationStep.imputation: ImputationNormalizer,
    NormalizationStep.feature_standardization: FeatureStandardizationNormalizer,
    NormalizationStep.image_standardization: ImageStandardizationNormalizer,
    NormalizationStep.log1p: Log1pNormalizer,
}

validate_normalization_handler_types(NORMALIZATION_HANDLERS, AbstractNormalizer)

TransformationBuilder = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ModalityCapability:
    """Shared preparation and representation contract for one modality."""

    builder: TransformationBuilder | None
    spec: ModalitySpec
    encoder_preset: EncoderPresetEntry

    def build_transformer(self, params: dict[str, Any]) -> Any:
        if self.builder is None:
            raise ValueError("Modality has no transformation builder.")
        return self.builder(params)


MODALITY_CAPABILITIES: dict[MultimodalModality, ModalityCapability] = {
    MultimodalModality.raw: ModalityCapability(
        builder=None,
        spec=ModalitySpec(allowed_ndim=(3,)),
        encoder_preset=EncoderPresetEntry(
            raw_encoder_config,
            EncoderShapeArg.in_channels,
            1,
        ),
    ),
    MultimodalModality.stats: ModalityCapability(
        builder=TorchQuantileExtractor,
        spec=ModalitySpec(allowed_ndim=(2,)),
        encoder_preset=EncoderPresetEntry(
            stats_encoder_config,
            EncoderShapeArg.in_features,
            1,
        ),
    ),
    MultimodalModality.gaf: ModalityCapability(
        builder=GAF,
        spec=ModalitySpec(allowed_ndim=(4,)),
        encoder_preset=EncoderPresetEntry(
            gaf_encoder_config,
            EncoderShapeArg.in_channels,
            1,
        ),
    ),
    MultimodalModality.stft: ModalityCapability(
        builder=STFTSpectrogram,
        spec=ModalitySpec(allowed_ndim=(4,)),
        encoder_preset=EncoderPresetEntry(
            stft_encoder_config,
            EncoderShapeArg.in_channels,
            1,
        ),
    ),
    MultimodalModality.mtf: ModalityCapability(
        builder=MTF,
        spec=ModalitySpec(allowed_ndim=(4,)),
        encoder_preset=EncoderPresetEntry(
            gaf_encoder_config,
            EncoderShapeArg.in_channels,
            1,
        ),
    ),
}


DEFAULT_MODALITY_SPECS: dict[MultimodalModality, ModalitySpec] = {
    modality: capability.spec
    for modality, capability in MODALITY_CAPABILITIES.items()
}
