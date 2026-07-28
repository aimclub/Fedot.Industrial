from fedot_ind.core.multimodal.enums import (
    MultimodalModality,
    NormalizationMethod,
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
from fedot_ind.core.operation.transformation.torch_backend.enums import (
    StatisticalFeature,
    STAT_FEATURE_CONFIG,
)


DEFAULT_STAT_FEATURE_CONFIG: STAT_FEATURE_CONFIG = dict(DEFAULT_CONFIG)
DEFAULT_STAT_FEATURE_GLOBAL_CONFIG: STAT_FEATURE_CONFIG = dict(DEFAULT_GLOBAL_CONFIG)
DEFAULT_STAT_FEATURES = tuple(
    feature.value for feature in (
        *DEFAULT_STAT_FEATURE_CONFIG.keys(),
        *DEFAULT_STAT_FEATURE_GLOBAL_CONFIG.keys(),
    )
)

NORMALIZATION_HANDLERS = {
    NormalizationMethod.imputation: ImputationNormalizer,
    NormalizationMethod.feature_standardization: FeatureStandardizationNormalizer,
    NormalizationMethod.image_standardization: ImageStandardizationNormalizer,
    NormalizationMethod.log1p: Log1pNormalizer,
}

validate_normalization_handler_types(NORMALIZATION_HANDLERS, AbstractNormalizer)

TRANSFORMATION_HANDLERS = {
    MultimodalModality.stats: TorchQuantileExtractor,
    MultimodalModality.gaf: GAF,
    MultimodalModality.stft: STFTSpectrogram,
    MultimodalModality.mtf: MTF,
}

# raw needs no transformation handler: it is the series the others are built from.
SUPPORTED_PREPARATION_MODALITIES = frozenset(TRANSFORMATION_HANDLERS) | {
    MultimodalModality.raw,
}

DEFAULT_MODALITY_SPECS: dict[MultimodalModality, ModalitySpec] = {
    MultimodalModality.raw: ModalitySpec(allowed_ndim=(3,)),
    MultimodalModality.stats: ModalitySpec(allowed_ndim=(2,)),
    MultimodalModality.gaf: ModalitySpec(allowed_ndim=(4,)),
    MultimodalModality.stft: ModalitySpec(allowed_ndim=(4,)),
    MultimodalModality.mtf: ModalitySpec(allowed_ndim=(4,)),
}
