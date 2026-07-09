from fedot_ind.core.multimodal.enums import (
    MultimodalModality,
    NormalizationMethod,
)
from fedot_ind.core.multimodal.normalization import (
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
    TorchQuantileExtractor,
)
from fedot_ind.core.operation.transformation.torch_backend.enums import (
    StatisticalFeature,
    STAT_FEATURE_CONFIG,
)


DEFAULT_STAT_FEATURE_CONFIG: STAT_FEATURE_CONFIG = {
    StatisticalFeature.median: {},
    StatisticalFeature.max: {},
    StatisticalFeature.min: {},
    StatisticalFeature.q5: {},
    StatisticalFeature.q25: {},
    StatisticalFeature.q75: {},
    StatisticalFeature.q95: {},
}

DEFAULT_STAT_FEATURE_GLOBAL_CONFIG: STAT_FEATURE_CONFIG = {
    StatisticalFeature.skewness: {},
    StatisticalFeature.kurtosis: {},
    StatisticalFeature.n_peaks: {"normalized": True},
    StatisticalFeature.slope: {},
    StatisticalFeature.ben_corr: {},
    StatisticalFeature.interquartile_range: {},
    StatisticalFeature.energy: {},
    StatisticalFeature.cross_rate: {},
    StatisticalFeature.autocorrelation: {},
    StatisticalFeature.ptp_amplitude: {},
    StatisticalFeature.mean_ptp_distance: {"normalized": True},
    StatisticalFeature.crest_factor: {},
    StatisticalFeature.mean_ema: {},
    StatisticalFeature.mean_moving_median: {},
    StatisticalFeature.hjorth_mobility: {},
    StatisticalFeature.hjorth_complexity: {},
    StatisticalFeature.hurst_exponent: {},
    StatisticalFeature.petrosian_fractal_dimension: {},
}
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

TRANSFORMATION_HANDLERS = {
    MultimodalModality.stats: TorchQuantileExtractor,
    MultimodalModality.gaf: GAF,
    MultimodalModality.stft: STFTSpectrogram,
    MultimodalModality.mtf: MTF,
}
