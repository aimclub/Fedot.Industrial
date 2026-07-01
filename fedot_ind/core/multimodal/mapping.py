from typing import Any

import torch

from fedot_ind.core.multimodal.enums import (
    MultimodalModality,
    NormalizationMethod,
    StatisticalFeature,
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
from fedot_ind.core.operation.transformation.torch_backend.statistical.stat_features import (
    interquantile_range_torch, max_torch, mean_ptp_distance_torch, mean_torch, median_torch,
    min_torch, q25_torch, q75_torch, q5_torch, q95_torch,
    skewness_torch, kurtosis_torch, ben_corr_torch, energy_torch, zero_crossing_rate_torch,
    autocorrelation_torch, ptp_amp_torch, mean_ema_torch, mean_moving_median_torch,
    hjorth_mobility_torch, hjorth_complexity_torch, hurst_exponent_torch, pfd_torch, slope_torch, std_torch,
    crest_factor_torch, n_peaks_torch,
)


# "Window" stat features
STAT_METHODS = {
    "median_": median_torch,
    "max_": max_torch,
    "min_": min_torch,
    "q5_": q5_torch,
    "q25_": q25_torch,
    "q75_": q75_torch,
    "q95_": q95_torch,
}

STAT_METHODS_GLOBAL = {
    "skewness_": skewness_torch,
    "kurtosis_": kurtosis_torch,
    "n_peaks_": lambda array, axis=-1: n_peaks_torch(array, axis=axis, normalized=True),
    "slope_": slope_torch,
    "ben_corr_": ben_corr_torch,
    "interquartile_range_": interquantile_range_torch,
    "energy_": energy_torch,
    "cross_rate_": zero_crossing_rate_torch,
    "autocorrelation_": autocorrelation_torch,
    "ptp_amplitude_": ptp_amp_torch,
    "mean_ptp_distance_": lambda array, axis=-1: mean_ptp_distance_torch(
        array,
        axis=axis,
        normalized=True,
    ),
    "crest_factor_": crest_factor_torch,
    "mean_ema_": mean_ema_torch,
    "mean_moving_median_": mean_moving_median_torch,
    "hjorth_mobility_": hjorth_mobility_torch,
    "hjorth_complexity_": hjorth_complexity_torch,
    "hurst_exponent_": hurst_exponent_torch,
    "petrosian_fractal_dimension_": pfd_torch,
}


STAT_FEATURE_TO_METHOD = {
    StatisticalFeature.mean.value: mean_torch,
    StatisticalFeature.std.value: std_torch,
    StatisticalFeature.median.value: STAT_METHODS["median_"],
    StatisticalFeature.max.value: STAT_METHODS["max_"],
    StatisticalFeature.min.value: STAT_METHODS["min_"],
    StatisticalFeature.q5.value: STAT_METHODS["q5_"],
    StatisticalFeature.q25.value: STAT_METHODS["q25_"],
    StatisticalFeature.q75.value: STAT_METHODS["q75_"],
    StatisticalFeature.q95.value: STAT_METHODS["q95_"],
    StatisticalFeature.skewness.value: STAT_METHODS_GLOBAL["skewness_"],
    StatisticalFeature.kurtosis.value: STAT_METHODS_GLOBAL["kurtosis_"],
    StatisticalFeature.n_peaks.value: STAT_METHODS_GLOBAL["n_peaks_"],
    StatisticalFeature.slope.value: STAT_METHODS_GLOBAL["slope_"],
    StatisticalFeature.ben_corr.value: STAT_METHODS_GLOBAL["ben_corr_"],
    StatisticalFeature.interquartile_range.value: STAT_METHODS_GLOBAL["interquartile_range_"],
    StatisticalFeature.energy.value: STAT_METHODS_GLOBAL["energy_"],
    StatisticalFeature.cross_rate.value: STAT_METHODS_GLOBAL["cross_rate_"],
    StatisticalFeature.autocorrelation.value: STAT_METHODS_GLOBAL["autocorrelation_"],
    StatisticalFeature.ptp_amplitude.value: STAT_METHODS_GLOBAL["ptp_amplitude_"],
    StatisticalFeature.mean_ptp_distance.value: STAT_METHODS_GLOBAL["mean_ptp_distance_"],
    StatisticalFeature.crest_factor.value: STAT_METHODS_GLOBAL["crest_factor_"],
    StatisticalFeature.mean_ema.value: STAT_METHODS_GLOBAL["mean_ema_"],
    StatisticalFeature.mean_moving_median.value: STAT_METHODS_GLOBAL["mean_moving_median_"],
    StatisticalFeature.hjorth_mobility.value: STAT_METHODS_GLOBAL["hjorth_mobility_"],
    StatisticalFeature.hjorth_complexity.value: STAT_METHODS_GLOBAL["hjorth_complexity_"],
    StatisticalFeature.hurst_exponent.value: STAT_METHODS_GLOBAL["hurst_exponent_"],
    StatisticalFeature.petrosian_fractal_dimension.value: STAT_METHODS_GLOBAL[
        "petrosian_fractal_dimension_"
    ],
}

DEFAULT_STAT_FEATURES = tuple(STAT_FEATURE_TO_METHOD.keys())


NORMALIZATION_HANDLERS = {
    NormalizationMethod.imputation: ImputationNormalizer,
    NormalizationMethod.feature_standardization: FeatureStandardizationNormalizer,
    NormalizationMethod.image_standardization: ImageStandardizationNormalizer,
    NormalizationMethod.log1p: Log1pNormalizer,
}


def _coerce_stat_feature(value: StatisticalFeature | str) -> str:
    if isinstance(value, StatisticalFeature):
        return value.value
    return StatisticalFeature(str(value)).value


def transform_stats(series: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    feature_names = tuple(
        _coerce_stat_feature(name)
        for name in params.get("feature_names", DEFAULT_STAT_FEATURES)
    )

    extracted = None
    try:
        from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import (
            TorchQuantileExtractor,
        )

        extractor = TorchQuantileExtractor(
            {
                "window_size": params.get("window_size", 0),
                "stride": params.get("stride", 1),
                "add_global_features": params.get("add_global_features", True),
            }
        )
        extracted = extractor.generate_features_from_ts(series).to(series.device)
    except Exception:
        extracted = None

    if extracted is not None and feature_names == DEFAULT_STAT_FEATURES:
        return torch.nan_to_num(extracted)

    vectors = []
    for feature_name in feature_names:
        method = STAT_FEATURE_TO_METHOD[feature_name]
        feature = torch.as_tensor(method(series, axis=-1), dtype=torch.float32, device=series.device)
        vectors.append(feature.reshape(series.shape[0], -1))
    return torch.nan_to_num(torch.cat(vectors, dim=-1))


def transform_gaf(series: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    return GAF(params).transform(series)


def transform_stft(series: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    return STFTSpectrogram(params).transform(series)


def transform_mtf(series: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    return MTF(params).transform(series)


TRANSFORMATION_HANDLERS = {
    MultimodalModality.stats: transform_stats,
    MultimodalModality.gaf: transform_gaf,
    MultimodalModality.stft: transform_stft,
    MultimodalModality.mtf: transform_mtf,
}
