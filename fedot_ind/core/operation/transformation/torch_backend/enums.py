from enum import Enum
from typing import Dict, Any


class StatisticalFeature(Enum):
    mean = 'mean'
    median = 'median'
    std = 'std'
    max = 'max'
    min = 'min'
    q5 = 'q5'
    q25 = 'q25'
    q75 = 'q75'
    q95 = 'q95'
    skewness = 'skewness'
    kurtosis = 'kurtosis'
    n_peaks = 'n_peaks'
    slope = 'slope'
    ben_corr = 'ben_corr'
    interquartile_range = 'interquartile_range'
    energy = 'energy'
    cross_rate = 'cross_rate'
    autocorrelation = 'autocorrelation'
    ptp_amplitude = 'ptp_amplitude'
    mean_ptp_distance = 'mean_ptp_distance'
    crest_factor = 'crest_factor'
    mean_ema = 'mean_ema'
    mean_moving_median = 'mean_moving_median'
    hjorth_mobility = 'hjorth_mobility'
    hjorth_complexity = 'hjorth_complexity'
    hurst_exponent = 'hurst_exponent'
    petrosian_fractal_dimension = 'petrosian_fractal_dimension'


STAT_FEATURE_CONFIG = Dict[StatisticalFeature, Dict[str, Any]]
