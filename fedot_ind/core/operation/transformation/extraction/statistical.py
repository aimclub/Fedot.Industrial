from typing import Union

from fedot_ind.core.operation.transformation.extraction.statistical_methods import *

stat_methods = {'mean_': np.mean,
                'median_': np.median,
                'std_': np.std,
                'max_': np.max,
                'min_': np.min,
                'q5_': q5,
                'q25_': q25,
                'q75_': q75,
                'q95_': q95,
                'sum_': np.sum,
                'dif_': diff}

stat_methods_global = {
    'skewness_': skewness,
    'kurtosis_': kurtosis,
    'n_peaks_': n_peaks,
    'slope_': slope,
    'ben_corr_': ben_corr,
    'interquartile_range_': interquartile_range,
    'energy_': energy,
    'cross_rate_': zero_crossing_rate,
    'autocorrelation_': autocorrelation,
    # 'base_entropy_': base_entropy,
    'shannon_entropy_': shannon_entropy,
    'ptp_amplitude_': ptp_amp,
    'crest_factor_': crest_factor,
    'mean_ema_': mean_ema,
    'mean_moving_median_': mean_moving_median,
    'hjorth_mobility_': hjorth_mobility,
    'hjorth_complexity_': hjorth_complexity,
    'hurst_exponent_': hurst_exponent,
    'petrosian_fractal_dimension_': pfd,
}
