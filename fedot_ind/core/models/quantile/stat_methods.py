from fedot_ind.core.models.quantile.stat_features import *

stat_methods = {'mean_': np.mean,
                'median_': np.median,
                'std_': np.std,
                'max_': np.max,
                'min_': np.min,
                'q5_': q5,
                'q25_': q25,
                'q75_': q75,
                'q95_': q95,
                # 'sum_': np.sum,
                # 'dif_': diff
                }

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
    'mean_ptp_distance_': mean_ptp_distance,
    'crest_factor_': crest_factor,
    'mean_ema_': mean_ema,
    'mean_moving_median_': mean_moving_median,
    'hjorth_mobility_': hjorth_mobility,
    'hjorth_complexity_': hjorth_complexity,
    'hurst_exponent_': hurst_exponent,
    'petrosian_fractal_dimension_': pfd,
}

# class StatFeaturesExtractor:
#     """Class for generating quantile features for a given time series.
#
#     """
#
#     @staticmethod
#     def create_baseline_features(time_series: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
#         """
#         Method for creating baseline quantile features for a given time series.
#
#         Args:
#             time_series: time series for which features are generated
#
#         Returns:
#             Row vector of quantile features in the form of a pandas DataFrame
#
#         """
#         names = []
#         vals = []
#         # flatten time series
#         if isinstance(time_series, (pd.DataFrame, pd.Series)):
#             time_series = time_series.values
#         time_series = time_series.flatten()
#
#         for name, method in stat_methods.items():
#             try:
#                 vals.append(method(time_series))
#                 names.append(name)
#             except ValueError:
#                 continue
#         return pd.DataFrame([vals], columns=names)
