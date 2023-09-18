import math
from typing import Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf


class WindowSizeSelector:
    """This class helps to find appropriate window size for time series analysis.

    There are two group of algorithms implemented:

    Whole-Series-Based (WSB):
        1. 'highest_autocorrelation'
        2. 'dominant_fourier_frequency'

    Subsequence-based (SB):
        1. 'multi_window_finder'
        2. 'summary_statistics_subsequence'

    Note:
        All algorithms has O(n)! Important to set window_max and window_min parameters in case of big time series.

    Args:
        method: by ``default``, it is 'dominant_fourier_frequency'.
            You can choose between:
             'highest_autocorrelation', 'dominant_fourier_frequency',
             'summary_statistics_subsequence' or 'multi_window_finder'.

        window_range: % of time series length, by ``default`` it is (5, 50).

    Attributes:
        length_ts(int): length of the time_series.
        window_max(int): maximum window size in real values.
        window_min(int): minimum window size in real values.
        dict_methods(dict): dictionary with all implemented methods.

    Reference:
        (c) "Windows Size Selection in Unsupervised Time Series Analytics: A Review and Benchmark. Arik Ermshaus,
    Patrick Schafer, and Ulf Leser. 2022"

    """

    def __init__(self,
                 method: str = 'dominant_fourier_frequency',
                 window_range: tuple = (5, 50)):

        assert window_range[0] < window_range[1], 'Upper bound of window range should be bigger than lower bound'

        self.dict_methods = {'highest_autocorrelation': self.autocorrelation,
                             'dominant_fourier_frequency': self.dominant_fourier_frequency,
                             # 'multi_window_finder': self.multi_window_finder,
                             # 'summary_statistics_subsequence': self.summary_statistics_subsequence
                             }
        self.wss_algorithm = method
        self.window_range = window_range
        self.window_max = None
        self.window_min = None
        self.length_ts = None

    def apply(self, time_series: Union[pd.DataFrame, np.array], average: str = 'median') -> int:
        """Method to run WSS class over selected time series in parallel mode via joblib

        Args:
            time_series: time series to study
            average: 'mean' or 'median' to average window size over all time series

        Returns:
            window_size_selected: value which has been chosen as appropriate window size
        """
        methods = {'mean': np.mean, 'median': np.median}
        assert average in methods.keys(), 'Hyperparameters error: `average` should be mean or median'

        if isinstance(time_series, pd.DataFrame):
            time_series = time_series.values

        window_list = [self.get_window_size(ts) for ts in time_series]
        return round(methods[average](window_list))

    def get_window_size(self, time_series: np.array) -> int:
        """Main function to run WSS class over selected time series

        Note:
            One of the reason of ValueError is that time series size can be equal or smaller than 50.
            In case of it try to initially set window_size min and max.

        Raises:
            ValueError: If `self.window_max` is equal or smaller to `self.window_min`.

        Returns:
            window_size_selected: value which has been chosen as appropriate window size
        """
        if time_series.shape[0] == 1:  # If time series is a part of multivariate one
            time_series = np.array(time_series[0])
        self.length_ts = len(time_series)

        self.window_max = int(round(self.length_ts * self.window_range[1]/100))  # in real values
        self.window_min = int(round(self.length_ts * self.window_range[0]/100))  # in real values

        window_size_selected = self.dict_methods[self.wss_algorithm](time_series=time_series)
        return round(window_size_selected * 100 / self.length_ts)  # in %

    def dominant_fourier_frequency(self, time_series: np.array) -> int:
        fourier = np.fft.fft(time_series)
        freq = np.fft.fftfreq(time_series.shape[0], 1)

        magnitudes, window_sizes = [], []

        for coef, freq in zip(fourier, freq):
            if coef and freq > 0:
                window_size = int(1 / freq)
                mag = math.sqrt(coef.real * coef.real + coef.imag * coef.imag)

                if self.window_min <= window_size < self.window_max:
                    window_sizes.append(window_size)
                    magnitudes.append(mag)

        return window_sizes[np.argmax(magnitudes)]

    def autocorrelation(self, time_series):
        acf_values = acf(time_series, fft=True, nlags=int(time_series.shape[0] / 2))

        peaks, _ = find_peaks(acf_values)
        peaks = peaks[np.logical_and(peaks >= self.window_min, peaks < self.window_max)]
        corrs = acf_values[peaks]

        if peaks.shape[0] == 0: # if there is no peaks in range (window_min, window_max) return window_min
            return self.window_range[0]
        return peaks[np.argmax(corrs)]

    # def summary_statistics_subsequence(self, lbound: int = 20, threshold: float = 0.89) -> Tuple[int, list]:
    #     """Main function for the summary_statistics_subsequence (SuSS) to find an appropriate window size.
    #
    #     Note:
    #         This is the fastest implementation - O(log(N))
    #
    #     Args:
    #         lbound: first assumption about window_size.
    #         threshold: normalized anomaly score.
    #
    #     Returns:
    #         lbound: selected window size for the time series.
    #         distance_scores: score list of SuSS score and window_size to understand window size selection.
    #
    #     """
    #     ts_fix = np.array(self.time_series)
    #     ts = (ts_fix - ts_fix.min()) / (ts_fix.max() - ts_fix.min())
    #
    #     ts_mean = np.mean(ts)
    #     ts_std = np.std(ts)
    #     ts_min_max = np.max(ts) - np.min(ts)
    #
    #     stats = (ts_mean, ts_std, ts_min_max)
    #
    #     max_score = self.suss_score(ts, 1, stats)
    #     min_score = self.suss_score(ts, ts.shape[0] - 1, stats)
    #
    #     exp = 0
    #
    #     list_score = []
    #     while True:
    #         window_size = 2 ** exp
    #         if window_size < lbound:
    #             exp += 1
    #             continue
    #         score = 1 - (self.suss_score(ts, window_size, stats) - min_score) / (max_score - min_score)
    #         if score > threshold:
    #             break
    #         exp += 1
    #         list_score.append([score, window_size])
    #     lbound, ubound = max(lbound, 2 ** (exp - 1)), 2 ** exp + 1
    #     while lbound <= ubound:
    #         window_size = int((lbound + ubound) / 2)
    #         score = 1 - (self.suss_score(ts, window_size, stats) - min_score) / (max_score - min_score)
    #         list_score.append([score, window_size])
    #         if score < threshold:
    #             lbound = window_size + 1
    #         elif score > threshold:
    #             ubound = window_size - 1
    #         else:
    #             break
    #     return lbound

    # def suss_score(self, ts: np.array, window_size: int, stats: list) -> float:
    #     """Find difference between global statistic and statistic of subsequences with different window
    #        for the SuSS.
    #
    #     Args:
    #         ts: normalized numpy time series.
    #         window_size: temporary selected window size.
    #         stats: statistic over all normalized time series.
    #
    #     Returns:
    #         np.mean(x): not normalized euclidian distance between statistic
    #             for selected window size and general time series statistic.
    #
    #     """
    #     roll = pd.Series(ts).rolling(window_size)
    #     ts_mean, ts_std, ts_min_max = stats
    #
    #     roll_mean = roll.mean().to_numpy()[window_size:]
    #     roll_std = roll.std(ddof=0).to_numpy()[window_size:]
    #     roll_min = roll.min().to_numpy()[window_size:]
    #     roll_max = roll.max().to_numpy()[window_size:]
    #
    #     x = np.array([roll_mean - ts_mean, roll_std - ts_std, (roll_max - roll_min) - ts_min_max])
    #     x = np.sqrt(np.sum(np.square(x), axis=0)) / np.sqrt(window_size)
    #     return np.mean(x)

    # def multi_window_finder(self) -> Tuple[int, list]:
    #     """Main multi_window_finder (MWF) function to find an appropriate window size.
    #
    #     Note:
    #         In case of 1 global minimum over ts. it is better to try increase max-min boards of ts or change algorithm.
    #
    #     Returns:
    #         window_size_selected: selected window size for the time series.
    #         distance_scores: score list of MWF metric to understand window size selection.
    #
    #     """
    #     distance_scores = [self.mwf_metric(i) for i in range(self.window_min, self.window_max)]
    #     minimum_id_list, id_max = self.top_local_minimum(distance_scores)
    #     k = 1  # the true first local minimum
    #     if len(minimum_id_list) < 2:
    #         k = 0
    #     window_size_selected = minimum_id_list[k] * 10 + self.window_min + id_max
    #     return window_size_selected

    # def mwf_metric(self, window_selected_temp: int) -> float:
    #     """Function for MWF method to find metric value for a chosen window size.
    #
    #     Args:
    #         window_selected_temp: selected window size for your time series.
    #
    #     Returns:
    #         distance_k: value which is the metric distance for selected window size.
    #
    #     """
    #     coeff_temp = 1 / window_selected_temp
    #     m_values = []
    #     for k in range(self.length_ts - window_selected_temp + 1):
    #         m_k = [self.time_series[g + k] for g in range(window_selected_temp - 1)]
    #         m_values.append(coeff_temp * sum(m_k))
    #     distance_k = sum(np.log10(abs(m_values - np.mean(m_values))))
    #     return distance_k
    #
    # def top_local_minimum(self, distance_scores: list) -> Tuple[int, list]:
    #     """Find top list of local minimum in scores for MWF method.
    #
    #     Args:
    #         distance_scores: list of distance according to MWF metric for selected window sizes
    #
    #     Returns:
    #         id_local_minimum_list: list of index where ndarray has local minimum.
    #         id_max: id where distance_scores has global max value for distance scores.
    #
    #     """
    #     id_max = distance_scores.index(max(distance_scores))
    #     score_temp = distance_scores[id_max:]
    #     number_windows_temp = len(score_temp) // 10
    #     scorer_list = [sum(abs(score_temp[i:i + 10] \
    #                            - np.mean(score_temp[i:i + 10]))) \
    #                    for i in range(number_windows_temp)]
    #     id_local_minimum_list = argrelextrema(np.array(scorer_list), np.less)[0]
    #     return id_local_minimum_list, id_max
