from typing import Union

import numpy as np
from scipy.signal import argrelextrema, find_peaks


class WindowSizeSelection:
    """Window Size Selection class (WSS).

    This class helps to find appropriate window size in time series to study its features.

    At WSS class we have two group of algorithms. They are Whole-Series-Based (WSB) and subsequence-based (SB).

    Whole-Series-Based (WSB):
        1. 'highest autocorrelation'
        2. 'dominant fourier_frequency'

    Subsequence-based (SB):
        1. 'multi window finder'
        2. 'summary statistics subsequence'

    Note:
        All algorithms has O(n)! Important to set window_max and window_min parameters in case of big time series.

    Args:
        time_series (list or np.array()): time series sequences to study.
        window_max (int): maximum window length to study time series. By default, it is len(time_series)/2.
        window_min (int): minimum window length to study time series. By default, it is 10.
        wss_algorithm (str): type of WSS algorithm for your task. By default, it is 'dominant_fourier_frequency'.
            You can chose between:
             'highest_autocorrelation', 'dominant_fourier_frequency',
             'summary_statistics_subsequence' or 'multi_window_finder'.

    Reference:
        (c) "Windows Size Selection in Unsupervised Time Series Analytics: A Review and Benchmark. Arik Ermshaus,
    Patrick Schafer, and Ulf Leser. 2022"
    """

    def __init__(self,
                 wss_algorithm: str = 'dominant_fourier_frequency',
                 time_series: Union[list, np.array] = None,
                 window_max: int = None,
                 window_min: int = None):

        self.dict_methods = {'highest_autocorrelation': self.autocorrelation,
                             'multi_window_finder': self.multi_window_finder,
                             'dominant_fourier_frequency': self.dominant_fourier_frequency,
                             'summary_statistics_subsequence': self.summary_statistics_subsequence}

        self.time_series = time_series
        self.window_max = window_max
        self.window_min = window_min
        self.wss_algorithm = wss_algorithm
        self.length_ts = len(time_series)

        if self.window_max is None:
            self.window_max = int(len(time_series) / 5)

        if self.window_min is None:
            self.window_min = 10

    def autocorrelation(self):
        """Main function for the highest_autocorrelation (AC) to find an appropriate window size.

        Returns:
            window_size_selected (int): selected window size for the time series
            distance_scores (list): score list of DFF metric to understand window size selection
        """
        list_score = [self.high_ac_metric(self.time_series[i:] + self.time_series[:i], i) \
                      for i in range(self.window_min, self.window_max)]
        selected_size_window, list_score_peaks = self.local_max_search(list_score)
        return selected_size_window, list_score

    def high_ac_metric(self, copy_ts, i):
        """Find metric score for the AC.

        Args:
            copy_ts (list): a list of lagged time series according to window size.
            i (int): selected window size.

        Returns:
            a_score (float): value which is AC metric for selected window size.
        """
        temp_len = len(copy_ts)
        temp_coef = 1 / (temp_len - i)
        mean_ts = np.mean(copy_ts)
        std_ts = np.std(copy_ts)
        score_list = [(copy_ts[g] - mean_ts) * (self.time_series[g] - mean_ts) / (std_ts ** 2) for g in range(temp_len)]
        a_score = max(score_list) * temp_coef
        return a_score

    def local_max_search(self, score_list):
        """Find global max id in time series over score list for AC

        Args:
            score_list (list): a list of lagged time series according to window size.

        Returns:
            window_size_selected (int): id for the highest peak in score_list.
            list_score_peaks (list): list of peaks id in score_list.
        """
        list_probably_peaks = find_peaks(score_list)[0][1:]
        list_score_peaks = [score_list[i] for i in list_probably_peaks]
        max_score = max(list_score_peaks)
        window_size_selected = score_list.index(max_score) + self.window_min
        return window_size_selected, list_score_peaks

    def dominant_fourier_frequency(self):
        """Main function for the dominant_fourier_frequency (DFF) to find an appropriate window size.

        Returns:
            window_size_selected (int): selected window size for the time series
            distance_scores (list): score list of DFF metric to understand window size selection
        """
        list_score_k = [self.coeff_metrics(i) for i in range(self.window_min, self.window_max)]
        max_score = max(list_score_k)  # take max value for selected window
        window_size_selected = list_score_k.index(max_score) + self.window_min
        return window_size_selected, list_score_k

    def coeff_metrics(self, temp_size):
        """Find metric score for the DFF.

        Args:
            temp_size (int): window_size_selected.

        Returns:
            score_coeff (float): value which is DFF metric distance for selected window size.
        """
        length_n = len(self.time_series)
        temp_list_coeff = [ts * np.exp((-2) * np.pi * (complex(0, -1)) * index * temp_size / length_n) \
                           for index, ts in enumerate(self.time_series)]
        complex_coeff = sum(temp_list_coeff)
        score_coeff = np.sqrt(complex_coeff.real ** 2 + complex_coeff.imag ** 2)
        return score_coeff

    def summary_statistics_subsequence(self):
        """Main function for the summary_statistics_subsequence (SuSS) to find an appropriate window size.

        Note:
            This implementation is time-consuming.

        Returns:
            window_size_selected (int): selected window size for the time series.
            distance_scores (list): score list of SuSS metric to understand window size selection.
        """
        ts = (self.time_series - np.min(self.time_series)) / (np.max(self.time_series) - np.min(self.time_series))

        stats_ts = [np.mean(ts), np.std(ts), np.max(ts) - np.min(ts)]
        list_score = [self.suss_score(ts, window_size, stats_ts) for window_size
                      in range(self.window_min, self.window_max)]
        window_size_selected = next(x[0] for x in enumerate(list_score) if x[1] > 0.89) + self.window_min
        return window_size_selected, list_score

    def stats_diff(self, ts, window_size, stats_ts):
        """Find difference between global statistic and statistic of subsequences with different window
           for the SuSS.

        Args:
            ts (list): normalized time series.
            window_size (int): temporary selected window size.
            stats_ts (list): statistic over all normalized time series.

        Returns:
            np.mean(stat_diff) (float): not normalized euclidian distance between statistic
                for selected window size and general time series statistic.
        """
        stat_w = [[np.mean(ts[i:i + window_size]), np.std(ts[i:i + window_size]),
                   np.max(ts[i:i + window_size]) - np.min(ts[i:i + window_size])] for i in range(self.length_ts)]
        stat_diff = [[(1 / window_size) * np.sqrt((stats_ts[0] - stat_w[i][0]) ** 2 \
                                                  + (stats_ts[1] - stat_w[i][1]) ** 2 \
                                                  + (stats_ts[2] - stat_w[i][2]) ** 2)] for i in range(len(stat_w))]
        return np.mean(stat_diff)

    def suss_score(self, ts, window_size, stats_ts):
        """Normalized and calculate euclidian distance of SuSS algorithm for selected window size.

        Args:
            ts (list): normalized time series.
            window_size (int): temporary selected window size.
            stats_ts (list): statistic over all normalized time series .

        Returns:
            1 - score_normalize (float): normalized euclidian distance between statistic
                for selected window size and general time series statistic.
        """
        s_min, s_max = self.stats_diff(ts, len(ts), stats_ts), self.stats_diff(ts, 1, stats_ts)
        score = self.stats_diff(ts, window_size, stats_ts)
        score_normalize = (score - s_min) / (s_max - s_min)
        return 1 - score_normalize

    def multi_window_finder(self):
        """Main multi_window_finder (MWF) function to find an appropriate window size.

        Note:
            In case of 1 global minimum over ts. it is better to try increase max-min boards of ts or change algorithm.

        Returns:
            window_size_selected (int): selected window size for the time series.
            distance_scores (list): score list of MWF metric to understand window size selection.
        """
        distance_scores = [self.mwf_metric(i) for i in range(self.window_min, self.window_max)]
        minimum_id_list, id_max = self.top_local_minimum(distance_scores)
        print(minimum_id_list)
        k = 1  # the true first local minimum
        if len(minimum_id_list) < 2:
            k = 0
        window_size_selected = minimum_id_list[k] * 10 + self.window_min + id_max
        return window_size_selected, distance_scores

    def mwf_metric(self, window_selected_temp):
        """Function for MWF method to find metric value for a chosen window size.

        Args:
            window_selected_temp (list): selected window size for your time series.

        Returns:
            distance_k (int): value which is the metric distance for selected window size.
        """
        coeff_temp = 1 / window_selected_temp
        m_values = []
        for k in range(self.length_ts - window_selected_temp + 1):
            m_k = [self.time_series[g + k] for g in range(window_selected_temp - 1)]
            m_values.append(coeff_temp * sum(m_k))
        distance_k = sum(np.log10(abs(m_values - np.mean(m_values))))
        return distance_k

    def top_local_minimum(self, distance_scores):
        """Find top list of local minimum in scores for MWF method.

        Args:
            distance_scores (list): list of distance according to MWF metric for selected window sizes

        Returns:
            id_local_minimum_list (int): list of index where ndarray has local minimum.
            id_max (int): id where distance_scores has global max value for distance scores.
        """
        id_max = distance_scores.index(max(distance_scores))
        score_temp = distance_scores[id_max:]
        number_windows_temp = len(score_temp) // 10
        scorer_list = [sum(abs(score_temp[i:i + 10] \
                               - np.mean(score_temp[i:i + 10]))) \
                       for i in range(number_windows_temp)]
        id_local_minimum_list = argrelextrema(np.array(scorer_list), np.less)[0]
        return id_local_minimum_list, id_max

    def runner_wss(self):
        """Main function to run WSS class over selected time series

        Note:
            One of the reason of ValueError is that time series size can be equal or smaller than 50.
            In case of it try to initially set window_size min and max.

        Raises:
            ValueError: If `self.window_max` is equal or smaller to `self.window_min`.

        Returns:
            window_size_selected (int): value which has been chosen as appropriate window size
            list_score (list): score list for chosen algorithm metric to understand window size selection
        """
        if self.window_max <= self.window_min:
            raise ValueError('Hyperparameters error! self.window_max is equal or smaller to self.window_min')
        else:
            window_size_selected, list_score = self.dict_methods[self.wss_algorithm]()
        return window_size_selected, list_score
