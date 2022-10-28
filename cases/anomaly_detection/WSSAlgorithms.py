from typing import Union

import numpy as np
from scipy.signal import argrelextrema, find_peaks


class WindowSizeSelection:
    """Window Size Selection class. Whole-Series-Based (WSB) and subsequence-based (SB) algorithms.

    time_series: it can be a list or np.array()
        time series sequences
    window_max: int
        maximum window length to check (O(n)!). Default is len(time_series)/2
    window_min: int
        minimum window length to check (O(n)!). Default is 1
    wss_algorithm: str
        type of WSS algorithm. It can be 'highest_autocorrelation', 'dominant_fourier_frequency',
        'summary_statistics_subsequence' or 'multi_window_finder'.
        By default it is 'dominant_fourier_frequency'
    Reference:
        (c) "Windows Size Selection in Unsupervised Time Series Analytics: A Review and Benchmark. Arik Ermshaus,
    Patrick Schafer, and Ulf Leser"
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

        if int(len(time_series) / 5) <= 200:
            self.window_max = len(time_series)

    def autocorrelation(self):
        """
        Main function for the highest_autocorrelation method

        :return: a tuple of selected window size and list of scores for this method
        """
        list_score = [self.high_ac_metric(self.time_series[i:] + self.time_series[:i], i) \
                      for i in range(self.window_min, self.window_max)]
        selected_size_window, list_score_peaks = self.local_max_search(list_score)
        return selected_size_window, list_score

    def high_ac_metric(self, copy_ts, i):
        """
        Calculate metric value based on chosen chosen window size for the highest_autocorrelation method

        :param copy_ts: a list of lagged time series
        :param i: temporary window size (or the lagged value)
        :return: score for selected window size
        """
        temp_len = len(copy_ts)
        temp_coef = 1 / (temp_len - i)
        mean_ts = np.mean(copy_ts)
        std_ts = np.std(copy_ts)
        score_list = [(copy_ts[g] - mean_ts) * (self.time_series[g] - mean_ts) / (std_ts ** 2) for g in range(temp_len)]
        a_score = max(score_list) * temp_coef
        return a_score

    def local_max_search(self, score_list):
        """
        Find global max value id for the highest_autocorrelation method.

        :param score_list: a list of scores obtained
        :return: a tuple of window_size_selected and list_score
        """
        list_probably_peaks = find_peaks(score_list)[0][1:]
        list_score_peaks = [score_list[i] for i in list_probably_peaks]
        max_score = max(list_score_peaks)
        window_size_selected = score_list.index(max_score) + self.window_min
        return window_size_selected, list_score_peaks

    def dominant_fourier_frequency(self):
        """
        Main function for the dominant_fourier_frequency

        :return: a tuple of window_size_selected and list_score
        """
        list_score_k = []
        for i in range(self.window_min, self.window_max):
            coeff_temp = self.coeff_metrics(i)
            list_score_k.append(coeff_temp)
        max_score = max(list_score_k)  # take max value for selected window
        window_size_selected = list_score_k.index(max_score) + self.window_min
        return window_size_selected, list_score_k

    def coeff_metrics(self, temp_size):
        """
        Find score coefficient for the dominant_fourier_frequency

        :param temp_size: temporary selected window size
        :return: a score metric distance
        """

        length_n = len(self.time_series)
        temp_list_coeff = [ts * np.exp((-2) * np.pi * (complex(0, -1)) * index * temp_size / length_n) \
                           for index, ts in enumerate(self.time_series)]
        complex_coeff = sum(temp_list_coeff)
        score_coeff = np.sqrt(complex_coeff.real ** 2 + complex_coeff.imag ** 2)
        return score_coeff

    def summary_statistics_subsequence(self):
        """
        Main function for the summary_statistics_subsequence

        :return: selected window size and a list of score
        """
        ts = (self.time_series - np.min(self.time_series)) / (np.max(self.time_series) - np.min(self.time_series))

        stats_ts = [np.mean(ts), np.std(ts), np.max(ts) - np.min(ts)]
        list_score = [self.suss_score(ts, window_size, stats_ts) for window_size
                      in range(self.window_min, self.window_max)]
        window_size_selected = next(x[0] for x in enumerate(list_score) if x[1] > 0.89) + self.window_min
        return window_size_selected, list_score

    def stats_diff(self, ts, window_size, stats_ts):
        """
        Find difference between global statistic and statistic of subsequnces with different window
           for the summary_statistics_subsequence

        :param ts: time series data
        :param window_size: temporary selected window size
        :param stats_ts: statistic over all ts for calculations
        :return: not normalized euclidian distance between selected window size and general statistic for ts
        """
        stat_w = [[np.mean(ts[i:i + window_size]), np.std(ts[i:i + window_size]),
                   np.max(ts[i:i + window_size]) - np.min(ts[i:i + window_size])] for i in range(self.length_ts)]
        stat_diff = [[(1 / window_size) * np.sqrt((stats_ts[0] - stat_w[i][0]) ** 2 \
                                                  + (stats_ts[1] - stat_w[i][1]) ** 2 \
                                                  + (stats_ts[2] - stat_w[i][2]) ** 2)] for i in range(len(stat_w))]
        return np.mean(stat_diff)

    def suss_score(self, ts, window_size, stats_ts):
        """
        Find score coefficient for the the summary_statistics_subsequence

        :param ts: time series data
        :param window_size: temporary selected window size
        :param stats_ts: statistic over all ts for calculations
        :return: normalized euclidian distance between selected window size and general statistic for ts
        """
        s_min, s_max = self.stats_diff(ts, len(ts), stats_ts), self.stats_diff(ts, 1, stats_ts)
        score = self.stats_diff(ts, window_size, stats_ts)
        score_normalize = (score - s_min) / (s_max - s_min)
        return 1 - score_normalize

    def multi_window_finder(self):
        """
        Main function for multi_window_finder method

        :return: selected window size and a list of scores for this method
        """
        distance_scores = [self.mwf_metric(i) for i in range(self.window_min, self.window_max)]
        minimum_id_list, id_max = self.top_local_minimum(distance_scores)
        print(minimum_id_list)
        window_size_selected = minimum_id_list[1] * 10 + self.window_min + id_max
        return window_size_selected, distance_scores

    def mwf_metric(self, window_selected_temp):
        """
        Find multi_window_finder method metric value for a chosen window size

        :param window_selected_temp: temporary window selected
        :return: value which is the MWF distance metric
        """
        coeff_temp = 1 / window_selected_temp
        m_values = []
        for k in range(self.length_ts - window_selected_temp + 1):
            m_k = [self.time_series[g + k] for g in range(window_selected_temp - 1)]
            m_values.append(coeff_temp * sum(m_k))
        distance_k = sum(np.log10(abs(m_values - np.mean(m_values))))
        return distance_k

    def top_local_minimum(self, distance_scores):
        """
        Find a list of local minimum for multi_window_finder method

        :param distance_scores: list of distance scores from mwf_metric
        :return: list of index where narray has minimum, max id for distance_scores list
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
        if int(len(self.time_series)) <= self.window_min:
            window_size_selected, list_score = int(len(self.time_series)), []
        else:
            window_size_selected, list_score = self.dict_methods[self.wss_algorithm]()
        return window_size_selected, list_score
