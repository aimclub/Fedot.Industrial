from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks


class WindowSizeSelection:
    """Window Size Selection class (WSS).

    This class helps to find appropriate window size in time series to study its features.

    At WSS class we have two group of algorithms. They are Whole-Series-Based (WSB) and subsequence-based (SB).

    Whole-Series-Based (WSB):
        1. 'highest_autocorrelation'
        2. 'dominant_fourier_frequency'

    Subsequence-based (SB):
        1. 'multi_window_finder'
        2. 'summary_statistics_subsequence'

    Note:
        All algorithms has O(n)! Important to set window_max and window_min parameters in case of big time series.

    Args:
        time_series: time series sequences to study.
        window_max: maximum window length to study time series. By default, it is len(time_series)/2.
        window_min: minimum window length to study time series. By default, it is 10.
        wss_algorithm: type of WSS algorithm for your task. By default, it is 'dominant_fourier_frequency'.
            You can choose between:
             'highest_autocorrelation', 'dominant_fourier_frequency',
             'summary_statistics_subsequence' or 'multi_window_finder'.

    Attributes:
        length_ts(int): length of the time_series.

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
            self.window_max = int(len(self.time_series) / 3)

        if self.window_min is None:
            self.window_min = int(len(self.time_series) / 10)

    def autocorrelation(self) -> Tuple[int, list]:
        """Main function for the highest_autocorrelation (AC) to find an appropriate window size.

        Note:
            self.time_series have to be a list!

        Returns:
            window_size_selected: selected window size for the time series
            distance_scores: score list of DFF metric to understand window size selection

        """
        cutted_ts = []
        list_score = [self.high_ac_metric(self.time_series[i:] + self.time_series[:i], i) \
                      for i in range(self.window_min, self.window_max)]
        selected_size_window, list_score_peaks = self.local_max_search(list_score)
        return selected_size_window, list_score

    def high_ac_metric(self, copy_ts: list, i: int) -> float:
        """Find metric score for the AC.

        Args:
            copy_ts: a list of lagged time series according to window size.
            i: selected window size.

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

    def local_max_search(self, score_list: list) -> Tuple[int, list]:
        """Find global max id in time series over score list for AC

        Args:
            score_list: a list of lagged time series according to window size.

        Returns:
            window_size_selected: id for the highest peak in score_list.
            list_score_peaks: list of peaks id in score_list.

        """
        list_probably_peaks = find_peaks(score_list)[0][1:]
        list_score_peaks = [score_list[i] for i in list_probably_peaks]
        max_score = max(list_score_peaks)
        window_size_selected = score_list.index(max_score) + self.window_min
        return window_size_selected, list_score_peaks

    def dominant_fourier_frequency(self) -> Tuple[int, list]:
        """Main function for the dominant_fourier_frequency (DFF) to find an appropriate window size.

        Returns:
            window_size_selected: selected window size for the time series
            distance_scores: score list of DFF metric to understand window size selection

        """
        list_score_k = [self.coeff_metrics(i) for i in range(self.window_min, self.window_max)]
        max_score = max(list_score_k)  # take max value for selected window
        window_size_selected = list_score_k.index(max_score) + self.window_min
        return window_size_selected, list_score_k

    def coeff_metrics(self, temp_size: int) -> float:
        """Find metric score for the DFF.

        Args:
            temp_size: window_size_selected.

        Returns:
            score_coeff: value which is DFF metric distance for selected window size.

        """
        length_n = len(self.time_series)
        temp_list_coeff = [ts * np.exp((-2) * np.pi * (complex(0, -1)) * index * temp_size / length_n) \
                           for index, ts in enumerate(self.time_series)]
        complex_coeff = sum(temp_list_coeff)
        score_coeff = np.sqrt(complex_coeff.real ** 2 + complex_coeff.imag ** 2)
        return score_coeff

    def summary_statistics_subsequence(self, lbound: int = 20, threshold: float = 0.89) -> Tuple[int, list]:
        """Main function for the summary_statistics_subsequence (SuSS) to find an appropriate window size.

        Note:
            This is the fastest implementation - O(log(N))

        Args:
            lbound: first assumption about window_size.
            threshold: normalized anomaly score.

        Returns:
            lbound: selected window size for the time series.
            distance_scores: score list of SuSS score and window_size to understand window size selection.

        """
        ts_fix = np.array(self.time_series)
        ts = (ts_fix - ts_fix.min()) / (ts_fix.max() - ts_fix.min())

        ts_mean = np.mean(ts)
        ts_std = np.std(ts)
        ts_min_max = np.max(ts) - np.min(ts)

        stats = (ts_mean, ts_std, ts_min_max)

        max_score = self.suss_score(ts, 1, stats)
        min_score = self.suss_score(ts, ts.shape[0] - 1, stats)

        exp = 0

        list_score = []
        while True:
            window_size = 2 ** exp
            if window_size < lbound:
                exp += 1
                continue
            score = 1 - (self.suss_score(ts, window_size, stats) - min_score) / (max_score - min_score)
            if score > threshold:
                break
            exp += 1
            list_score.append([score, window_size])
        lbound, ubound = max(lbound, 2 ** (exp - 1)), 2 ** exp + 1
        while lbound <= ubound:
            window_size = int((lbound + ubound) / 2)
            score = 1 - (self.suss_score(ts, window_size, stats) - min_score) / (max_score - min_score)
            list_score.append([score, window_size])
            if score < threshold:
                lbound = window_size + 1
            elif score > threshold:
                ubound = window_size - 1
            else:
                break
        return lbound, list_score

    def suss_score(self, ts: np.array, window_size: int, stats: list) -> float:
        """Find difference between global statistic and statistic of subsequences with different window
           for the SuSS.

        Args:
            ts: normalized numpy time series.
            window_size: temporary selected window size.
            stats: statistic over all normalized time series.

        Returns:
            np.mean(x): not normalized euclidian distance between statistic
                for selected window size and general time series statistic.

        """
        roll = pd.Series(ts).rolling(window_size)
        ts_mean, ts_std, ts_min_max = stats

        roll_mean = roll.mean().to_numpy()[window_size:]
        roll_std = roll.std(ddof=0).to_numpy()[window_size:]
        roll_min = roll.min().to_numpy()[window_size:]
        roll_max = roll.max().to_numpy()[window_size:]

        x = np.array([roll_mean - ts_mean, roll_std - ts_std, (roll_max - roll_min) - ts_min_max])
        x = np.sqrt(np.sum(np.square(x), axis=0)) / np.sqrt(window_size)
        return np.mean(x)

    def multi_window_finder(self) -> Tuple[int, list]:
        """Main multi_window_finder (MWF) function to find an appropriate window size.

        Note:
            In case of 1 global minimum over ts. it is better to try increase max-min boards of ts or change algorithm.

        Returns:
            window_size_selected: selected window size for the time series.
            distance_scores: score list of MWF metric to understand window size selection.

        """
        distance_scores = [self.mwf_metric(i) for i in range(self.window_min, self.window_max)]
        minimum_id_list, id_max = self.top_local_minimum(distance_scores)
        k = 1  # the true first local minimum
        if len(minimum_id_list) < 2:
            k = 0
        window_size_selected = minimum_id_list[k] * 10 + self.window_min + id_max
        return window_size_selected, distance_scores

    def mwf_metric(self, window_selected_temp: int) -> float:
        """Function for MWF method to find metric value for a chosen window size.

        Args:
            window_selected_temp: selected window size for your time series.

        Returns:
            distance_k: value which is the metric distance for selected window size.

        """
        coeff_temp = 1 / window_selected_temp
        m_values = []
        for k in range(self.length_ts - window_selected_temp + 1):
            m_k = [self.time_series[g + k] for g in range(window_selected_temp - 1)]
            m_values.append(coeff_temp * sum(m_k))
        distance_k = sum(np.log10(abs(m_values - np.mean(m_values))))
        return distance_k

    def top_local_minimum(self, distance_scores: list) -> Tuple[int, list]:
        """Find top list of local minimum in scores for MWF method.

        Args:
            distance_scores: list of distance according to MWF metric for selected window sizes

        Returns:
            id_local_minimum_list: list of index where ndarray has local minimum.
            id_max: id where distance_scores has global max value for distance scores.

        """
        id_max = distance_scores.index(max(distance_scores))
        score_temp = distance_scores[id_max:]
        number_windows_temp = len(score_temp) // 10
        scorer_list = [sum(abs(score_temp[i:i + 10] \
                               - np.mean(score_temp[i:i + 10]))) \
                       for i in range(number_windows_temp)]
        id_local_minimum_list = argrelextrema(np.array(scorer_list), np.less)[0]
        return id_local_minimum_list, id_max

    def get_window_size(self) -> Tuple[int, list]:
        """Main function to run WSS class over selected time series

        Note:
            One of the reason of ValueError is that time series size can be equal or smaller than 50.
            In case of it try to initially set window_size min and max.

        Raises:
            ValueError: If `self.window_max` is equal or smaller to `self.window_min`.

        Returns:
            window_size_selected: value which has been chosen as appropriate window size
            list_score: score list for chosen algorithm metric to understand window size selection

        """
        if self.window_max <= self.window_min:
            raise ValueError('Hyperparameters error! self.window_max is equal or smaller to self.window_min')
        else:
            window_size_selected, list_score = self.dict_methods[self.wss_algorithm]()
        return window_size_selected, list_score


class WindowCutter:
    """
    Window cutter.
        input format: dict with "data" and "labels" fields
        output: the same dict but with additional windows_list and labels for it
    """

    def __init__(self, window_len, window_step):
        #super().__init__(name="Window Cutter", operation="window cutting")
        self.window_len = window_len
        self.window_step = window_step
        self.output_window_list = []

    def load_data(self, input_dict: dict) -> None:
        self.input_dict = input_dict

    def get_windows(self) -> dict:
        return self.output_window_list

    def run(self) -> None:
        """
        Cut data to windows
        :return: none
        """
        self.output_window_list = self._cut_ts_to_windows(self.input_dict)

    def _cut_ts_to_windows(self, ts: dict) -> list:
        start_idx = 0
        end_idx = len(ts[list(ts.keys())[0]]) - self.window_len
        temp_windows_list = []
        for i in range(start_idx, end_idx, self.window_step):
            temp_window_dict = {}
            for key in ts.keys():
                temp_window = []
                for j in range(i, i + self.window_len):
                    temp_window.append(ts[key][j])
                temp_window_dict[key] = temp_window
        temp_windows_list.append(temp_window_dict)
        return temp_windows_list
