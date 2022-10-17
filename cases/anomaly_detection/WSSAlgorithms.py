from typing import Union
import numpy as np
from scipy.signal import find_peaks, argrelextrema


class WindowSizeSelection:
    """Window size selection class in unsupervised Time Series Analytics.
            ----------
            
       Класс выбора размера окна временного ряда при помощи аналитики без участия человека.
    """

    def __init__(self,
                 wss_algorithm: str = 'dominant_fourier_frequency',
                 time_series: Union[list, np.array] = None,
                 window_max: int = None,
                 window_min: int = None):
        """Window Size Selection class.

           Whole-Series-Based (WSB) and subsequence-based (SB) algorithms.
                ----------
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
        
        last_update_#2: 21.10.2022 @GishB
        """
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
            self.window_max = int(len(time_series)/5)

        if self.window_min is None:
            self.window_min = 10

    def autocorrelation(self):
        """ Main function for the highest_autocorrelation method
        --------------------------------------------------------
            output = selected_size_window, list_score
        """
        list_score = [self.high_ac_metric(self.time_series[i:] + self.time_series[:i], i)\
                      for i in range(self.window_min, self.window_max)]
        selected_size_window, list_score_peaks = self.local_max_search(list_score)
        return selected_size_window, list_score

    def high_ac_metric(self, copy_ts, i):
        """ Calculate metric value based on chosen chosen window size for the highest_autocorrelation method
        ----------------------------------------------------------------------------------------------------
            input = chosen window size, copy of time series changed via the window size
            output = value which is the highest_autocorrelation distance metric
        """
        temp_len = len(copy_ts)
        temp_coef = 1/(temp_len-i-1)
        mean_ts = np.mean(copy_ts)
        std_ts = np.std(copy_ts)
        score_list = [(copy_ts[g] - mean_ts)*(self.time_series[g] - mean_ts)/(std_ts**2) for g in range(temp_len)]
        a_score = max(score_list)*temp_coef
        return a_score

    def local_max_search(self, score_list):
        """ Find global max value id for the highest_autocorrelation method 
        -------------------------------------------------------------------
            input = score_list
            output = window_size_selected, list_score_peaks
        """
        list_probably_peaks = find_peaks(score_list)[0][1:]
        list_score_peaks = [score_list[i] for i in list_probably_peaks]
        max_score = max(list_score_peaks)
        window_size_selected = score_list.index(max_score) + self.window_min
        return window_size_selected, list_score_peaks

    def dominant_fourier_frequency(self):
        """ Main function for the dominant_fourier_frequency
        ----------------------------------------------------
            output = selected_size_window, list_score
        """
        list_score_k = []
        list_f = []  # Зачем сохранять этот список? В статье это важно было, но не указано где использовать
        for i in range(self.window_min, self.window_max):
            coeff_temp, f = self.coeff_metrics(i)
            list_score_k.append(coeff_temp)
            list_f.append(f)
        max_score = max(list_score_k)  # take max value for selected window
        window_size_selected = list_score_k.index(max_score) + self.window_min
        return window_size_selected, list_score_k

    def coeff_metrics(self, temp_size):
        """ Find score coefficient for the dominant_fourier_frequency
        ------------------------------------------------
            input = temp_size ---> (selected window size)
            output = score_coeff, f_iter
        """
        length_n = len(self.time_series)
        temp_list_coeff = [ts*np.exp((-2)*np.pi*(complex(0, -1))*index*temp_size/length_n)\
                           for index, ts in enumerate(self.time_series)]
        complex_coeff = sum(temp_list_coeff)
        score_coeff = np.sqrt(complex_coeff.real**2 + complex_coeff.imag**2)
        f_iter = temp_size/length_n
        return score_coeff, f_iter

    def summary_statistics_subsequence(self):
        """ Main function for the summary_statistics_subsequence
        --------------------------------------------------------
        output = selected_size_window, list_score
        """
        stats_ts = [np.mean(self.time_series), np.std(self.time_series), np.var(self.time_series)]
        list_score = [self.suss_score(self.time_series, window_size, stats_ts)\
                      for window_size in range(self.window_min, self.window_max)]
        window_size_selected = next(x[0] for x in enumerate(score_list) if x[1] > 0.89) + self.window_min
        return window_size_selected, list_score

    def stats_diff(self, ts, window_size, stats_ts):
        """Find difference between global statistic and statistic of subsequnces with different window
           for the the summary_statistics_subsequence
        ----------------------------------------------------------------------------------------------
            input = time_series, selected window, stats_ts(np.mean,np.std,np.var)
            output = np.mean(stat_diff)
        """
        # Внимание np.var - это то что я понимаю под "roll range(T,w)" из статьи в Reference!!!!!!!!!!!!!!!
        stat_w = [[np.mean(ts[i:i+window_size]), np.std(ts[i:i+window_size]), np.var(ts[i:i+window_size])]\
                  for i in range(self.length_ts)]
        stat_diff = [[(1/window_size)*np.sqrt((stats_ts[0]-stat_w[i][0])**2\
                                              + (stats_ts[1]-stat_w[i][1])**2\
                                              + (stats_ts[2]-stat_w[i][2])**2)] for i in range(len(stats_w))]
        return np.mean(stat_diff)

    def suss_score(self, ts, window_size, stats_ts):
        """Find score coefficient for the the summary_statistics_subsequence
        -------------------------------------------------------------------
            input = time_series, selected window, stats_ts(np.mean, np.std, np.var)
            output = 1 - score_normalize
        """
        s_min, s_max = self.stats_diff(ts, len(ts), stats_ts), self.stats_diff(ts, 1, stats_ts)
        score = self.stats_diff(ts, window_size, stats_ts)
        score_normalize = (score - s_min) / (s_max - s_min)
        print(1 - score_normalize)
        return 1 - score_normalize

    def multi_window_finder(self):
        """ Main function for multi_window_finder method
        ------------------------------------------------
            output = selected_size_window, list_score
        """
        distance_scores = [self.mwf_metric(i) for i in range(self.window_min, self.window_max)]
        top_minimum_list_window = self.top_local_minimum(distance_scores)
        window_size_selected = top_minimum_list_window[1] + self.window_min  # take second value as the result -> [1]
        return window_size_selected, distance_scores

    def mwf_metric(self, window_selected_temp):
        """ Find multi_window_finder method metric value for a chosen window size
        --------------------------------------------------
            input = chosen window size
            output = value which is the MWF distance metric
        """
        coeff_temp = 1/window_selected_temp
        m_values = []
        for k in range(self.length_ts-window_selected_temp+1):
            m_k = [self.time_series[g+k] for g in range(window_selected_temp-1)]
            m_values.append(coeff_temp*sum(m_k))
        distance_k = sum(np.log10(abs(m_values - np.mean(m_values))))
        return distance_k

    def top_local_minimum(self, distance_scores):
        """ Find a list of local minimum for multi_window_finder method
        ---------------------------------------------------------------
            input = list of distance scores from mwf_metric
            output = list of index where narray has minimum
        """
        top_list_minimum = argrelextrema(np.array(distance_scores), np.less)[0]
        return top_list_minimum

    def runner_wss(self):
        window_size_selected, list_score = self.dict_methods[self.wss_algorithm]()
        return window_size_selected, list_score
