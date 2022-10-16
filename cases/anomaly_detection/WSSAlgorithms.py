#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd #работа с dataframe

from io import StringIO #для закачки данных с yandex.cloud
import requests #для закачки данных с yandex.cloud

import matplotlib.pyplot as plt #для визуализации

from scipy.signal import find_peaks #для нахождения пиков в данных - метод AC


# In[2]:


# additional modules
import sys
sys.path.append(r'C:\Users\1\Desktop\GitHub\SKAB\utils')
import t2 # https://github.com/waico/SKAB/blob/master/utils/t2.py


# In[19]:


class WindowSizeSelection:
    """Window size selection class in unsupervised Time Series Analytics.
            ----------
       Класс выбора размера окна временного ряда при помощи аналитики без участия учителя.
    """

    def __init__(self,
                 time_series,
                 window_max: int = None,
                 window_min: int = None,
                 wss_algorithm: str = None):
        """Window Size Selection class for Segmentation Task. Whole-Series-Based (WSB) and subsequence-based (SB) algorithms.
                ----------
           Класс выбора размера окна временного ряда для задач сегментации. В классе представленны алгоритмы базируемые на анализе
           всего временного ряда (WSB) и анализе подпоследовательностей временного ряда (SB)
                ----------
        time_series: it can be a list or np.array() / временной ряд может быть представлен как list или np.array()
            time series sequences / временной ряд 
        window_max: int
            maximum window length which an algorithm checks (keep in mind that O(n)!). Default is len(time_series)/2 / максимальный 
            размер окна который будет анализирован алгоритмом (помним, что алгоритмы иммеют сложность O(n)!). По умолчанию len(time_series)/2.
        window_min: int
            minimum window length which an algorithm checks (keep in mind that O(n)!). Default is 1 / минимальный размер окна который будет
            анализирован алгоритмом (помним, что алгоритмы иммеют сложность O(n)!). По умолчанию значение 1.
        wss_algorithm: str
            type of WSS algorithm which you are going to use. It can be 'highest_autocorrelation' or 'multi_window_finder' /
            тип WSS алгоритма, который будет использован для выбора окна. Вы можете выбрать между 'highest_autocorrelation' и
            'multi_window_finder'. По умолчанию всегда будет 'dominant_fourier_frequency'
            
        Reference:
            (c) "Windows Size Selection in Unsupervised Time Series Analytics: A Review and Benchmark. Arik Ermshaus, 
        Patrick Schafer, and Ulf Leser"
        
        last_update: 02.10.2022 @GishB
        """
        self.time_series = time_series
        self.window_max = window_max
        self.window_min = window_min
        self.wss_algorithm = wss_algorithm
        
        if self.window_max == None:
            self.window_max = int(len(time_series)/5)
        
        if self.window_min == None:
            self.window_min = 5
            
        if self.wss_algorithm == None:
            self.wss_algorithm = 'dominant_fourier_frequency'
        
    def autocorrelation(self):
        list_score = []
        for i in range(self.window_min, self.window_max):
            copy_ts = self.time_series[i:] + self.time_series[:i]
            score = self.high_ac_metric(copy_ts, i)
            list_score.append(score)
        #выбор наилучшего результата исходя из list_score
        selected_size_window, list_score_peaks = self.local_max_search(list_score)
        return selected_size_window, list_score
        
    def high_ac_metric(self, copy_ts, i):
        temp_len = len(copy_ts)
        temp_coef = 1/(temp_len-i-1)
        mean_ts = np.mean(copy_ts)
        std_ts = np.std(copy_ts)
        score_list = []
        for g in range(temp_len):
            score_list.append((copy_ts[g] - mean_ts)*(self.time_series[g] - mean_ts)/(std_ts**2))
        a_score = max(score_list)*temp_coef
        return a_score
    
    def local_max_search(self, score_list):
        list_probably_peaks = find_peaks(score_list)[0][1:]
        list_score_peaks = []
        for i in list_probably_peaks:
            list_score_peaks.append(score_list[i])
        max_score = max(list_score_peaks)
        window_size_selected = score_list.index(max_score)
        return window_size_selected, list_score_peaks
    
    def dominant_fourier_frequency(self):
        list_score_k = []
        list_f = []
        for i in range(self.window_min, self.window_max):
            coeff_temp, f = self.coeff_metrics(i)
            list_score_k.append(coeff_temp)
            list_f.append(f)
        max_score = max(list_score_k)
        window_size_selected = list_score_k.index(max_score) + self.window_min
        return window_size_selected, list_score_k
    
    def coeff_metrics(self, temp_size):
        temp_list_coeff = []
        length_n = len(self.time_series)
        for j in range(length_n):
            int_ts_coeff = self.time_series[j]*np.exp((-2)*np.pi*(complex(0, -1))*j*temp_size/length_n)
            temp_list_coeff.append(int_ts_coeff)
        complex_coeff = sum(temp_list_coeff)
        score_coeff = np.sqrt(complex_coeff.real**2 + complex_coeff.imag**2)
        f_iter = temp_size/length_n
        return score_coeff, f_iter
    
    def summary_statistics_subsequence(self):
        ts = self.time_series
        stats_ts = [np.mean(ts), np.std(ts), min(ts), max(ts)] # что такое этот roll_ts (min, max значения?!)
        list_score = []
        for i in range(self.window_min, self.window_max):
            window_size = i
            list_score.append(self.suss_score(ts, window_size, stats_ts))
        window_size_selected = self.calc_suss(ts, t)
        return window_size_selected, list_score
    
    def stats_diff(self, ts, window_size, stats_ts):
        list_stats_diff = []
        for i in range(len(ts)):
            temp_sub = ts[i:window_size+i]
            stats_temp = [np.mean(temp_sub), np.std(temp_sub), min(temp_sub), max(temp_sub)]
            stats_diff = (1/np.sqrt(window_size))*np.sqrt((stats_temp[0]-stats_ts[0])**2 + (stats_temp[1]-stats_ts[1])**2                                                           + (stats_temp[2]-stats_ts[2])**2 + (stats_temp[3]-stats_ts[3])**2)
            list_stats_diff.append(stats_diff)
        return np.mean(list_stats_diff) #что такое этот roll_range?! roll_mean?! , roll_std?!
    
    def suss_score(self, ts, window_size, stats_ts):
        s_min, s_max = self.stats_diff(ts, len(ts), stats_ts), self.stats_diff(ts, 1, stats_ts)
        score = self.stats_diff(ts, window_size, stats_ts)
        score_scaled = (score - s_min)/(s_max - s_min)
        print(score_scaled)
        return 1 - score_scaled
    
    def calc_suss(self, T, t):
        #что такое predifined threshold = t e [0,....,1] !!!!!!!!!!!!!!!!!!!?!
        #min-max scaler T
        #lbound, ubound - expotential search
        window_size = 1 #binary search
        return window_size

    def multi_window_finder(self):
#         distance_score = []
#         for i in range(self.window_min, self.window_max):
#             distance = self.mwf_metric(i)
#             distance_score.append(distance)
# #         window_size_selected = first_local_minimum(distance_score)
#         window_size_selected = 1
#         return window_size_selected, distance_score
        pass
    
    def mwf_metric(self, i):
#         list_scores_temp = []
#         coeff_temp = 1/i
#         for g in range(len(self.time_series)-i):
            
# #         for g in range(len(self.time_series)-i):
#             window_iter_temp = self.time_series[g:i+g]
# #             list_mean_values.append(np.full((len(window_iter_temp)-1, 1), np.mean(window_iter_temp)).reshape(1,-1)[0])
#             mean_iter = np.mean(window_iter_temp)
#             cumsum = np.cumsum(np.insert(window_iter_temp , 0, 0))
#             temp_average_vect = (cumsum[2:] - cumsum[:-2]) / float(2)
#             list_scores_temp.append(sum(abs(temp_average_vect - mean_iter)))
#         result = np.log(coeff_temp*sum(list_scores_temp))
#         return result
        pass
            
    def runner_wss(self):
        if self.wss_algorithm == 'highest_autocorrelation':
            window_size_selected, list_score = self.autocorrelation()
        elif self.wss_algorithm == 'multi_window_finder':
            window_size_selected, list_score = self.multi_window_finder()
        elif self.wss_algorithm == 'dominant_fourier_frequency':
            window_size_selected, list_score = self.dominant_fourier_frequency()
        elif self.wss_algorithm == 'summary_statistics_subsequence':
                window_size_selected, list_score = self.summary_statistics_subsequence()
        return window_size_selected, list_score


# In[20]:


def dataframe_expirement(i):
    if i == 1:
        url = "https://storage.yandexcloud.net/cloud-files-public/dataframe.csv"  # #путь к файлу на yandex.cloud
        dataframe = pd.read_csv(StringIO(requests.get(url).content.decode('utf-8')), sep='|')
        dataframe_columns = dataframe.columns  # #колонки dataframe
        first_label_list = dataframe[dataframe_columns[0]].unique()  # #уникальные наименования скважин
        dataframe_aa564g = dataframe[dataframe[dataframe_columns[0]] == first_label_list[0]]  # #срез df по AA564G
        dataframe_aa564g_first = dataframe_aa564g.drop(axis=1, labels=(dataframe_aa564g.columns[0]))             .drop(axis=1, labels=(dataframe_aa564g.columns[1]))[['m', 'v/v', 'v/v.1',
                                                                 'uR/h', 'ohmm', 'ohmm.1', 'ohmm.2', 'ohmm.3', 'ohmm.5',
                                                                 'ohmm.6',
                                                                 'unitless', 'unitless.1']].reset_index(drop=True)
        dataframe_edited_ = dataframe_aa564g_first.loc[dataframe_aa564g_first['unitless.1'] >= 0]             .loc[dataframe_aa564g_first['unitless'] >= 0].loc[dataframe_aa564g_first['ohmm'] >= 0]             .loc[dataframe_aa564g_first['ohmm.1'] >= 0].loc[dataframe_aa564g_first['ohmm.2'] >= 0]             .loc[dataframe_aa564g_first['ohmm.3'] >= 0].loc[dataframe_aa564g_first['ohmm.5'] >= 0]             .loc[dataframe_aa564g_first['ohmm.6'] >= 0].loc[dataframe_aa564g_first['uR/h'] >= 0]             .loc[dataframe_aa564g_first['v/v.1'] >= 0].loc[dataframe_aa564g_first['v/v'] >= 0]             .reset_index(drop=True)
    else:
        url = "https://storage.yandexcloud.net/cloud-files-public/noname_ECG_2022.csv"  # #путь к файлу на yandex.cloud
        dataframe_edited_ = pd.read_csv(StringIO(requests.get(url).content.decode('utf-8')), sep=',')
    return dataframe_edited_


# In[21]:


def plot_data_scores_and_selected_window(ts, score_list, window_size_selected):
    #тут код для проверки визуальной того что получилось
    f, ax = plt.subplots(2, 1, figsize=(30, 20))
    ax[0].plot(ts)
    ax[0].vlines(x=window_size_selected, ymin=min(ts), ymax=max(ts),
        linestyle = ':',
        linewidth = 6,
        color = 'darkblue')
    ax[0].set_title("ts")
    ax[1].plot(score_list, "r")
    ax[1].set_title("score")
    f.show()


# In[22]:


if __name__ == "__main__": 
#         ts_1 = list(dataframe_expirement(1)['uR/h'])
#         scorer = WindowSizeSelection(time_series = ts_1)
#         window_size_selected, score_list = scorer.runner_wss()
#         plot_data_scores_and_selected_window(ts_1,
#                                             score_list,
#                                             window_size_selected)
#         print(f'{window_size_selected} - гамма-каротаж при помощи DFF')
    
        ts_12 = list(dataframe_expirement(1)['uR/h'])
        scorer_12 = WindowSizeSelection(time_series = ts_12, wss_algorithm = 'summary_statistics_subsequence')
        window_size_selected_12, score_list_12 = scorer_12.runner_wss()
        plot_data_scores_and_selected_window(ts_12,
                                            score_list_12,
                                            window_size_selected_12)
        print(f'{window_size_selected_12} - гамма-каротаж при помощи AC')
        
#     ts_2 = list(dataframe_expirement(0)['EcgWaveform'])
#     scorer_2 = WindowSizeSelection(time_series = ts_2)
#     window_size_selected_2, score_list_2 = scorer_2.runner_wss()
#     plot_data_scores_and_selected_window(ts_2,
#                                         score_list_2,
#                                         window_size_selected_2)
#     print(f'{window_size_selected_2} - ЭКГ киберспортсмена за игрой в доту')
    


# In[ ]:




