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
            'multi_window_finder'. По умолчанию всегда будет 'highest_autocorrelation'

        return:
            window_size_selected and list_score
            
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
            self.window_max = int(len(time_series)/2)
        
        if self.window_min == None:
            self.window_min = 0
            
        if self.wss_algorithm == None:
            self.wss_algorithm = 'highest_autocorrelation'
        
    def autocorrelation(self):
        #код
        #Надо итеративно брать копию списка с каким-то лагом (скажем 1) и проверять по какой-то из метрик схожесть
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

    def multi_window_finder(self):
        #код
        pass

    def runner_wss(self):
        if self.wss_algorithm == 'highest_autocorrelation':
            window_size_selected, list_score = self.autocorrelation()
        elif self.wss_algorithm == 'multi_window_finder':
            window_size_selected = self.multi_window_finder()
        return window_size_selected, list_score


# In[3]:


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


# In[4]:


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


# In[5]:


if __name__ == "__main__":
    df_test_well = dataframe_expirement(1)
    df_test_ecg = dataframe_expirement(0)
    time_series = list(df_test_well['uR/h'])
    time_series_2 = list(df_test_ecg['EcgWaveform'][2500:])


# In[6]:


if __name__ == "__main__":
    scorer = WindowSizeSelection(time_series = time_series,
                                window_max = 500)
    window_size_selected, score_list = scorer.runner_wss()
    plot_data_scores_and_selected_window(time_series,
                                        score_list,
                                        window_size_selected)
    print(f'{window_size_selected} - гамма-каротаж')
    
    scorer_2 = WindowSizeSelection(time_series = time_series_2,
                                  window_max = 500)
    window_size_selected_2, score_list_2 = scorer_2.runner_wss()
    plot_data_scores_and_selected_window(time_series_2,
                                        score_list_2,
                                        window_size_selected_2)
    print(f'{window_size_selected_2} - ЭКГ киберспортсмена за игрой в доту')


# In[ ]:




