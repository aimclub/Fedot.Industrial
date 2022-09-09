#!/usr/bin/env python
# coding: utf-8

# In[1]:


if __name__ == "__main__":
    import sys
    sys.path.append(r"C:\Users\1\Desktop\GitHub\Fedot.Industrial")


# # Подгружаем библиотеки

# In[2]:


import pandas as pd
import numpy as np
from numba import jit
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
from core.models.spectral.SSA import Spectrum
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns


# # Часто используемые функции

# In[3]:


def plot_data_and_score(raw_data, score,
                           oil_x, rock_x):

    f, ax = plt.subplots(4, 1, figsize=(30, 20))
    ax[0].plot(raw_data)
    ax[0].set_title("raw data")
    ax[1].plot(score, "r")
    ax[1].set_title("score")
    ax[2].plot(oil_x)
    ax[2].set_title("oil_x")
    ax[3].plot(rock_x)
    ax[3].set_title("rock_x")
    f.show()
    
def runner(ts_window_length, lag, trajectory_window_length,
        quantile_rate, n_components, view, dynamic_mode):

    scorer = SingularSpectrumTransformation(time_series=np.array(dataframe_edited_X['uR/h'][500:2500]),
                                            ts_window_length=ts_window_length,
                                            lag=lag,
                                            trajectory_window_length=trajectory_window_length,
                                            quantile_rate=quantile_rate,
                                            n_components=n_components,
                                            view=True)
    score = scorer.score_offline(dynamic_mode=dynamic_mode)
    plot_data_and_score(np.array(dataframe_edited_X['uR/h'][500:2500]), score,
                        np.array(dataframe_edited_X['unitless.1'][500:2500]),
                        np.array(dataframe_edited_X['unitless'][500:2500]))


# # Загружаем\обрабатываем csv файл - dataframe_edited_X (по скважине AA564G)

# In[4]:

if __name__ == "__main__":
    path = r'C:\Users\1\Desktop\Магистратура в ИТМО\Диссертация магистра\Работа 2 семестр\Compo_PTRNZHG_A_PTRNZHG_A_test.xlsx'

    dataframe = pd.read_excel(path, skiprows=3)
    dataframe_columns = dataframe.columns
    first_label_list = dataframe[dataframe_columns[0]].unique()
    dataframe_AA564G = dataframe[dataframe[dataframe_columns[0]] == first_label_list[0]]
    dataframe_AA564G_first = dataframe_AA564G.drop(axis=1, labels=(dataframe_AA564G.columns[0]))\
                                            .drop(axis=1, labels=(dataframe_AA564G.columns[1]))\
                                            [['m','v/v','v/v.1','uR/h','ohmm','ohmm.1','ohmm.2',
                                              'ohmm.3','ohmm.5','ohmm.6','unitless','unitless.1']]\
                                            .reset_index(drop=True)
    dataframe_edited_X = dataframe_AA564G_first.loc[dataframe_AA564G_first['unitless.1'] >= 0]\
        .loc[dataframe_AA564G_first['unitless'] >= 0].loc[dataframe_AA564G_first['ohmm'] >= 0]\
        .loc[dataframe_AA564G_first['ohmm.1'] >= 0].loc[dataframe_AA564G_first['ohmm.2'] >= 0]\
        .loc[dataframe_AA564G_first['ohmm.3'] >= 0].loc[dataframe_AA564G_first['ohmm.5'] >= 0]\
        .loc[dataframe_AA564G_first['ohmm.6'] >= 0].loc[dataframe_AA564G_first['uR/h'] >= 0]\
        .loc[dataframe_AA564G_first['v/v.1'] >= 0].loc[dataframe_AA564G_first['v/v'] >= 0]\
        .reset_index(drop=True)


# # Основная логика класса SST с SVD


# # Работа с данными
# #### ts_window_length - длина рассматриваемого окна для "гусениц" матриц H 
# #### lag - длина лага между двумя матрицами 
# #### trajectory_window_length - длина векторов в матрице H
# #### quantile_rate - коеффициент аномалии

# In[6]:


vect_1 = np.array(dataframe_edited_X['uR/h'][500:2500])
vect_2 = vect_1.reshape((len(vect_1), 1))
sns.heatmap(vect_1 + vect_2, norm=LogNorm())


# ### Исходя из экспресс анализа по heatmap можно выделить несколько участков с возможными паттернами. При анализе следует обратить внимание на ярко-светлые и темные повторяющиеся участки.
# 
# ### Рассмотрим работу алгоритма над временным рядом гамма каротажа в динамическом режиме:
# dynamic_mode= True
# #### *Сравним результат визуализации с тепловой картой

# In[7]:


if __name__ == "__main__":
    runner(ts_window_length=100,lag=20,trajectory_window_length=20,
        quantile_rate=0.95,n_components=2,view=True, dynamic_mode=True)


# #### oil_x - бинарная функция, где 1 - нефть, 0 - нет нефти
# #### rock_x - тип горный породы, где 5 - песчаник, 2 - глина (остальные не помню)
# 
# #### При принятых гиперпараметрах на графиках заметно, что в целом алгоритм SST с SVD реагирует на изменений в гамма каротже по следующим причинам:
# ##### A) Типа горной породы (см. промежуток 1250-1500 метров)
# ##### Б) Нефтенасыщенности породы (см. промежуток 750-1000, 1750-2000)
# 
# #### Однако, при принятом коеффициенте аномалий quantile_rate = 0.95 и кол-во компонент n_components = 2 возникают ложные срабатывания.
# #### *Под ложными срабатываниями следует принимать изменения на "красном графике" score, которые не соответсвуют графикам по горным породам (5 классов) и нефтенасыщенности пород (бинарный класс)

# ##### Изменим коеффициент аномалий для борьбы с ложными срабатываниями и проверим результат

# In[8]:


if __name__ == "__main__":    
    runner(ts_window_length=100,lag=20,trajectory_window_length=20,
           quantile_rate=0.99,n_components=2,view=True,dynamic_mode=True)


# #### Исходя из визуального сравнения очевидно, что повышения порога аномалии не ведет к приемлимым результатам

# ## №2. Поработаем в оффлайн режиме с аналогичными гиперпараметрами

# In[9]:


if __name__ == "__main__":
    run(ts_window_length=100,lag=20,trajectory_window_length=20,
        quantile_rate=0.95,n_components=2,view=True,dynamic_mode=False)


# #### При принятых гиперпараметрах алгоритм SST с SVD реагирует (визуально) на изменений в гамма каротже аналогично динамическому режиму.
# 

# # Выводы

# #### a) Dynamic mode = True \ False - работает аналогично на данных гамма-каротажа. 
# #### b) Повышение коеффициента аномальности quantile_rate приводит к ухудшению результатов
# #### c) Алгоритм SST with svd, явно, реагирует на некоторые изменения в данных, которые не были отражены при экспертной интерпритации данных.*
# #### d) Алгоритм SST with svd обладает артефактом (в начале ряда на score, почти всегда, эквивалентен 1)
# #### e) Правило оптимального выбора гипепараметров trajectory_window_length, ts_window_length, lag для временного ряда не очевидно.** 
# 
# ###### *Необходимо кластеризировать данные при помощи алгоритмов типа K-means, spectral clastering (как? - Fedot или sclearn)
# ###### **Граничные условия могут быть следующие: ts_window_length < time_series/2 ; trajectory_window_length < int(ts_window_length/2) ; lag > 1
# ##### p.s. max значений на гамма-каротаже в петрофизики принимается за глину. Аналогично для песчаника (min значение -  нефтенасышенный песчаник)
# ##### p.s.s класс горной породы и нефтенасыщенность только по одному временному ряду не определяют

# In[ ]:




