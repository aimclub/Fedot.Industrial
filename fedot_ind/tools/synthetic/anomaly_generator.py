from typing import Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.tools.synthetic.anomalies import AddNoise, DecreaseDispersion, Dip, IncreaseDispersion, Peak, \
    ShiftTrendDOWN, \
    ShiftTrendUP
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator


class AnomalyGenerator:
    """
    AnomalyGenerator class is used to generate anomalies in time series data. It takes time series data as input and
    returns time series data with anomalies. Anomalies are generated based on anomaly_config parameter. It is a dict
    with anomaly class names as keys and anomaly parameters as values. Anomaly class names must be the same as anomaly
    class names in anomalies.py.

    Attributes:
        types: dict. Dict with anomaly class names as keys and anomaly parameters as values.
        anomaly_config: dict, default=BASE_CONFIG. Dict with anomaly class names as keys and anomaly parameters as
            values.
        taken_slots: np.ndarray, default=None. Array of 0 and 1. 1 means that this time slot is already taken by
            another anomaly.
        overlap: float, default=0.1. Argument of `generate` method. Defines the maximum overlap between anomalies.

    Example:
        First, we need to create an instance of AnomalyGenerator class with config as its argument where every anomaly
        type hyperparameters are defined::
            config = {'add_noise': {'level': 80,
                                    'number': 6,
                                    'noise_type': 'gaussian',
                                    'min_anomaly_length': 10,
                                    'max_anomaly_length': 20}}
            generator = AnomalyGenerator(config=config)

        Then we can generate anomalies in time series data using method `generate` which arguments are
        `time_series_data` (`np.array` of config for synthetic ts_data), `plot` and acceptable `overlap`::
            initial_ts, modified_ts, intervals = generator.generate(time_series_data=data,
                                                                    plot=True,
                                                                    overlap=0.1)

        This method returns initial time series data, modified time series data and dict with anomaly intervals.

    """

    def __init__(self, **params):
        self.types = {
            'decrease_dispersion': DecreaseDispersion,
            'increase_dispersion': IncreaseDispersion,
            'shift_trend_up': ShiftTrendUP,
            'shift_trend_down': ShiftTrendDOWN,
            'add_noise': AddNoise,
            'dip': Dip,
            'peak': Peak}

        self.anomaly_config = params.get(
            'config', ValueError('config must be defined'))
        self.taken_slots = None
        self.overlap = None

    def select_interval(self, max_length: int, min_length: int) -> tuple:
        ts_length = self.taken_slots.size
        start_idx = np.random.randint(max_length, ts_length - max_length)
        end_idx = start_idx + np.random.randint(min_length, max_length + 1)

        if self.taken_slots[start_idx:end_idx].mean() > self.overlap:
            return self.select_interval(max_length, min_length)
        else:
            self.taken_slots[start_idx:end_idx] = 1
            return start_idx, end_idx

    def generate(self,
                 time_series_data: Union[np.ndarray,
                                         dict],
                 plot: bool = False,
                 overlap: float = 0.1):
        """
        Generate anomalies in time series data.

        Args:
            time_series_data: either np.ndarray or dict with config for synthetic ts_data.
            plot: if True, plot initial and modified time series data with rectangle spans of anomalies.
            overlap: float, ``default=0.1``. Defines the maximum overlap between anomalies.

        Returns:
            returns initial time series data, modified time series data and dict with anomaly intervals.

        """
        if isinstance(time_series_data, dict):
            ts_generator = TimeSeriesGenerator(time_series_data)
            t_series = ts_generator.get_ts()

        elif isinstance(time_series_data, np.ndarray):
            t_series = time_series_data
        else:
            raise ValueError('time_series_data must be np.ndarray or dict')

        initial_ts = t_series.copy()
        anomaly_intervals_dict = {}

        self.taken_slots = pd.Series([0 for _ in t_series])
        self.overlap = overlap

        for anomaly_cls in self.anomaly_config.keys():
            n = self.anomaly_config[anomaly_cls]['number']
            anomaly_obj = self.types[anomaly_cls]
            params = self.anomaly_config[anomaly_cls]
            max_length = params.get('max_anomaly_length', ValueError(
                f'max_anomaly_length must be defined for {anomaly_cls} type'))
            min_length = params.get('min_anomaly_length', ValueError(
                f'min_anomaly_length must be defined for {anomaly_cls} type'))

            for i in range(n):
                start_idx, end_idx = self.select_interval(
                    max_length, min_length)
                t_series = anomaly_obj(params).get(
                    ts=t_series, interval=(start_idx, end_idx))

                if anomaly_cls in anomaly_intervals_dict:
                    anomaly_intervals_dict[anomaly_cls].append(
                        [start_idx, end_idx])
                else:
                    anomaly_intervals_dict[anomaly_cls] = [
                        [start_idx, end_idx]]

        if plot:
            self.plot_anomalies(initial_ts=initial_ts, modified_ts=t_series,
                                anomaly_intervals_dict=anomaly_intervals_dict)

        return initial_ts, t_series, anomaly_intervals_dict

    def plot_anomalies(self, initial_ts, modified_ts, anomaly_intervals_dict):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(modified_ts, label='Modified Time Series')
        ax.plot(initial_ts, label='Initial Time Series')
        ax.set_title('Time Series with Anomalies')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

        cmap = self.generate_colors(len(anomaly_intervals_dict.keys()))
        color_dict = {cls: color for cls, color in zip(
            anomaly_intervals_dict.keys(), cmap)}

        legend_patches = [
            patches.Patch(
                color=color_dict[cls],
                label=cls) for cls in anomaly_intervals_dict.keys()]

        for anomaly_class, intervals in anomaly_intervals_dict.items():
            for interval in intervals:
                start_idx, end_idx = interval
                ax.axvspan(start_idx, end_idx, alpha=0.3,
                           color=color_dict[anomaly_class])

        # Put a legend to the right of the current axis
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(
            1, 0.5), handles=set(legend_patches))
        plt.show()

    def generate_colors(self, num_colors: int) -> list:
        colormap = plt.cm.get_cmap('tab10')
        colors = [colormap(i) for i in range(num_colors)]
        return colors


if __name__ == '__main__':

    # config = {'decrease_dispersion': {'level': 70,
    #                                   'number': 2,
    #                                   'min_anomaly_length': 10,
    #                                   'max_anomaly_length': 15},
    #           'dip': {'level': 20,
    #                   'number': 2,
    #                   'min_anomaly_length': 10,
    #                   'max_anomaly_length': 20},
    #
    #           'peak': {'level': 2,
    #                    'number': 2,
    #                    'min_anomaly_length': 5,
    #                    'max_anomaly_length': 10},
    #           'increase_dispersion': {'level': 70,
    #                                   'number': 2,
    #                                   'min_anomaly_length': 30,
    #                                   'max_anomaly_length': 40},
    #           'shift_trend_up': {'level': 10,
    #                              'number': 2,
    #                              'min_anomaly_length': 10,
    #                              'max_anomaly_length': 20},
    #           'shift_trend_down': {'level': 10,
    #                                'number': 2,
    #                                'min_anomaly_length': 10,
    #                                'max_anomaly_length': 20},
    #           'add_noise': {'level': 80,
    #                         'number': 2,
    #                         'noise_type': 'uniform',
    #                         'min_anomaly_length': 50,
    #                         'max_anomaly_length': 60}
    #           }
    #
    # generator = AnomalyGenerator(config=config)
    #
    # ts_conf = {'ts_type': 'sin',
    #            'ts_length': 2000}
    #
    # init_ts, mot_ts, inters = generator.generate(time_series_data=ts_conf,
    #                                              plot=True,
    #                                              overlap=0.1)

    synth_ts = {'ts_type': 'sin',
                'length': 1000,
                'amplitude': 10,
                'period': 500}

    anomaly_config = {'dip': {'level': 20,
                              'number': 5,
                              'min_anomaly_length': 10,
                              'max_anomaly_length': 20},
                      'peak': {'level': 2,
                               'number': 5,
                               'min_anomaly_length': 5,
                               'max_anomaly_length': 10},
                      'decrease_dispersion': {'level': 70,
                                              'number': 2,
                                              'min_anomaly_length': 10,
                                              'max_anomaly_length': 15},
                      # 'increase_dispersion': {'level': 50,
                      #                         'number': 2,
                      #                         'min_anomaly_length': 10,
                      #                         'max_anomaly_length': 15},
                      # 'shift_trend_up': {'level': 10,
                      #                    'number': 2,
                      #                    'min_anomaly_length': 10,
                      #                    'max_anomaly_length': 20},
                      # 'add_noise': {'level': 80,
                      #               'number': 2,
                      #               'noise_type': 'uniform',
                      #               'min_anomaly_length': 10,
                      #               'max_anomaly_length': 20}
                      }

    generator = AnomalyGenerator(config=anomaly_config)

    init_synth_ts, mod_synth_ts, synth_inters = generator.generate(
        time_series_data=synth_ts, plot=True, overlap=0.1)
    _ = 1


def generate_univariate_anomaly_data(n_samples: int = 1000, n_anomalies: int = 50,
                                     random_state: int = 42) -> tuple:
    """Генерация одномерных временных рядов с аномалиями"""
    np.random.seed(random_state)

    # Нормальные данные: синус + шум
    time = np.linspace(0, 4 * np.pi, n_samples)
    normal_data = np.sin(time) + 0.1 * np.random.normal(size=n_samples)

    # Создание аномалий
    anomalies_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    anomalies_data = normal_data.copy()

    for idx in anomalies_indices:
        # Разные типы аномалий
        anomaly_type = np.random.choice(['spike', 'level_shift', 'variance_change'])

        if anomaly_type == 'spike':
            anomalies_data[idx] += 3 * np.random.randn()
        elif anomaly_type == 'level_shift':
            anomalies_data[idx:min(idx + 5, n_samples)] += 2
        else:  # variance_change
            window = slice(max(0, idx - 3), min(n_samples, idx + 4))
            anomalies_data[window] += 1.5 * np.random.randn(len(anomalies_data[window]))

    # Метки аномалий
    labels = np.zeros(n_samples)
    labels[anomalies_indices] = 1
    anomalies_data = anomalies_data.reshape(-1, 1)
    input_data = InputData(idx=np.arange(len(anomalies_data)),
                           features=anomalies_data,
                           target=labels,
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


def generate_multivariate_anomaly_data(n_samples: int = 1000, n_features: int = 3,
                                       n_anomalies: int = 50, random_state: int = 42) -> tuple:
    """Генерация многомерных временных рядов с аномалиями"""
    np.random.seed(random_state)

    # Базовые сигналы
    time = np.linspace(0, 4 * np.pi, n_samples)
    data = np.column_stack([
        np.sin(time),
        np.cos(0.5 * time),
        np.sin(2 * time + 1)
    ])

    # Добавление коррелированного шума
    covariance = np.array([[1.0, 0.7, 0.3],
                           [0.7, 1.0, 0.5],
                           [0.3, 0.5, 1.0]])

    noise = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=covariance,
        size=n_samples
    )

    data += 0.1 * noise

    # Добавление аномалий
    anomalies_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    labels = np.zeros(n_samples)

    for idx in anomalies_indices:
        labels[idx] = 1
        # Аномалии в разных измерениях
        feature_idx = np.random.randint(0, n_features)
        anomaly_magnitude = 2 + np.random.randn()
        data[idx, feature_idx] += anomaly_magnitude

    return data, labels


def generate_online_anomaly_data():
    # Генерация данных с дрейфом
    np.random.seed(42)
    n_samples = 1000

    # Первая часть данных
    time1 = np.linspace(0, 2 * np.pi, n_samples // 2)
    data1 = np.sin(time1).reshape(-1, 1) + 0.1 * np.random.normal(size=(n_samples // 2, 1))

    # Вторая часть данных с дрейфом
    time2 = np.linspace(2 * np.pi, 4 * np.pi, n_samples // 2)
    data2 = 1.5 * np.sin(time2).reshape(-1, 1) + 0.2 * np.random.normal(size=(n_samples // 2, 1))

    X_online = np.vstack([data1, data2])
    return X_online
