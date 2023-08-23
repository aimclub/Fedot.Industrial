from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from fedot_ind.api.main import FedotIndustrial


def generate_time_series(ts_length: int = 500,
                         dimension: int = 1,
                         num_anomaly_classes: int = 4,
                         num_of_anomalies: int = 20,
                         min_anomaly_length: int = 5,
                         max_anomaly_length: int = 15):
    np.random.seed(42)

    if dimension == 1:
        time_series = np.random.normal(0, 1, ts_length)
    else:
        time_series = np.vstack([np.random.normal(0, 1, ts_length) for _ in range(dimension)]).swapaxes(1, 0)
    anomaly_classes = [f'anomaly{i + 1}' for i in range(num_anomaly_classes)]

    anomaly_intervals = {}

    for i in range(num_of_anomalies):
        anomaly_class = np.random.choice(anomaly_classes)

        start_idx = np.random.randint(max_anomaly_length,
                                      ts_length - max_anomaly_length)

        end_idx = start_idx + np.random.randint(min_anomaly_length,
                                                max_anomaly_length + 1)

        anomaly = np.random.normal(int(anomaly_class[-1]), 1, end_idx - start_idx)

        if dimension == 1:
            time_series[start_idx:end_idx] += anomaly
        else:
            for i in range(time_series.shape[1]):
                time_series[start_idx:end_idx, i] += anomaly

        if anomaly_class in anomaly_intervals:
            anomaly_intervals[anomaly_class].append([start_idx, end_idx])
        else:
            anomaly_intervals[anomaly_class] = [[start_idx, end_idx]]

    return time_series, anomaly_intervals


def plot_anomalies(series: np.array, anomaly_dict: Dict):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(series)
    ax.set_title('Time Series with Anomalies')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

    def generate_colors(num_colors):
        colormap = plt.cm.get_cmap('tab10')
        colors = [colormap(i) for i in range(num_colors)]
        return colors

    cmap = generate_colors(len(anomaly_dict.keys()))
    color_dict = {cls: color for cls, color in zip(anomaly_dict.keys(), cmap)}

    legend_patches = [mpatches.Patch(color=color_dict[cls],
                                     label=cls) for cls in anomaly_dict.keys()]

    for anomaly_class, intervals in anomaly_dict.items():
        for interval in intervals:
            start_idx, end_idx = map(int, interval)
            ax.axvspan(start_idx, end_idx, alpha=0.3, color=color_dict[anomaly_class])

    plt.legend(handles=set(legend_patches))
    plt.show()


def split_series(series, anomaly_dict, test_part: int = 200):
    time_series_train = series[:-test_part]
    time_series_test = series[-test_part:]

    anomaly_intervals_train = {}
    anomaly_intervals_test = {}
    for anomaly_class in anomaly_dict:
        single_class_anomalies_train = []
        single_class_anomalies_test = []
        for interval in anomaly_dict[anomaly_class]:
            if interval[1] > len(time_series_train):
                single_class_anomalies_test.append(
                    [interval[0] - len(time_series_train), interval[1] - len(time_series_train)])
            else:
                single_class_anomalies_train.append(interval)
        anomaly_intervals_train[anomaly_class] = single_class_anomalies_train
        anomaly_intervals_test[anomaly_class] = single_class_anomalies_test
    return time_series_train, anomaly_intervals_train, time_series_test, anomaly_intervals_test


def convert_anomalies_dict_to_points(series: np.array, anomaly_dict: Dict) -> np.array:
    points = np.array(['no_anomaly' for _ in range(len(series))], dtype=object)
    for anomaly_class in anomaly_dict:
        for interval in anomaly_dict[anomaly_class]:
            points[interval[0]:interval[1]] = anomaly_class
    return points


if __name__ == "__main__":
    time_series, anomaly_intervals = generate_time_series(
        ts_length=1000,
        dimension=5,
        num_anomaly_classes=7,
        num_of_anomalies=50)

    # plot_anomalies(time_series, anomaly_intervals)

    series_train, anomaly_train, series_test, anomaly_test = split_series(time_series, anomaly_intervals, test_part=300)

    point_test = convert_anomalies_dict_to_points(series_test, anomaly_test)

    industrial = FedotIndustrial(task='anomaly_detection',
                                 dataset='custom_dataset',
                                 strategy='fedot_preset',
                                 use_cache=False,
                                 timeout=0.5,
                                 n_jobs=1,
                                 logging_level=20,
                                 output_folder='.')

    model = industrial.fit(features=series_train,
                           anomaly_dict=anomaly_train)

    labels = industrial.predict(features=series_test)
    probs = industrial.predict_proba(features=series_test)

    industrial.solver.get_metrics(target=point_test,
                                  metric_names=['f1', 'roc_auc'])

    print(classification_report(point_test, labels))

    industrial.solver.plot_prediction(series_test, point_test)
