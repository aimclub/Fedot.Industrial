import math
import random
import re
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class TSSplitter:
    """
    Class for splitting single time series based on anomaly dictionary into train and test parts
    for time series classification task.

    Args:
        time_series: time series to split
        anomaly_dict: dictionary where keys are anomaly labels and values are anomaly index ranges.
        strategy: strategy for splitting time series. Available strategies: 'frequent'.


    Attributes:
        selected_non_anomaly_intervals: list of non-anomaly intervals which were selected for splitting.

    Example:
        Create time series and anomaly dictionary::
            ts = np.random.rand(800)

            # of for multivariate time series
            # ts = [ts1, ts2, ts3]

            anomaly_d = {
                'anomaly1': '40:50, 60:80, 200:220, 410:420, 513:524, 641:645',
                'anomaly2': '130:170, 300:320, 400:410, 589:620, 715:720',
                'anomaly3': '500:530, 710:740',
                'anomaly4': '77:90, 98:112, 145:158, 290:322'}

        Split time series into train and test parts::
            from fedot_ind.core.operation.transformation.splitter import TSSplitter
            splitter = TSSplitter(ts, anomaly_d)
            train, test = splitter.split(plot=True, binarize=False)

    """

    def __init__(self, time_series: Union[np.ndarray, list],
                 anomaly_dict: dict,
                 strategy: str = 'frequent',
                 delimiter: str = ':'):

        self.delimiter = delimiter
        self.time_series = time_series
        self.anomaly_dict = anomaly_dict
        self.strategy = strategy
        self.selected_non_anomaly_intervals = []
        self.multivariate = self.__check_multivariate(time_series)

    def split(self, plot: bool = False, binarize: bool = False) -> tuple:
        """
        Method for splitting time series into train and test parts.

        Args:
            plot: if True, plot time series with anomaly intervals.
            binarize: if True, target will be binarized. Recommended for classification task if classes are imbalanced.

        Returns:
            tuple with train and test parts of time series ready for classification task with FedotIndustrial.

        """
        classes, intervals = self._get_anomaly_intervals()

        if self.strategy == 'frequent':
            freq_length = self._get_frequent_anomaly_length(intervals)
            transformed_intervals = self._transform_intervals(intervals, freq_length)
        else:
            raise ValueError('Unknown strategy')

        target, features = self._split_by_intervals(classes, transformed_intervals)

        non_anomaly_inters = self._get_non_anomaly_intervals(transformed_intervals)

        target, features = self.balance_with_non_anomaly(target, features, non_anomaly_inters)

        if binarize:
            target = self._binarize_target(target)

        if self.multivariate:
            features = self.convert_features_dimension(features)

        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            target,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            stratify=target)

        if plot and not self.multivariate:
            self.plot_classes_and_intervals(classes=classes,
                                            intervals=intervals,
                                            transformed_intervals=transformed_intervals)

        return (pd.DataFrame(X_train), y_train), (pd.DataFrame(X_test), y_test)

    def _get_anomaly_intervals(self) -> Tuple[List[str], List[list]]:

        labels = list(self.anomaly_dict.keys())
        label_intervals = []
        for anomaly_label in labels:
            # Extract the intervals using regular expressions
            s = self.anomaly_dict[anomaly_label]
            intervals = re.findall(r'\d+{0}\d+'.format(re.escape(self.delimiter)), s)
            result = [list(map(int, interval.split(self.delimiter))) for interval in intervals]

            label_intervals.append(result)
        return labels, label_intervals

    def _get_frequent_anomaly_length(self, intervals: List[list]):
        flat_intervals = []
        for sublist in intervals:
            for element in sublist:
                flat_intervals.append(element)

        lengths = list(map(lambda x: x[1] - x[0], flat_intervals))
        return max(set(lengths), key=lengths.count)

    def _transform_intervals(self, intervals, freq_len):
        new_intervals = []
        for class_inter in intervals:
            new_class_intervals = []
            for i in class_inter:
                current_len = i[1] - i[0]
                abs_diff = abs(current_len - freq_len)
                left_add = math.ceil(abs_diff / 2)
                right_add = math.floor(abs_diff / 2)
                # Calculate new borders

                # If current anomaly interval is less than frequent length,
                # we expand current interval to the size of frequent
                if current_len < freq_len:
                    left = i[0] - left_add
                    right = i[1] + right_add
                    # If left border is negative, shift right border to the right
                    if left < 0:
                        right += abs(left)
                        left = 0
                    # If right border is greater than time series length, shift left border to the left
                    if right > len(self.time_series):
                        left -= abs(right - len(self.time_series))
                        right = len(self.time_series)

                # If current anomaly interval is greater than frequent length,
                # we shrink current interval to the size of frequent
                elif current_len > freq_len:
                    left = i[0] + left_add
                    right = i[1] - right_add
                else:
                    left = i[0]
                    right = i[1]

                new_class_intervals.append([left, right])
            new_intervals.append(new_class_intervals)
        return new_intervals

    def _split_by_intervals(self, classes, transformed_intervals) -> Tuple[List[str], List[list]]:
        all_labels, all_ts = [], []

        for i, label in enumerate(classes):
            for inter in transformed_intervals[i]:
                all_labels.append(label)
                if self.multivariate:
                    all_ts.append(self.time_series[inter[0]:inter[1], :])
                else:
                    all_ts.append(self.time_series[inter[0]:inter[1]])
        return all_labels, all_ts

    def plot_classes_and_intervals(self, classes, intervals, transformed_intervals):
        fig, axes = plt.subplots(3, 1, figsize=(17, 7))
        fig.tight_layout()
        for ax in axes:
            ax.plot(self.time_series, color='black', linewidth=1, alpha=1)

        axes[0].set_title('Initial intervals')
        axes[1].set_title('Transformed intervals')
        axes[2].set_title('Non-anomaly samples')

        for i, label in enumerate(classes):
            for interval_ in transformed_intervals[i]:
                axes[1].axvspan(interval_[0], interval_[1], alpha=0.3, color='blue', edgecolor="black")
                axes[1].text(interval_[0], 0.5, label, fontsize=12, rotation=90)
            for interval in intervals[i]:
                axes[0].axvspan(interval[0], interval[1], alpha=0.3, color='red', edgecolor="black")
                axes[0].text(interval[0], 0.5, label, fontsize=12, rotation=90)

        if self.selected_non_anomaly_intervals is not None:
            for interval in self.selected_non_anomaly_intervals:
                axes[2].axvspan(interval[0], interval[1], alpha=0.3, color='green', edgecolor="black")
                axes[2].text(interval[0], 0.5, 'no_anomaly', fontsize=12, rotation=90)
        plt.show()


    def _binarize_target(self, target):
        new_target = []
        for label in target:
            if label == 'no_anomaly':
                new_target.append(0)
            else:
                new_target.append(1)
        return new_target

    def balance_with_non_anomaly(self, target, features, non_anomaly_intervals):
        number_of_anomalies = len(target)
        anomaly_len = len(features[0])
        non_anomaly_ts_list = []
        ts = self.time_series.copy()
        counter = 0
        while len(non_anomaly_ts_list) != number_of_anomalies and counter != number_of_anomalies * 100:
            seed = np.random.randint(1000)
            random.seed(seed)
            random_inter = random.choice(non_anomaly_intervals)
            cropped_ts_len = random_inter[1] - random_inter[0]
            counter += 1
            if cropped_ts_len < anomaly_len:
                continue
            random_start_index = random.randint(random_inter[0], random_inter[0] + cropped_ts_len - anomaly_len)
            stop_index = random_start_index + anomaly_len
            if self.multivariate:
                non_anomaly_ts = ts[random_start_index:stop_index, :]
            else:
                non_anomaly_ts = ts[random_start_index:stop_index]

            non_anomaly_ts_list.append(non_anomaly_ts)

            self.selected_non_anomaly_intervals.append([random_start_index, stop_index])

        if len(non_anomaly_ts_list) == 0:
            raise Exception('No non-anomaly intervals found')

        target.extend(['no_anomaly'] * len(non_anomaly_ts_list))
        features.extend(non_anomaly_ts_list)

        return target, features

    def _get_non_anomaly_intervals(self, anom_intervals: List[list]):
        flat_intervals_list = []
        for sublist in anom_intervals:
            for element in sublist:
                flat_intervals_list.append(element)

        if self.multivariate:
            series = pd.Series(self.time_series[:, 0]).copy()
        else:
            series = pd.Series(self.time_series).copy()

        for single_interval in flat_intervals_list:
            series[single_interval[0]:single_interval[1]] = np.nan

        non_nan_intervals = []
        for k, g in series.groupby((series.notnull() != series.shift().notnull()).cumsum()):
            if g.notnull().any():
                non_nan_intervals.append((g.index[0], g.index[-1]))

        return non_nan_intervals

    def __check_multivariate(self, time_series: np.ndarray):
        if isinstance(time_series, list):
            self.time_series = np.array(time_series).T
            return True
        return False

    def convert_features_dimension(self, features: np.ndarray):
        multi_dimension = features[0].shape[1]
        features_df = pd.DataFrame(columns=[f'dim{i}' for i in range(multi_dimension)],
                                   index=[i for i in range(len(features))])
        for row, sample in enumerate(features):
            for dim, measurement in enumerate(sample.T):
                _measurement = pd.Series(measurement)
                features_df.at[row, f'dim{dim}'] = _measurement

        return features_df


if __name__ == '__main__':

    uni_ts = np.random.rand(800)
    anomaly_d_uni = {'anomaly1': '40:50, 60:80, 200:220, 410:420, 513:524, 641:645',
                     'anomaly2': '130:170, 300:320, 400:410, 589:620, 715:720',
                     'anomaly3': '500:530, 710:740',
                     'anomaly4': '77:90, 98:112, 145:158, 290:322'}

    ts1 = np.arange(0, 100)
    multi_ts = [ts1, ts1 * 2, ts1 * 3]
    anomaly_d_multi = {'anomaly1': '0:5, 15:20, 22:24, 55:63, 70:90',
                       'anomaly2': '10:12, 15:16, 27:31, 44:50, 98:100',
                       'anomaly3': '0:3, 15:18, 19:24, 55:60, 85:90',}

    splitter_multi = TSSplitter(multi_ts, anomaly_d_multi)
    train_multi, test_multi = splitter_multi.split(plot=False, binarize=True)

    splitter_uni = TSSplitter(uni_ts, anomaly_d_uni)
    train_uni, test_uni = splitter_uni.split(plot=True, binarize=True)
    _ = 1
