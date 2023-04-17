from typing import Union

import numpy as np
import pandas as pd

from fedot_ind.core.architecture.settings.hyperparams import select_hyper_param

stat_methods = select_hyper_param('statistical_methods')
stat_methods_extra = select_hyper_param('statistical_methods_extra')
quantile_dict = {'q5_': 0.05,
                 'q25_': 0.25,
                 'q75_': 0.75,
                 'q95_': 0.95}


class StatFeaturesExtractor:
    """Class for generating statistical features for a given time series.

    """

    @staticmethod
    def create_baseline_features(feature_to_aggregation: Union[pd.DataFrame, np.ndarray]):
        stat_list = []
        column_name = []
        feature_to_aggregation = pd.DataFrame(feature_to_aggregation)
        if feature_to_aggregation.shape[0] != 1:
            feature_to_aggregation = feature_to_aggregation.T
        for method_name, method_func in stat_methods_extra.items():
            try:
                tmp = feature_to_aggregation.copy(deep=True)
            except Exception:
                tmp = feature_to_aggregation.copy()

            if method_name.startswith('q'):
                _ = []
                for idx, row in tmp.iterrows():
                    _.append(method_func(row, q=quantile_dict[method_name]))
                tmp = np.array(_)
                stat_list.append(tmp)
                column_name.append(method_name)
            elif method_name.startswith('l'):
                tmp = tmp.apply(method_func, axis=1)
                tmp = tmp.astype(int)
                stat_list.append(tmp.sum(axis=1).values)
                column_name.append(method_name)
            elif method_name.startswith('d'):
                tmp = tmp.apply(method_func, axis=1)
                stat_list.append(tmp.apply(np.mean).values)
                column_name.append(method_name + 'mean')
                stat_list.append(tmp.apply(np.min).values)
                column_name.append(method_name + 'min')
                stat_list.append(tmp.apply(np.max).values)
                column_name.append(method_name + 'max')
            else:
                stat_list.append(tmp.apply(method_func, axis=1).values)
                column_name.append(method_name)

            del tmp

        df_points_stat = pd.DataFrame(stat_list)
        df_points_stat = df_points_stat.T
        df_points_stat.columns = column_name

        return df_points_stat

    @staticmethod
    def create_features(feature_to_aggregation: Union[pd.DataFrame, np.ndarray]):
        stat_list = []
        column_name = []
        feature_to_aggregation = pd.DataFrame(feature_to_aggregation)
        for method_name, method_func in stat_methods.items():
            tmp = feature_to_aggregation.copy()

            if method_name.startswith('q'):
                for col in tmp.columns:
                    tmp[col] = method_func(tmp[col], q=quantile_dict[method_name])
                    tmp = tmp.drop_duplicates()
            else:
                tmp = pd.DataFrame(tmp.apply(method_func))
                tmp = tmp.T

            for feature in feature_to_aggregation.columns:
                column_name.append(method_name + str(feature))

            stat_list.append(tmp.values)

        df_points_stat = pd.DataFrame(np.concatenate(stat_list, axis=1))
        df_points_stat.columns = column_name
        return df_points_stat

    def _transform(self, X, intervals) -> np.ndarray:
        """
        Transform X for given intervals. Compute the mean, standard deviation and slope for given
        intervals of input data X.

        Args:
            X: input data
            intervals: list of intervals for which to compute the mean, standard deviation and slope

        Returns:
            Array of shape (len(X), 3 * len(intervals))
        """
        n_instances, _ = X.shape
        n_intervals, _ = intervals.shape
        transformed_x = np.empty(shape=(3 * n_intervals, n_instances), dtype=np.float32)
        for j in range(n_intervals):
            X_slice = X[:, intervals[j][0]: intervals[j][1]]
            means = np.mean(X_slice, axis=1)
            std_dev = np.std(X_slice, axis=1)
            slope = _slope(X_slice, axis=1)
            transformed_x[3 * j] = means
            transformed_x[3 * j + 1] = std_dev
            transformed_x[3 * j + 2] = slope

        return transformed_x.T

    @staticmethod
    def _get_intervals(n_intervals: int,
                       min_interval: int,
                       series_length: int,
                       rng) -> np.ndarray:
        """Generate random intervals for given parameters.

        Args:
            n_intervals: Number of intervals to generate
            min_interval: Minimum length of an interval
            series_length: Length of the time series
            rng: ...

        Returns:
            Array containing the intervals.

        """
        intervals = np.zeros((n_intervals, 2), dtype=int)
        for j in range(n_intervals):
            intervals[j][0] = rng.randint(series_length - min_interval)
            length = rng.randint(series_length - intervals[j][0] - 1)
            if length < min_interval:
                length = min_interval
            intervals[j][1] = intervals[j][0] + length
        return intervals

