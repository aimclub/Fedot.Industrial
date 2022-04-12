from typing import Union
import numpy as np
import pandas as pd
from pipe import *
from core.operation.settings.Hyperparams import *
from core.operation.utils.Decorators import type_check_decorator
import copy

stat_methods = ParamSelector('statistical_methods')
stat_methods_extra = ParamSelector('statistical_methods_extra')
supported_types = (pd.Series, np.ndarray, list)
quantile_dict = {'q5_': 0.05,
                 'q25_': 0.25,
                 'q75_': 0.75,
                 'q95_': 0.95
                 }


class AggregationFeatures:

    def create_baseline_features(self, feature_to_aggregation: Union[pd.DataFrame, np.ndarray]):
        stat_list = []
        column_name = []
        for method_name, method_func in stat_methods_extra.items():
            tmp = feature_to_aggregation.copy(deep=True)

            if method_name.startswith('q'):
                _ = []
                for idx, row in tmp.iterrows():
                    _.append(method_func(row, q=quantile_dict[method_name]))
                tmp = np.array(_)
                stat_list.append(tmp)
            elif method_name.startswith('l'):
                tmp = tmp.apply(method_func, axis=1)
                tmp = tmp.astype(int)
                stat_list.append(tmp.sum(axis=1).values)
            else:
                stat_list.append(tmp.apply(method_func, axis=1).values)

            column_name.append(method_name)

        df_points_stat = pd.DataFrame(stat_list)
        df_points_stat = df_points_stat.T
        df_points_stat.columns = column_name
        feature_disp = df_points_stat.var()

        for col in feature_disp.index.values:
            if feature_disp[col] < 0.001:
                del df_points_stat[col]

        return df_points_stat

    # @type_check_decorator(types_list=supported_types)
    def create_features(self, feature_to_aggregation: Union[pd.DataFrame, np.ndarray]):
        stat_list = []
        column_name = []
        for method_name, method_func in stat_methods.items():
            tmp = feature_to_aggregation.copy(deep=True)

            if method_name.startswith('q'):
                for col in tmp.columns:
                    tmp[col] = method_func(tmp[col], q=quantile_dict[method_name])
                    tmp = tmp.drop_duplicates()
            else:
                tmp = pd.DataFrame(tmp.apply(method_func))
                tmp = tmp.T

            for feature in feature_to_aggregation.columns:
                column_name.append(method_name + feature)

            stat_list.append(tmp.values)

        df_points_stat = pd.DataFrame(np.concatenate(stat_list, axis=1))
        df_points_stat.columns = column_name
        return df_points_stat

    def _transform(X, intervals):
        """Transform X for given intervals.

        Compute the mean, standard deviation and slope for given intervals of input data X.

        Parameters
        ----------
        Xt: np.ndarray or pd.DataFrame
            Panel data to transform.
        intervals : np.ndarray
            Intervals containing start and end values.

        Returns
        -------
        Xt: np.ndarray or pd.DataFrame
         Transformed X, containing the mean, std and slope for each interval
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

    def _get_intervals(n_intervals, min_interval, series_length, rng):
        """Generate random intervals for given parameters."""
        intervals = np.zeros((n_intervals, 2), dtype=int)
        for j in range(n_intervals):
            intervals[j][0] = rng.randint(series_length - min_interval)
            length = rng.randint(series_length - intervals[j][0] - 1)
            if length < min_interval:
                length = min_interval
            intervals[j][1] = intervals[j][0] + length
        return intervals
