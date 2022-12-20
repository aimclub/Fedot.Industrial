from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

from core.models.spectral.spectrum_decomposer import SpectrumDecomposer
from core.models.statistical.stat_features_extractor import StatFeaturesExtractor


class SingularSpectrumTransformation:
    """SingularSpectrumTransformation class.

    Change point detection with Singular Spectrum Transformation.

    Note:
        In case of 1D time series to find appropriate hyperparameters value
         for ts_window_length and trajectory_window_length
         are recommended to use WSSAlgorithms.py, WindowSizeSelection class.

    Args:
        time_series: time series sequences to study.
        ts_window_length: window length of vectors in Hankel matrices.
        trajectory_window_length: window length of subsequences in ts_window_length.
        lag: interval between the nearest ts_window_length sequences.
        n_components: PCA components number describes changes in time-data (usually 1,2 or 3).
        quantile_rate: threshold coefficient for change point detection (0,95 recommended).
        view: if true, not filtered score will be returned.
        is_scaled: if false, min-max scaling will be applied.

    """

    def __init__(self,
                 time_series: Union[list, np.array] = None,
                 quantile_rate: float = None,
                 trajectory_window_length: int = None,
                 ts_window_length: int = None,
                 lag: int = None,
                 is_scaled: bool = False,
                 view: bool = True,
                 n_components: int = None):

        self.spectrum_extractor = SpectrumDecomposer(time_series=time_series, window_length=trajectory_window_length)
        self.ts = time_series
        self.ts_window_length = ts_window_length
        self.trajectory_window_length = trajectory_window_length  # equal self.L
        self.L = trajectory_window_length  # equal self.trajectory_window_length or self.L
        self.K = ts_window_length - self.L + 1
        self.quantile_rate = quantile_rate
        self.n_components = n_components
        self.view = view
        self.aggregator = StatFeaturesExtractor()

        if self.ts_window_length is None:
            self.ts_window_length = self.L

        self.lag = lag

        if self.lag is None:
            self.lag = np.round(self.ts_window_length / 2)

        self.is_scaled = is_scaled

        if self.quantile_rate is None:
            self.quantile_rate = 0.95

        self.n_components = None

    def score_offline_2d(self, dynamic_mode: bool = True) -> list:
        """Run function of offline score calculation. FOR 2D or more D.

        Args:
            dynamic_mode: type of model to check differences in the sequence

        Returns:
            filtered_score: represented a list of values with 0 and 1 where 1 is an anomaly if view is True

        """
        if not self.is_scaled:
            x_scaled = MinMaxScaler(feature_range=(1, 2)) \
                           .fit_transform(self.ts.reshape(-1, 1))[:, 0]
        else:
            x_scaled = self.ts
        return self._score_offline_2d(x_scaled, dynamic_mode=dynamic_mode)

    def score_offline_2d_average(self, dynamic_mode: bool = True) -> list:
        """Run function of offline score calculation with average features. FOR 2D or more D.

        Args:
            dynamic_mode: type of model to check differences in the sequence

        Returns:
            filtered_score: represented a list of values with 0 and 1 where 1 is an anomaly if view is True

        """
        if not self.is_scaled:
            x_scaled = MinMaxScaler(feature_range=(1, 2)) \
                           .fit_transform(self.ts.reshape(-1, 1))[:, 0]
        else:
            x_scaled = self.ts
        return self._score_offline_2d_average(x_scaled, dynamic_mode=dynamic_mode)

    def _get_window_from_ts_complex(self, ts_complex: list, start: int, end: int) -> list:
        window: list = []
        if start < 0 or start >= len(ts_complex[0]):
            raise ValueError("Start value is less than zero or more then length of time series!")
        if end < 0 or end >= len(ts_complex[0]):
            raise ValueError("End value is less than zero or more then length of time series!")

        if end < start:
            raise ValueError("Start > End!")
        for _ in ts_complex:
            window.append([])
        for i in range(start, end):
            for j in range(len(ts_complex)):
                window[j].append(ts_complex[j][i])
        return window

    def _score_offline_2d_average(self, x_scaled: list, dynamic_mode: bool = True) -> list:
        """Core implementation of offline score calculation with average features. FOR 2D or more D.

        Args:
            x_scaled: normalized time series if is_scaled False
            dynamic_mode: type of model to check differences in the sequence

        Returns:
            filtered_score: represented a list of values with 0 and 1 where 1 is an anomaly if view is True

        """
        score_list = []
        if not dynamic_mode:
            step = self.lag
            start_idx = step
            end_idx = len(x_scaled[0]) - step
            horm_hist = None
            average_features = self.average_features_2d(end_idx, start_idx, x_scaled)

            for current_index in range(start_idx, end_idx, step):
                current_features = self.current_features_2d(current_index, step, x_scaled)
                current_features = np.reshape(current_features, (len(x_scaled), 7))
                if horm_hist is None:
                    horm_hist = np.linalg.norm(distance_matrix(average_features.T, current_features.T), 2)
                score_list.append(np.linalg.norm(distance_matrix(average_features.T, current_features.T), 2))
        else:
            start_idx = self.ts_window_length + self.lag
            end_idx = len(x_scaled[0]) - self.ts_window_length - 1
            horm_hist = None
            x_history_arr = []
            average_features = self.average_features_2d(end_idx, start_idx, x_scaled)
            x_history = None
            for ts_number in range(len(average_features)):
                x_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=average_features[ts_number],
                    K=self.ts_window_length - self.L + 1,
                    L=len(average_features[ts_number]))
                x_history_arr.extend(x_history)

            for t in range(start_idx, end_idx, self.lag):  # get Hankel matrix
                if horm_hist is None:
                    horm_hist = np.linalg.norm(x_history, 1)

                current_features = self.current_features_2d(t, self.ts_window_length, x_scaled)
                current_features = current_features.reshape(current_features.shape[0],
                                                            (current_features.shape[1] * current_features.shape[2]))

                x_test_arr = []
                x_test = None
                for ts_number in range(len(current_features)):
                    x_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                        timeseries=current_features[ts_number],
                        K=self.ts_window_length - self.L + 1,
                        L=len(current_features[ts_number]))
                    x_test_arr.extend(x_test)

                if self.n_components is None:
                    self._n_components(x_test, x_history)

                score_list.append(
                    self._sst_svd(x_test_arr, x_history_arr))
        score_diff = np.diff(score_list)
        q_95 = np.quantile(score_diff, self.quantile_rate)
        filtered_score = score_diff
        self.n_components = None
        if self.view:
            filtered_score = list(map(lambda _x: 1 if _x > q_95 else 0, score_diff))
        return filtered_score

    def average_features_2d(self, end_idx: int = None, start_idx: int = None, x_scaled: list = None) -> list:
        """Core implementation of extracting average features. FOR 2D or more D.

        Args:
            end_idx: end index for time series
            start_idx: start index for time series
            x_scaled: normalized time series if is_scaled False

        Returns:
            filtered_score: represented a list of values with 0 and 1 where 1 is an anomaly if view is True

        """
        temp_average_features = [[]] * len(x_scaled)
        for current_index in range(start_idx, end_idx, self.lag):
            current_features = self.current_features_2d(current_index, self.lag, x_scaled)
            for i in range(len(x_scaled)):
                temp_average_features[i].append(current_features[i])
        average_features = self.features_average(temp_average_features)
        return average_features

    def current_features_2d(self, current_index: int, step: int, x_scaled: list) -> np.asarray:
        """Implementation of offline score calculation function - features. FOR 2D or more D.

        Args:
            x_scaled: normalized time series if is_scaled False
            current_index: current index
            step: lag between the nearest sequences

        Returns:
            current_features: represented a list of features in chosen subsequence

        """
        current_window = self._get_window_from_ts_complex(x_scaled, current_index, current_index + step)
        current_features = self._get_features_vector_from_window(current_window)
        current_features = np.asarray(current_features)
        return current_features

    def features_average(self, temp_average_features: list) -> list:
        """Implementation of offline score calculation function - features_average. FOR 2D or more D.

        Args:
            temp_average_features: list of matrixes for a range of indexes over time series

        Returns:
            average_features: a list of averages features in chosen subsequence

        """
        temp_average_features = np.array(temp_average_features)
        average_features = []
        for i in range(len(temp_average_features)):
            average_features.append(np.average(temp_average_features[i], axis=0))
        average_features = np.array(average_features)
        average_features = average_features.reshape(
            average_features.shape[0],
            (average_features.shape[1] * average_features.shape[2])
        )
        return average_features

    def _score_offline_2d(self, x_scaled: list = None, dynamic_mode: bool = True) -> list:
        """Core implementation of offline score calculation. FOR 2D or more D.

        Args:
            x_scaled: normalized time series if is_scaled False
            dynamic_mode: type of model to check differences in the sequence

        Returns:
            filtered_score: represented a list of values with 0 and 1 where 1 is an anomaly if view is True

        """
        norm_list_real = []
        horm_hist = None
        if dynamic_mode:
            step = 1 * self.ts_window_length
            start_idx = step
            end_idx = len(x_scaled[0]) - step

            current_index = start_idx
            first_window = self._get_window_from_ts_complex(x_scaled, current_index, current_index + step)
            first_features = self._get_features_vector_from_window(first_window)
            first_features = np.asarray(first_features)
            first_features = np.reshape(first_features, (len(x_scaled), 7))

            for current_index in range(start_idx, end_idx, step):
                current_window = self._get_window_from_ts_complex(x_scaled, current_index, current_index + step)
                current_features = self._get_features_vector_from_window(current_window)
                current_features = np.asarray(current_features)
                current_features = np.reshape(current_features, (len(x_scaled), 7))
                if horm_hist is None:
                    horm_hist = np.linalg.norm(distance_matrix(first_features.T, current_features.T), 2)
                norm_list_real.append(np.linalg.norm(distance_matrix(first_features.T, current_features.T), 2))
        else:
            raise ValueError("Function dose not work when dynamic == False (FOR 2D or more D)")

        score_list = [horm_hist] + norm_list_real
        score_diff = np.diff(score_list)
        q_95 = np.quantile(score_diff, self.quantile_rate)
        self.n_components = None
        filtered_score = score_diff
        if self.view:
            filtered_score = list(map(lambda _x: 1 if _x > q_95 else 0, score_diff))
        return filtered_score

    def score_offline(self, _x: list = None, dynamic_mode: bool = True) -> list:
        """Calculate anomaly score (offline) for 1D.

        Args:
            dynamic_mode: mode for SST metrics calculation.
            _x: input 1D time series data.

        Returns
            score: 1d array change point score with 1 and 0 if view True.

        """
        _x = self.ts
        if not self.is_scaled:
            x_scaled = MinMaxScaler(feature_range=(1, 2)) \
                           .fit_transform(_x.reshape(-1, 1))[:, 0]
        else:
            x_scaled = _x
        return self._score_offline(x_scaled, dynamic_mode=dynamic_mode)

    def _score_offline(self, _x: list = None, dynamic_mode=True) -> list:
        """Core implementation of offline score calculation.

        Args:
            dynamic_mode: mode for SST metrics calculation.
            _x: input 1D time series data.

        Returns
            score: 1d array change point score with 1 and 0 if view True.

        """
        _K, _L, lag = self.K, self.L, self.lag
        start_idx, end_idx = _L + lag + 1, _x.size + 1
        score_list = np.zeros_like(_x)

        if dynamic_mode:
            for t in range(start_idx, end_idx):
                start_idx_hist, end_idx_hist = t - _L - lag, t - lag
                start_idx_test, end_idx_test = t - _L, t
                x_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=self.ts[start_idx_hist:end_idx_hist], K=_K, L=_L)
                x_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=self.ts[start_idx_test:end_idx_test], K=_K, L=_L)
                if self.n_components is None:
                    self._n_components(x_history, _L)
                score_list[t - 1] = self._sst_svd(x_test, x_history)

        else:
            x_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                timeseries=self.ts[:start_idx], K=_K, L=_L)
            for t in range(start_idx, end_idx):
                start_idx_test, end_idx_test = t - _L, t
                x_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=self.ts[start_idx_test:end_idx_test], K=_K, L=_L)
                if self.n_components is None:
                    self._n_components(x_history, _L)
                score_list[t - 1] = self._sst_svd(x_test, x_history)

        score_diff = np.diff(score_list)
        q_95 = np.quantile(score_diff, self.quantile_rate)
        filtered_score = score_diff
        if self.view:
            filtered_score = list(map(lambda _x: 1 if _x > q_95 else 0, score_diff))
        return filtered_score

    def _n_components(self, x_history: list = None, _l: int = None):
        """Number of relevant components which represent changing in time series.

        Args:
            x_history: historical matrix of features
            _l: window length of subsequences in ts_window_length.

        Returns
            self.n_components: number of relevant components

        """
        _s = self.spectrum_extractor.decompose_trajectory_matrix(x_history)[1]
        var, exp_var_by_component = self.spectrum_extractor.sv_to_explained_variance_ratio(_s, _l)
        exp_var_by_component = list(filter(lambda _s: _s > 0.05, exp_var_by_component))
        self.n_components = len(exp_var_by_component)

    def _sst_svd(self, x_test: list = None, x_history: list = None) -> float:
        """Singular value decomposition to count distance score between matrixes

        Args:
            x_test: current matrix of features
            x_history: historical matrix of features

        Returns
            1 - s[0]: distance score between two matrixes

        """
        n_components = self.n_components
        u_test, s_test, _ = np.linalg.svd(x_test, full_matrices=False)
        u_history, s_hist, _ = np.linalg.svd(x_history, full_matrices=False)
        s_cov = u_test[:, :n_components].T @ u_history[:, :n_components]
        u_cov, s, _ = np.linalg.svd(s_cov, full_matrices=False)
        return 1 - s[0]

    def _get_features_vector_from_window(self, window: list) -> list:
        features: list = []
        for ts in window:
            temp_features = self._generate_features_from_one_ts(ts)
            features.append(temp_features)
        return features

    def _generate_features_from_one_ts(self, time_series) -> list:
        time_series = np.asarray(time_series)
        time_series = np.reshape(time_series, (1, time_series.shape[0]))
        time_series = pd.DataFrame(time_series, dtype=float)
        feat = self.aggregator.create_baseline_features(time_series)
        """
        mean_ 0
        median_ 1
        lambda_less_zero 2 ??
        std_ 3 -/+
        var_ 4 -/+
        max 5
        min 6
        q5_ 7
        q25_ 8
        q75_ 9
        q95_ 10
        sum_ 11
        """

        keys = feat.columns
        values = feat._values
        out_values = []
        out_values.append(values[0][0])
        out_values.append(values[0][1])
        out_values.append(values[0][2])
        # out_values.append(values[0][3])
        # out_values.append(values[0][4])
        # out_values.append(values[0][5])
        # out_values.append(values[0][6])
        out_values.append(values[0][7])
        out_values.append(values[0][8])
        out_values.append(values[0][9])
        out_values.append(values[0][10])
        # out_values.append(values[0][11])
        out_values = np.array(out_values)
        reshaped_values = [out_values]
        return reshaped_values


if __name__ == '__main__':
    x0 = 1 * np.ones(1000) + np.random.rand(1000) * 1
    x1 = 3 * np.ones(1000) + np.random.rand(1000) * 2
    x2 = 5 * np.ones(1000) + np.random.rand(1000) * 1.5
    x = np.hstack([x0, x1, x2])
    x += np.random.rand(x.size)


    def plot_data_and_score(raw_data, score_list):
        f, ax = plt.subplots(2, 1, figsize=(20, 10))
        ax[0].plot(raw_data)
        ax[0].set_title("raw data")
        ax[1].plot(score_list, "r")
        ax[1].set_title("score")
        f.show()


    scorer = SingularSpectrumTransformation(time_series=x,
                                            ts_window_length=100,
                                            lag=10,
                                            trajectory_window_length=30)
    score = scorer.score_offline(dynamic_mode=False)
    plot_data_and_score(x, score)
    _ = 1
