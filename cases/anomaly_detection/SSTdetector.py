import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

from core.models.spectral.spectrum_decomposer import SpectrumDecomposer
from core.models.statistical.stat_features_extractor import StatFeaturesExtractor

import matplotlib.pyplot as plt


class SingularSpectrumTransformation:
    """SingularSpectrumTransformation class.

    Change point detection with Singular Spectrum Transformation.
    Parameters
    ----------
    ts_window_length : int
        window length of Hankel matrix.
    trajectory_window_length : int
        window lenght of vectors in Hankel matrixes
    lag : int
        interval between history Hankel matrix and test Hankel matrix.
    is_scaled : bool
        if false, min-max scaling will be applied(recommended).
    quantile_rate : float
        anomaly coefficient for change point detection(0,95 recommended).
    n_components : int
        PCA components number describes changes in time-data (usually we have 1,2 or 3).
    view : bool
        test parameter to plot data for experiments (default == True)
    """
    def __init__(self,
                 time_series,
                 quantile_rate: float = None,
                 trajectory_window_length: int = None,
                 ts_window_length: int = None,
                 lag: int = None,
                 is_scaled=False,
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
            self.lag = np.round(self.ts_window_length/2)

        self.is_scaled = is_scaled

        if self.quantile_rate is None:
            self.quantile_rate = 0.95

        self.n_components = None

    def score_offline_2d(self, dynamic_mode: bool = True):
        if not self.is_scaled:
            x_scaled = MinMaxScaler(feature_range=(1, 2)) \
                           .fit_transform(self.ts.reshape(-1, 1))[:, 0]
        else:
            x_scaled = self.ts

        return self._score_offline_2d(x_scaled, dynamic_mode=dynamic_mode)

    def score_offline_2d_average(self, dynamic_mode: bool = True):
        if not self.is_scaled:
            x_scaled = MinMaxScaler(feature_range=(1, 2)) \
                           .fit_transform(self.ts.reshape(-1, 1))[:, 0]
        else:
            x_scaled = self.ts
        score = self._score_offline_2d_average(x_scaled, dynamic_mode=dynamic_mode)
        return score

    def _get_window_from_ts_complex(self, ts_complex, start: int, end: int) -> list:
        window: list = []
        if start < 0 or start >= len(ts_complex[0]):
            raise ValueError("Start value is less than zero or more then lenght of time series!")
        if end < 0 or end >= len(ts_complex[0]):
            raise ValueError("End value is less than zero or more then lenght of time series!")

        if end < start:
            raise ValueError("Start > End!")
        for _ in ts_complex:
            window.append([])
        for i in range(start, end):
            for j in range(len(ts_complex)):
                window[j].append(ts_complex[j][i])
        return window

    def _score_offline_2d_average(self, x_scaled, dynamic_mode: bool = True):
        """Core implementation of offline score calculation. FOR 2D or more D"""
        if not dynamic_mode:
            score_list = np.zeros_like(x_scaled[0])
            step = 1 * self.ts_window_length
            start_idx = step
            step = self.lag
            end_idx = len(x_scaled[0]) - step
            horm_hist = None
            temp_average_features = [[]] * len(x_scaled)
            for current_index in range(start_idx, end_idx, step):
                current_window = self._get_window_from_ts_complex(x_scaled, current_index, current_index + step)
                current_features = self._get_features_vector_from_window(current_window)
                current_features = np.asarray(current_features)
                for i in range(len(x_scaled)):
                    temp_average_features[i].append(current_features[i])
            average_features = self.features_average_(temp_average_features)  # 1

            for current_index in range(start_idx, end_idx, step):
                current_window = self._get_window_from_ts_complex(x_scaled, current_index, current_index + step)
                current_features = self._get_features_vector_from_window(current_window)
                current_features = np.asarray(current_features)
                current_features = np.reshape(current_features, (len(x_scaled), 7))
                if horm_hist is None:
                    horm_hist = np.linalg.norm(distance_matrix(average_features.T, current_features.T), 2)
                score_list.append(np.linalg.norm(distance_matrix(average_features.T, current_features.T), 2))
        else:
            score_list = []
            start_idx = self.ts_window_length + self.lag
            end_idx = len(x_scaled[0]) - self.ts_window_length - 1
            horm_hist = None
            x_history_arr = []
            temp_average_features = [[]] * len(x_scaled)
            for current_index in range(start_idx, end_idx, self.lag):
                current_window = self._get_window_from_ts_complex(x_scaled, current_index, current_index + self.lag)
                current_features = self._get_features_vector_from_window(current_window)
                current_features = np.asarray(current_features)
                for i in range(len(x_scaled)):
                    temp_average_features[i].append(current_features[i])
            average_features = self.features_average_(temp_average_features)
            for ts_number in range(len(average_features)):
                x_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=average_features[ts_number],
                    K=self.ts_window_length - self.L + 1,
                    L=len(average_features[ts_number]))
                x_history_arr.extend(x_history)

            for t in range(start_idx, end_idx, self.lag):  # get Hankel matrix
                if horm_hist is None:
                    horm_hist = np.linalg.norm(x_history, 1)

                current_window = self._get_window_from_ts_complex(x_scaled, t, t + self.ts_window_length)
                current_features = self._get_features_vector_from_window(current_window)
                current_features = np.asarray(current_features)
                current_features = current_features.reshape(current_features.shape[0],
                                                            (current_features.shape[1] * current_features.shape[2]))

                x_test_arr = []
                for ts_number in range(len(current_features)):
                    x_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                        timeseries=current_features[ts_number],
                        K=self.ts_window_length - self.L + 1,
                        L=len(current_features[ts_number]))
                    x_test_arr.extend(x_test)

                if self.n_components is None:
                    _n_components(x_test, x_history, self.trajectory_window_length)

                score_list.append(
                        self._sst_svd(x_test_arr, x_history_arr, self.n_components))
        score_diff = np.diff(score_list)
        q_95 = np.quantile(score_diff, self.quantile_rate)
        filtred_score = list(map(lambda _x: 1 if _x > q_95 else 0, score_diff))

        self.n_components = None
        return filtred_score

    def features_average_(self, temp_average_features):
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

    def _score_offline_2d(self, x_scaled, dynamic_mode: bool = True):
        """Core implementation of offline score calculation. FOR 2D or more D"""
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
        filtred_score = list(map(lambda _x: 1 if _x > q_95 else 0, score_diff))
        self.n_components = None
        return filtred_score

    def score_offline(self, _x=None, dynamic_mode: bool = True):
        """Calculate anomaly score (offline).
        Parameters
        ----------
        @param dynamic_mode : bool
            (default = True).
        @param _x: 1d numpy array
            input time series data.
        Returns
        -------
        score : 1d array
            change point score.
        """
        _x = self.ts
        if not self.is_scaled:
            x_scaled = MinMaxScaler(feature_range=(1, 2)) \
                           .fit_transform(_x.reshape(-1, 1))[:, 0]
        else:
            x_scaled = _x
        return self._score_offline(x_scaled, dynamic_mode=dynamic_mode)

    def _score_offline(self, _x, dynamic_mode=True):
        """Core implementation of offline score calculation."""
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
                score_list[t - 1] = self._sst_svd(x_test, x_history, self.n_components)

        else:
            x_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                timeseries=self.ts[:start_idx], K=_K, L=_L)
            for t in range(start_idx, end_idx):
                start_idx_test, end_idx_test = t - _L, t

                x_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=self.ts[start_idx_test:end_idx_test], K=_K, L=_L)
                if self.n_components is None:
                    self._n_components(x_history, _L)
                score_list[t - 1] = self._sst_svd(x_test, x_history, self.n_components)

        score_diff = np.diff(score_list)
        q_95 = np.quantile(score_diff, self.quantile_rate)
        filtred_score = score_diff
        if self.view:
            filtred_score = list(map(lambda _x: 1 if _x > q_95 else 0, score_diff))
        return filtred_score

    def _n_components(self, x_history, _l):
        _s = self.spectrum_extractor.decompose_trajectory_matrix(x_history)[1]
        var, exp_var_by_component = self.spectrum_extractor.sv_to_explained_variance_ratio(_s, _l)
        exp_var_by_component = list(filter(lambda _s: _s > 0.05, exp_var_by_component))
        self.n_components = len(exp_var_by_component)

    def _sst_svd(self, x_test, x_history, n_components):
        """Run sst algorithm with svd."""
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
        values = feat._values
        out_values = [values[0][0], values[0][1], values[0][2], values[0][7], values[0][8], values[0][9], values[0][10]]
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
