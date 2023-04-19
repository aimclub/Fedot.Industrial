import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix


# def _score_offline_2d_average(self, x_scaled: list) -> list:
#     """Core implementation of offline score calculation with average features. FOR 2D or more D.
#
#     Args:
#         x_scaled: normalized time series if is_scaled False
#         self.dynamic_mode: type of model to check differences in the sequence
#
#     Returns:
#         filtered_score: represented a list of values with 0 and 1 where 1 is an anomaly if view is True
#
#     """
#     score_list = []
#     if not self.dynamic_mode:
#         step = self.lag
#         start_idx = step
#         end_idx = len(x_scaled[0]) - step
#         horm_hist = None
#         average_features = self.average_features_2d(end_idx, start_idx, x_scaled)
#
#         for current_index in range(start_idx, end_idx, step):
#             current_features = self.current_features_2d(current_index, step, x_scaled)
#             current_features = np.reshape(current_features, (len(x_scaled), 7))
#             if horm_hist is None:
#                 horm_hist = np.linalg.norm(distance_matrix(average_features.T, current_features.T), 2)
#             score_list.append(np.linalg.norm(distance_matrix(average_features.T, current_features.T), 2))
#     else:
#         start_idx = self.ts_window_length + self.lag
#         end_idx = len(x_scaled[0]) - self.ts_window_length - 1
#         horm_hist = None
#         x_history_arr = []
#         average_features = self.average_features_2d(end_idx, start_idx, x_scaled)
#         x_history = None
#         for ts_number in range(len(average_features)):
#             x_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
#                 timeseries=average_features[ts_number],
#                 K=self.ts_window_length - self.L + 1,
#                 L=len(average_features[ts_number]))
#             x_history_arr.extend(x_history)
#
#         for t in range(start_idx, end_idx, self.lag):  # get Hankel matrix
#             if horm_hist is None:
#                 horm_hist = np.linalg.norm(x_history, 1)
#
#             current_features = self.current_features_2d(t, self.ts_window_length, x_scaled)
#             current_features = current_features.reshape(current_features.shape[0],
#                                                         (current_features.shape[1] * current_features.shape[2]))
#
#             x_test_arr = []
#             x_test = None
#             for ts_number in range(len(current_features)):
#                 x_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
#                     timeseries=current_features[ts_number],
#                     K=self.ts_window_length - self.L + 1,
#                     L=len(current_features[ts_number]))
#                 x_test_arr.extend(x_test)
#
#             if self.n_components is None:
#                 self._n_components(x_test, x_history)
#
#             score_list.append(
#                 self._sst_svd(x_test_arr, x_history_arr))
#     score_diff = np.diff(score_list)
#     q_95 = np.quantile(score_diff, self.quantile_rate)
#     filtered_score = score_diff
#     self.n_components = None
#     if self.view:
#         filtered_score = list(map(lambda _x: 1 if _x > q_95 else 0, score_diff))
#     return filtered_score
#
#
# def _score_offline_2d(self, x_scaled: list = None) -> list:
#     """Core implementation of offline score calculation. FOR 2D or more D.
#
#     Args:
#         x_scaled: normalized time series if is_scaled False
#         self.dynamic_mode: type of model to check differences in the sequence
#
#     Returns:
#         filtered_score: represented a list of values with 0 and 1 where 1 is an anomaly if view is True
#
#     """
#     norm_list_real = []
#     horm_hist = None
#     if self.dynamic_mode:
#         step = 1 * self.ts_window_length
#         start_idx = step
#         end_idx = len(x_scaled[0]) - step
#
#         current_index = start_idx
#         first_window = self._get_window_from_ts_complex(x_scaled, current_index, current_index + step)
#         first_features = self._get_features_vector_from_window(first_window)
#         first_features = np.asarray(first_features)
#         first_features = np.reshape(first_features, (len(x_scaled), 7))
#
#         for current_index in range(start_idx, end_idx, step):
#             current_window = self._get_window_from_ts_complex(x_scaled, current_index, current_index + step)
#             current_features = self._get_features_vector_from_window(current_window)
#             current_features = np.asarray(current_features)
#             current_features = np.reshape(current_features, (len(x_scaled), 7))
#             if horm_hist is None:
#                 horm_hist = np.linalg.norm(distance_matrix(first_features.T, current_features.T), 2)
#             norm_list_real.append(np.linalg.norm(distance_matrix(first_features.T, current_features.T), 2))
#     else:
#         raise ValueError("Function dose not work when dynamic == False (FOR 2D or more D)")
#
#     score_list = [horm_hist] + norm_list_real
#     score_diff = np.diff(score_list)
#     return score_diff

class SingularSpectrumTransformation:
    """SingularSpectrumTransformation class.

    Change point detection with Singular Spectrum Transformation.

    Note:
        In case of 1D time series to find appropriate hyperparameters value
         for ts_window_length and trajectory_window_length
         are recommended to use WSSAlgorithms.py, WindowSizeSelection class.

    Parameters:
        model_hyperparams:
                        n_components: Number of principal components to keep from
                            functional principal component analysis. Defaults to 3.
                        window_length: Regularization object to be applied.
                        trajectory_window_length: .
    Attributes:

    """

    def __init__(self, model_hyperparams: dict = None):

        if model_hyperparams is not None:
            self.n_components = model_hyperparams['n_components']
            self.window_length = model_hyperparams['window_length']
            self.trajectory_window_length = model_hyperparams['trajectory_window_length']
            self.dynamic_mode = model_hyperparams['dynamic_mode']
            self.lag = model_hyperparams['delay_lag']

        if self.lag is None:
            self.lag = np.round(self.window_length / 2)
        if self.n_components is None:
            self.n_components = 2

    def _scale_ts(self, time_series: np.ndarray):
        time_series_scaled = MinMaxScaler(feature_range=(1, 2)) \
                                 .fit_transform(time_series.reshape(-1, 1))[:, 0]
        return time_series_scaled

    def fit(self, train_features: np.ndarray) -> list:
        if len(train_features) > 1:
            list_of_history, list_of_current = [], []
            for time_series in train_features:
                history_state, current_state = self._fit(train_features=time_series)
                list_of_history.append(history_state)
                list_of_current.append(current_state)
            return list_of_history, list_of_current
        else:
            return self._fit(train_features=train_features)

    def _fit(self, train_features: np.ndarray) -> list:
        """Core implementation of offline score calculation.

        Args:
            self.dynamic_mode: mode for SST metrics calculation.
            train_features: input 1D time series data.

        Returns
            score: 1d array change point score with 1 and 0 if view True.

        """
        train_features = np.array(train_features)
        start_idx, end_idx = self.window_length + self.lag + 1, train_features.size + 1
        test_features = []
        if self.dynamic_mode:
            self.model = []
            for t in range(start_idx, end_idx):
                start_idx_hist, end_idx_hist = t - self.window_length - self.lag, t - self.lag
                start_idx_test, end_idx_test = t - self.window_length, t
                x_history = HankelMatrix(time_series=train_features[start_idx_hist:end_idx_hist],
                                         window_length=self.trajectory_window_length)
                x_test = HankelMatrix(time_series=train_features[start_idx_test:end_idx_test],
                                      window_length=self.trajectory_window_length)
                self.model.append(x_history)
                test_features.append(x_test)

        else:
            self.model = HankelMatrix(time_series=train_features[:start_idx],
                                      window_length=self.trajectory_window_length)
            for t in range(start_idx, end_idx):
                start_idx_test, end_idx_test = t - self.lag, t
                x_test = HankelMatrix(time_series=train_features[start_idx_test:end_idx_test],
                                      window_length=self.trajectory_window_length)
                test_features.append(x_test)

        return self.model, test_features

    def predict(self, test_features: list, target: list) -> list:
        if len(test_features) > 1:
            residual_list = []
            for history_data, target_data in zip(test_features, target):
                residual = self._predict(history_data, target_data)
                residual_list.append(residual)
            return residual_list
        else:
            return self._predict(test_features=test_features,target=target)

    def _predict(self, test_features: HankelMatrix, target: HankelMatrix) -> list:
        """Core implementation of offline score calculation.

        Args:
            self.dynamic_mode: mode for SST metrics calculation.
            _x: input 1D time series data.

        Returns
            score: 1d array change point score with 1 and 0 if view True.
            :param target:
            :param test_features:

        """

        if self.dynamic_mode:
            dynamic_sst = lambda trajectory_data_tuple: self._sst_svd(trajectory_data_tuple[0].trajectory_matrix,
                                                                      trajectory_data_tuple[1].trajectory_matrix)
            trajectory_data_tuple = list(zip(test_features, target))
            score_list = list(map(dynamic_sst, trajectory_data_tuple))
        else:
            static_sst = lambda x_history: self._sst_svd(self.model.trajectory_matrix, x_history.trajectory_matrix)
            score_list = list(map(static_sst, test_features))
        residual = np.diff(score_list)
        return residual

    def _sst_svd(self, x_test: list = None, x_history: list = None) -> float:
        """Singular value decomposition to count distance score between matrix

        Args:
            x_test: current matrix of features
            x_history: historical matrix of features

        Returns
            1 - s[0]: distance score between two matrix

        """
        u_test, s_test, _ = np.linalg.svd(x_test, full_matrices=False)
        u_history, s_hist, _ = np.linalg.svd(x_history, full_matrices=False)
        s_cov = u_test[:, :self.n_components].T @ u_history[:, :self.n_components]
        u_cov, s, _ = np.linalg.svd(s_cov, full_matrices=False)
        return 1 - s[0]
