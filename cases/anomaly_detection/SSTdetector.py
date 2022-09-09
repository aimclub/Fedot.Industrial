import numpy as np
from numba import jit
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
from core.models.spectral.SSA import Spectrum
import pandas as pd
from core.models.statistical.Stat_features import AggregationFeatures


# from .util.linear_algebra import power_method, lanczos, eig_tridiag


class SingularSpectrumTransformation:
    """SingularSpectrumTransformation class."""

    # def __init__(self,
    #              time_series,
    #              trajectory_window_length: int = None,
    #              ts_window_length: int = None,
    #              lag=None,
    #              is_scaled=False,
    #              ):

    def __init__(self,
                 time_series,
                 quantile_rate: float = None,
                 trajectory_window_length: int = None,
                 ts_window_length: int = None,
                 lag: int = None,
                 is_scaled=False,
                 view: bool = True,
                 n_components: int = None):
        """Change point detection with Singular Spectrum Transformation.
        Parameters
        ----------
        win_length : int
            window length of Hankel matrix.
        lag : int
            interval between history Hankel matrix and test Hankel matrix.
        is_scaled : bool
            if false, min-max scaling will be applied(recommended).
        """
        self.spectrum_extractor = Spectrum(time_series=time_series, window_length=trajectory_window_length)
        self.ts = time_series
        self.trajectory_window_length = trajectory_window_length   # equal self.L
        # self.trajectory_win_length = trajectory_window_length  # equal self.L
        self.L = trajectory_window_length    # equal self.trajectory_window_length
        self.K = ts_window_length - self.L + 1
        self.quantile_rate = quantile_rate
        self.n_components = n_components
        self.view = view
        self.aggregator = AggregationFeatures()

        self.ts_window_length = ts_window_length

        if self.ts_window_length is None:
            self.ts_window_length = self.L

        self.lag = lag

        if self.lag is None:
            self.lag = self.ts_window_length / 2

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
        # score = self._score_offline_2d(dynamic_mode=dynamic_mode)
        return self._score_offline_2d(dynamic_mode=dynamic_mode)

    def score_offline_2d_average(self, dynamic_mode: bool = True):
        if not self.is_scaled:
            x_scaled = MinMaxScaler(feature_range=(1, 2)) \
                           .fit_transform(self.ts.reshape(-1, 1))[:, 0]
        else:
            x_scaled = self.ts
        score = self._score_offline_2d_average(dynamic_mode=dynamic_mode)

        return score

    def _get_window_from_ts_complex(self, ts_complex, start: int, end: int) -> list:
        window: list = []
        if start < 0 or start >= len(ts_complex[0]): raise ValueError\
            ("Start value is less than zero or more then lenght of time series!")
        if end < 0 or end >= len(ts_complex[0]): raise ValueError\
            ("End value is less than zero or more then lenght of time series!")
        if end < start : raise ValueError("Start > End!")
        for _ in ts_complex:
            window.append([])
        for i in range(start, end):
            for j in range(len(ts_complex)):
                window[j].append(ts_complex[j][i])
        return window
    
    def _score_offline_2d_average(self, dynamic_mode: bool = True):
        """Core implementation of offline score calculation. FOR 2D or more D"""
        score = np.zeros_like(self.ts[0])
        if not dynamic_mode:
            step = 1 * self.ts_window_length
            start_idx = step
            step = self.lag
            end_idx = len(self.ts[0]) - step
            norm_list_real = []
            horm_hist = None
            current_index = 0
            temp_average_features = [[]] * len(self.ts)
            for current_index in range(start_idx, end_idx, step):
                current_window = self._get_window_from_ts_complex(self.ts, current_index, current_index+step)
                current_features = self._get_features_vector_from_window(current_window)
                current_features = np.asarray(current_features)
                for i in range(len(self.ts)):
                    temp_average_features[i].append(current_features[i])
            temp_average_features = np.array(temp_average_features)
            average_features = []
            for i in range(len(temp_average_features)):
                average_features.append(np.average(temp_average_features[i], axis=0))
            
            average_features = np.array(average_features)
            average_features = average_features.reshape\
            (
                average_features.shape[0], 
                (average_features.shape[1]*average_features.shape[2])
            )
            #first_window = self._get_window_from_ts_complex(self.ts, current_index, current_index+step)
            #first_features = self._get_features_vector_from_window(first_window)
            #first_features = np.asarray(first_features)
            #first_features = np.reshape(first_features, (len(self.ts),7))
            

            from scipy.spatial import distance
            for current_index in range(start_idx, end_idx, step):
                # print('TEST_matrix_at_idx: {} - {}'.format(start_idx_test, end_idx_test))
                current_window = self._get_window_from_ts_complex(self.ts, current_index, current_index+step)
                current_features = self._get_features_vector_from_window(current_window)
                current_features = np.asarray(current_features)
                current_features = np.reshape(current_features, (len(self.ts),7))
                if horm_hist is None:
                    # horm_hist = np.linalg.norm(X_history, 'fro')
                    horm_hist = np.linalg.norm(distance_matrix(average_features.T, current_features.T), 2)

                # norm_list_real.append(np.linalg.norm(X_history-X_test, 'fro'))
                norm_list_real.append(np.linalg.norm(distance_matrix(average_features.T, current_features.T), 2))
                #norm_list_real.append(np.linalg.norm(abs(X_history-X_test), 1))
                #score[t - 1] = self._sst_svd(X_test, X_history, self.n_components)
            # score = [horm_hist - x for x in norm_list_real]
            # score = norm_list_real
        else:
            score = []
            start_idx = self.ts_window_length + self.lag
            end_idx = len(self.ts[0]) - self.ts_window_length - 1
            norm_list_real = []
            horm_hist = None
            x_history_arr = []
            temp_average_features = [[]] * len(self.ts)
            for current_index in range(start_idx, end_idx, self.lag):
                current_window = self._get_window_from_ts_complex(self.ts, current_index, current_index+self.lag)
                current_features = self._get_features_vector_from_window(current_window)
                current_features = np.asarray(current_features)
                for i in range(len(self.ts)):
                    temp_average_features[i].append(current_features[i])
            temp_average_features = np.array(temp_average_features)
            average_features = []
            for i in range(len(temp_average_features)):
                average_features.append(np.average(temp_average_features[i], axis=0))
            
            average_features = np.array(average_features)
            average_features = average_features.reshape\
            (
                average_features.shape[0], 
                (average_features.shape[1]*average_features.shape[2])
            )
            for ts_number in range(len(average_features)):
                X_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                                                                timeseries=average_features[ts_number],
                                                                K=self.ts_window_length - self.L + 1,
                                                                L=len(average_features[ts_number]))
                x_history_arr.extend(X_history)
            for t in range(start_idx, end_idx, self.lag):
                # get Hankel matrix
                start_idx_hist = t - self.ts_window_length - self.lag
                end_idx_hist = t - self.lag
                # print('HISTORY_matrix_at_idx: {} - {}'.format(start_idx_hist, end_idx_hist))
                
                if horm_hist is None:
                    horm_hist = np.linalg.norm(X_history, 1)

                start_idx_test = t - self.ts_window_length
                end_idx_test = t
                # print('TEST_matrix_at_idx: {} - {}'.format(start_idx_test, end_idx_test))
                current_window = self._get_window_from_ts_complex(self.ts, t, t+self.ts_window_length)
                current_features = self._get_features_vector_from_window(current_window)
                current_features = np.asarray(current_features)
                current_features = current_features.reshape\
                (
                    current_features.shape[0], 
                    (current_features.shape[1]*current_features.shape[2])
                )
                x_test_arr = []
                for ts_number in range(len(current_features)):
                    X_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                                                                  timeseries=current_features[ts_number],
                                                                  K=self.ts_window_length - self.L + 1,
                                                                  L=len(current_features[ts_number]))
                    x_test_arr.extend(X_test)

                if self.n_components is None:
                    _n_components(X_test, X_history, self.trajectory_window_length)

                score.append\
                (
                    self._sst_svd(x_test_arr, x_history_arr, self.n_components)
                )
                #norm_list_real.append(horm_hist)
        #score = [horm_hist] + norm_list_real
        score_diff = np.diff(score)
        q_95 = np.quantile(score_diff, 0.95)
        filtred_score = list(map(lambda x: 1 if x > q_95 else 0, score_diff))
        #dst_mtr = distance_matrix(score, score)
        self.n_components = None
        return filtred_score


    def _score_offline_2d(self, dynamic_mode: bool = True):
        """Core implementation of offline score calculation. FOR 2D or more D"""
        step = 1 * self.ts_window_length
        start_idx = step
        end_idx = len(self.ts[0]) - step
        norm_list_real = []
        horm_hist = None
        
        current_index = start_idx
        first_window = self._get_window_from_ts_complex(self.ts, current_index, current_index+step)
        first_features = self._get_features_vector_from_window(first_window)
        first_features = np.asarray(first_features)
        first_features = np.reshape(first_features, (len(self.ts),7))
        
        #q = np.quantile(first_features, 0.8)
        #m = np.median(first_features)
        #p = np.ptp(first_features)
        #n = np.nanmean(first_features)
        #first_matrix_metrics = np.asarray([q, m, p, n])
        from scipy.spatial import distance
        for current_index in range(start_idx, end_idx, step):
            # print('TEST_matrix_at_idx: {} - {}'.format(start_idx_test, end_idx_test))
            current_window = self._get_window_from_ts_complex(self.ts, current_index, current_index+step)
            current_features = self._get_features_vector_from_window(current_window)
            current_features = np.asarray(current_features)
            #q = np.quantile(current_features, 0.8)
            #m = np.median(current_features)
            #p = np.ptp(current_features)
            #n = np.nanmean(current_features)
            #current_matrix_metrics = np.asarray([q, m, p, n])
            current_features = np.reshape(current_features, (len(self.ts),7))
            if horm_hist is None:
                # horm_hist = np.linalg.norm(X_history, 'fro')
                horm_hist = np.linalg.norm(distance_matrix(first_features.T, current_features.T), 2)

            # norm_list_real.append(np.linalg.norm(X_history-X_test, 'fro'))
            norm_list_real.append(np.linalg.norm(distance_matrix(first_features.T, current_features.T), 2))
            #norm_list_real.append(np.linalg.norm(abs(X_history-X_test), 1))
            #score[t - 1] = self._sst_svd(X_test, X_history, self.n_components)
        # score = [horm_hist - x for x in norm_list_real]
        # score = norm_list_real

        score = [horm_hist] + norm_list_real
        score_diff = np.diff(score)
        q_95 = np.quantile(score_diff, 0.95)
        filtred_score = list(map(lambda x: 1 if x > q_95 else 0, score_diff))
        #dst_mtr = distance_matrix(score, score)
        self.n_components = None
        return filtred_score

    def score_offline(self, dynamic_mode: bool = True):
        """Calculate anomaly score (offline).
        Parameters
        ----------
        x : 1d numpy array
            input time series data.
        Returns
        -------
        score : 1d array
            change point score.
        """

        # all values should be positive for numerical stabilization
        if not self.is_scaled:
            x_scaled = MinMaxScaler(feature_range=(1, 2)) \
                           .fit_transform(self.ts.reshape(-1, 1))[:, 0]
        else:
            x_scaled = self.ts

        # score = self._score_offline(x_scaled, dynamic_mode=dynamic_mode)

        return self._score_offline(x_scaled, dynamic_mode=dynamic_mode)

    def _score_offline(self, x, dynamic_mode=True):
        """Core implementation of offline score calculation."""
        K, L, lag = self.K, self.L, self.lag
        start_idx, end_idx = L + lag + 1, x.size + 1
        score = np.zeros_like(x)

        if dynamic_mode:
            for t in range(start_idx, end_idx):
                start_idx_hist, end_idx_hist = t - L - lag, t - lag
                start_idx_test, end_idx_test = t - L, t

                X_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                          timeseries=self.ts[start_idx_hist:end_idx_hist], K=K, L=L)
                X_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                       timeseries=self.ts[start_idx_test:end_idx_test], K=K, L=L)
                if self.n_components is None:
                    self._n_components(X_test, X_history, L)
                score[t - 1] = self._sst_svd(X_test, X_history, self.n_components)

        else:
            X_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                                       timeseries=self.ts[:start_idx], K=K, L=L)
            for t in range(start_idx, end_idx):
                start_idx_test, end_idx_test = t - L, t

                X_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                       timeseries=self.ts[start_idx_test:end_idx_test], K=K, L=L)
                if self.n_components is None:
                    self._n_components(X_test, X_history, L)
                score[t - 1] = self._sst_svd(X_test, X_history, self.n_components)

        score_diff = np.diff(score)
        q_95 = np.quantile(score_diff, self.quantile_rate)
        filtred_score = score_diff
        if self.view:
            filtred_score = list(map(lambda x: 1 if x > q_95 else 0, score_diff))
        return filtred_score

    def _n_components(self, X_test, X_history, L):
        U, s, V, rank = self.spectrum_extractor.decompose_trajectory_matrix(X_history)
        var, exp_var_by_component = self.spectrum_extractor.sv_to_explained_variance_ratio(s, L)
        exp_var_by_component = list(filter(lambda s: s > 0.05, exp_var_by_component))
        self.n_components = len(exp_var_by_component)

    def _sst_svd(self, X_test, X_history, n_components):
        """Run sst algorithm with svd."""
        U_test, s_test, _ = np.linalg.svd(X_test, full_matrices=False)
        U_history, s_hist, _ = np.linalg.svd(X_history, full_matrices=False)
        S_cov = U_test[:, :n_components].T @ U_history[:, :n_components]
        U_cov, s, _ = np.linalg.svd(S_cov, full_matrices=False)
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
        #out_values.append(values[0][3])
        #out_values.append(values[0][4])
        #out_values.append(values[0][5])
        #out_values.append(values[0][6])
        out_values.append(values[0][7])
        out_values.append(values[0][8])
        out_values.append(values[0][9])
        out_values.append(values[0][10])
        #out_values.append(values[0][11])  
        out_values = np.array(out_values) 
        reshaped_values = [out_values] 
        return reshaped_values

if __name__ == '__main__':
    # x0 = np.sin(2 * np.pi * 1 * np.linspace(0, 10, 1000))
    # x1 = np.sin(2 * np.pi * 2 * np.linspace(0, 10, 1000))
    # x2 = np.sin(2 * np.pi * 8 * np.linspace(0, 10, 1000))
    # x = np.hstack([x0, x1, x2])
    # x += + np.random.rand(x.size)

    import matplotlib.pyplot as plt

    x0 = 1 * np.ones(1000) + np.random.rand(1000) * 1
    x1 = 3 * np.ones(1000) + np.random.rand(1000) * 2
    x2 = 5 * np.ones(1000) + np.random.rand(1000) * 1.5
    x = np.hstack([x0, x1, x2])
    x += np.random.rand(x.size)


    def plot_data_and_score(raw_data, score):
        f, ax = plt.subplots(2, 1, figsize=(20, 10))
        ax[0].plot(raw_data)
        ax[0].set_title("raw data")
        ax[1].plot(score, "r")
        ax[1].set_title("score")
        f.show()


    scorer = SingularSpectrumTransformation(time_series=x,
                                            ts_window_length=100,
                                            lag=10,
                                            trajectory_window_length=30)
    score = scorer.score_offline(dynamic_mode=False)
    plot_data_and_score(x, score)
    _ = 1
