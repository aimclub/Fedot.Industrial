import numpy as np
from numba import jit
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
from core.models.spectral.SSA import Spectrum
import scipy
import matplotlib.pyplot as plt


def generate_synt_data(data_type):
    if data_type == 'cyclic':
        x0 = np.sin(2 * np.pi * 1 * np.linspace(0, 10, 1000))
        x1 = np.sin(2 * np.pi * 2 * np.linspace(0, 10, 1000))
        x2 = np.sin(2 * np.pi * 8 * np.linspace(0, 10, 1000))
        x = np.hstack([x0, x1, x2])
        x += + np.random.rand(x.size)
    else:
        x0 = 1 * np.ones(1000) + np.random.rand(1000) * 1
        x1 = 3 * np.ones(1000) + np.random.rand(1000) * 2
        x2 = 5 * np.ones(1000) + np.random.rand(1000) * 1.5
        x = np.hstack([x0, x1, x2])
        x += np.random.rand(x.size)
    return x


def plot_data_and_score(raw_data, score):
    f, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].plot(raw_data)
    ax[0].set_title("raw data")
    ax[1].plot(score, "r")
    ax[1].set_title("score")
    f.show()


class SingularSpectrumTransformation:
    """SingularSpectrumTransformation class."""

    def __init__(self,
                 time_series,
                 trajectory_window_length: int = None,
                 ts_window_length: int = None,
                 lag=None,
                 is_scaled=False):
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
        self.trajectory_win_length = trajectory_window_length
        self.ts_window_length = ts_window_length

        if self.ts_window_length is None:
            self.ts_window_length = self.trajectory_win_length

        self.lag = lag

        if self.lag is None:
            self.lag = self.ts_window_length / 2

        self.is_scaled = is_scaled

        self.n_components = None

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

        score = self._score_offline(x_scaled, dynamic_mode=dynamic_mode)

        return score

    def _from_DM_to_GM(self, distance_matrix):
        centering_matrix = lambda n: np.identity(n) - (np.ones((n, 1)) @ np.ones((1, n))) / n
        N = distance_matrix.shape[0]
        gram_from_dist = -(centering_matrix(N) @ distance_matrix @ centering_matrix(N)) / 2

        # Compute the PC scores from Gram matrix
        w, v = np.linalg.eig(gram_from_dist)
        proj = np.diag(np.sqrt(w[:2])) @ v.T[:2]
        return w, v

    def _score_offline(self, x, dynamic_mode: bool = True):
        """Core implementation of offline score calculation."""
        start_idx = self.ts_window_length + self.lag + 1
        end_idx = x.size + 1
        norm_list_real = []
        horm_hist = None
        score = np.zeros_like(x)
        if dynamic_mode:
            for t in range(start_idx, end_idx):
                # get Hankel matrix

                start_idx_hist = t - self.ts_window_length - self.lag
                end_idx_hist = t - self.lag
                # print('HISTORY_matrix_at_idx: {} - {}'.format(start_idx_hist, end_idx_hist))
                X_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=self.ts[start_idx_hist:end_idx_hist],
                    K=self.ts_window_length - self.trajectory_win_length + 1,
                    L=self.trajectory_win_length)
                if horm_hist is None:
                    horm_hist = np.linalg.norm(X_history, 1)

                start_idx_test = t - self.ts_window_length
                end_idx_test = t
                # print('TEST_matrix_at_idx: {} - {}'.format(start_idx_test, end_idx_test))

                X_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=self.ts[start_idx_test:end_idx_test],
                    K=self.ts_window_length - self.trajectory_win_length + 1,
                    L=self.trajectory_win_length)

                if self.n_components is None:
                    U, s, V, rank = self.spectrum_extractor.decompose_trajectory_matrix(X_history)
                    var, exp_var_by_component = self.spectrum_extractor.sv_to_explained_variance_ratio(s,
                                                                                                       self.trajectory_win_length)
                    exp_var_by_component = list(filter(lambda s: s > 0.05, exp_var_by_component))
                    self.n_components = len(exp_var_by_component)
                    self.n_components = 2
                score[t - 1] = self._sst_svd(X_test, X_history, self.n_components)
        else:
            delay = 1 * self.ts_window_length
            start_idx = delay + self.lag + 1
            DM = []
            X_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                timeseries=self.ts[:delay],
                K=self.ts_window_length - self.trajectory_win_length + 1,
                L=self.trajectory_win_length)
            self.n_components = 2
            DM.append(X_history)
            for t in range(start_idx, end_idx):
                start_idx_test = t - self.ts_window_length
                end_idx_test = t
                # print('TEST_matrix_at_idx: {} - {}'.format(start_idx_test, end_idx_test))

                X_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=self.ts[start_idx_test:end_idx_test],
                    K=self.ts_window_length - self.trajectory_win_length + 1,
                    L=self.trajectory_win_length)
                DM.append(X_test)
                if horm_hist is None:
                    dist_mat = distance_matrix(X_history.T, X_history.T)
                    horm_hist = np.linalg.norm(dist_mat, 2)

                dist_mat = distance_matrix(X_history.T, X_test.T)
                norm_list_real.append(np.linalg.norm(dist_mat, 2))

        score = [horm_hist] + norm_list_real

        score_diff = np.diff(score)
        q_95 = np.quantile(score_diff, 0.95)
        filtred_score = list(map(lambda x: 1 if x > q_95 else 0, score_diff))
        self.n_components = None
        return filtred_score

    def _sst_svd(self, X_test, X_history, n_components):
        """Run sst algorithm with svd."""
        U_test, s_test, _ = np.linalg.svd(X_test, full_matrices=False)
        U_history, s_hist, _ = np.linalg.svd(X_history, full_matrices=False)
        S_cov = U_test[:, :n_components].T @ U_history[:, :n_components]
        U_cov, s, _ = np.linalg.svd(S_cov, full_matrices=False)
        return 1 - s[0]


if __name__ == '__main__':
    x = generate_synt_data(data_type='shifted')
    scorer = SingularSpectrumTransformation(time_series=x,
                                            ts_window_length=100,
                                            lag=10,
                                            trajectory_window_length=30)
    score = scorer.score_offline(dynamic_mode=False)
    plot_data_and_score(x, score)
    _ = 1

