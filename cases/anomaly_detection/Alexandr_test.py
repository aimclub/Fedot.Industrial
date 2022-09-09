#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numba import jit
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler
from core.models.spectral.SSA import Spectrum
import scipy
import matplotlib.pyplot as plt


# In[4]:


class SingularSpectrumTransformation:
    """SingularSpectrumTransformation class."""

    def __init__(self,
                 time_series,
                 quantile_rate: float = None,
                 trajectory_window_length: int = None,
                 ts_window_length: int = None,
                 lag=None,
                 is_scaled=False,
                 view: bool = True,
                 n_components: int = None):
        """Change point detection with Singular Spectrum Transformation.
        Parameters
        ----------
        quantile_rate: float
            anomaly coefficient in change point detection
        trajectory_window_length: int
            window lenght of vectors in Hankel matrix
        ts_window_length : int
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
        self.K = self.ts_window_length - self.trajectory_win_length + 1
        self.L = self.trajectory_win_length
        self.quantile_rate = quantile_rate
        self.n_components = n_components
        self.view = view
        
        
        if self.quantile_rate is None:
            self.quantile_rate = 0.95

        if self.ts_window_length is None:
            self.ts_window_length = self.trajectory_win_length

        self.lag = lag
        
        if self.lag is None:
            self.lag = self.ts_window_length / 2

        self.is_scaled = is_scaled


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
            x_scaled = MinMaxScaler(feature_range=(1, 2))                            .fit_transform(self.ts.reshape(-1, 1))[:, 0]
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

    def _score_offline(self, x, dynamic_mode = True):
        """Core implementation of offline score calculation."""
        K, L, lag = self.K, self.L, self.lag
        start_idx, end_idx = L + lag + 1, x.size + 1
        
        score = np.zeros_like(x)
        
        if dynamic_mode:
            for t in range(start_idx, end_idx):
                start_idx_hist, end_idx_hist = t - L - lag, t - lag
                start_idx_test, end_idx_test = t - L, t
                
                X_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=self.ts[start_idx_hist:end_idx_hist], K=K,L=L)

                X_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=self.ts[start_idx_test:end_idx_test],K=K,L=L)

                if self.n_components is None:
                    self._n_components(X_test, X_history, L)
                    
                score[t - 1] = self._sst_svd(X_test, X_history, self.n_components)
            
        else:
            X_history = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                timeseries=self.ts[:start_idx],K=K,L=L)

            for t in range(start_idx, end_idx):
                start_idx_test, end_idx_test = t - L, t

                X_test = self.spectrum_extractor.ts_vector_to_trajectory_matrix(
                    timeseries=self.ts[start_idx_test:end_idx_test],K=K,L=L)
                
                if self.n_components is None:
                    self._n_components(X_test, X_history, L)

#                 score[t - 1] = np.linalg.norm(distance_matrix(X_history.T, X_test.T)\
#                                                           ,self.n_components)
                score[t - 1] = self._sst_svd(X_test, X_history, self.n_components)
  
        score_diff = np.diff(score)
        q_95 = np.quantile(score_diff, self.quantile_rate)
        
        filtred_score = score_diff
        if self.view:
            filtred_score = list(map(lambda x: 1 if x > q_95 else 0, score_diff))
        return filtred_score

    def _sst_svd(self, X_test, X_history, n_components):
        """Run sst algorithm with svd."""
        U_test, s_test, _ = np.linalg.svd(X_test, full_matrices=False)
        U_history, s_hist, _ = np.linalg.svd(X_history, full_matrices=False)
        S_cov = U_test[:, :n_components].T @ U_history[:, :n_components]
        U_cov, s, _ = np.linalg.svd(S_cov, full_matrices=False)
        return 1 - s[0]
    
    def _n_components(self, X_test, X_history, L):
        U, s, V, rank = self.spectrum_extractor.decompose_trajectory_matrix(X_history)
        var, exp_var_by_component = self.spectrum_extractor.sv_to_explained_variance_ratio(s,
                                                                                           L)
#         print(exp_var_by_component)
        exp_var_by_component = list(filter(lambda s: s > 0.05, exp_var_by_component))
        self.n_components = len(exp_var_by_component)
#         print(self.n_components)
        

