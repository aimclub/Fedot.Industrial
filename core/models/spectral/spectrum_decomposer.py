from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from scipy.linalg import hankel
from sklearn.utils.extmath import randomized_svd

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth'] = 2

cols = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = cycler(color=cols)

supported_types = (pd.Series, np.ndarray, list)


def plot_2d(m, title=""):
    plt.imshow(m)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


class SpectrumDecomposer:
    """Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
    recorded at equal intervals.

    Args:
        time_series: The time series to decompose.
        window_length: The length of the window to use. Defaults to None.
        save_memory: Whether to save memory by not storing the elementary matrices. Defaults to True.

    """

    def __init__(self,
                 time_series: Union[pd.DataFrame, pd.Series, np.ndarray, list],
                 window_length: int = None,
                 save_memory: bool = True):
        self.__time_series = time_series
        self.__window_length = window_length
        self.__save_memory = save_memory
        self.__set_dimensions()
        self.__trajectory_matrix = self.__get_trajectory_matrix()

    def __check_windows_length(self):
        if self.__window_length is None:
            self.__window_length = len(self.__time_series) / 0.35
        if not 2 <= self.__window_length <= self.__ts_length / 2:
            raise ValueError("The window length must be in the interval [2, N/2].")

    def __set_dimensions(self):
        self.__ts_length = len(self.__time_series)
        self.__subseq_length = self.__ts_length - self.__window_length + 1

    def __get_trajectory_matrix(self):
        return np.array(
            [self.__time_series[i:self.__window_length + i] for i in range(0, self.__subseq_length)]).T

    @property
    def window_length(self):
        return self.__window_length

    @property
    def sub_seq_length(self):
        return self.__subseq_length

    @window_length.setter
    def window_length(self, window_length):
        self.__window_length = window_length

    @property
    def trajectory_matrix(self):
        return self.__trajectory_matrix

    @property
    def ts_length(self):
        return self.__ts_length

    @trajectory_matrix.setter
    def trajectory_matrix(self, trajectory_matrix: np.ndarray):
        self.__trajectory_matrix = trajectory_matrix

    def decompose(self, return_df=True, correlation_flag=False, rank_hyper=None):
        # Embed the time series in a trajectory matrix
        components_df = None
        Wcorr = None
        U, Sigma, VT = np.linalg.svd(self.__trajectory_matrix)
        rank = np.linalg.matrix_rank(self.__trajectory_matrix)
        # Decompose the trajectory matrix
        TS_comps = np.zeros((self.__ts_length, rank))

        if not self.__save_memory:
            # Construct and save all the elementary matrices
            X_elem = np.array([Sigma[i] * np.outer(U[:, i], VT[i, :]) for i in range(rank)])

            # Diagonally average the elementary matrices, store them as columns in array.
            for i in range(rank):
                X_rev = X_elem[i, ::-1]
                TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(rank):
                X_elem = Sigma[i] * np.outer(U[:, i], VT[i, :])
                X_rev = X_elem[::-1]
                TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            X_elem = "Re-run with save_mem=False to retain the elementary matrices."

            # The V array may also be very large under these circumstances, so we won't keep it.
            V = "Re-run with save_mem=False to retain the V matrix."

        rank = self.singular_value_hard_threshold(singular_values=Sigma)
        if rank_hyper is not None:
            rank = rank_hyper

        components_df = self.components_to_df(TS_comps.T, rank)

        n_components = [x / sum(Sigma) * 100 for x in Sigma]
        n_components = n_components[:rank]
        explained_dispersion = sum(n_components)
        n_components = rank

        return TS_comps, X_elem, V, components_df, Wcorr, n_components, explained_dispersion

    def calc_wcorr(self, TS_comps, rank):
        """Calculates the w-correlation matrix for the time series.

        Args:
            TS_comps (np.ndarray): The time series components.
            rank (int): The rank of the time series.

        Returns:
            ...
        """

        # Calculate the weights
        first = list(np.arange(self.__window_length) + 1)
        second = [self.__ts_length] * (self.__subseq_length - self.__window_length - 1)
        third = list(np.arange(self.__window_length) + 1)[::-1]
        w = np.array(first + second + third)

        def w_inner(F_i, F_j):
            return w.dot(F_i * F_j)

        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(TS_comps[:, i], TS_comps[:, i]) for i in range(rank)])
        F_wnorms = F_wnorms ** -0.5

        # Calculate Wcorr.
        Wcorr = np.identity(rank)
        components = [i for i in range(rank)]
        for i in components:
            for j in range(i + 1, rank):
                Wcorr[i, j] = abs(w_inner(TS_comps[:, i], TS_comps[:, j]) * F_wnorms[i] * F_wnorms[j])
                Wcorr[j, i] = Wcorr[i, j]

        combined_components = []

        for i in components:
            corellated_comp = [i for i, v in enumerate(Wcorr[:, i]) if v > 0.85]

            if len(corellated_comp) < 2:
                final_component = TS_comps[:, corellated_comp[0]]
            else:
                final_component = np.sum(TS_comps[:, corellated_comp], axis=1)

            for elem in corellated_comp:
                try:
                    components.remove(elem)
                except Exception:
                    _ = 1

            combined_components.append(final_component)

        return combined_components

    @staticmethod
    def components_to_df(TS_comps: np.ndarray, rank: int, n: int = 0) -> pd.DataFrame:
        """Converts all the time series components in a single Pandas DataFrame object.

        Args:
            TS_comps: ...
            rank: The rank of the time series.
            n: ...

        Returns:
            df: dataframe with all the time series components.
        """
        if n > 0:
            n = min(n, rank)
        else:
            n = rank

        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        df = pd.DataFrame(TS_comps).T.iloc[:,:rank]
        df.columns = cols
        return df

    def reconstruct(self, indices: Union[int, list, tuple, np.ndarray]) -> pd.Series:
        """Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.

        Args:
            indices: An integer, list of integers or slice(n,m) object, representing the elementary
            components to sum.

        Returns:
            The reconstructed time series.
        """
        if isinstance(indices, int): indices = [indices]

        ts_vals = self.TS_comps[:, indices].sum(axis=1)
        return pd.Series(ts_vals)

    def plot_wcorr(self, minimum=None, maximum=None):
        """Plots the w-correlation matrix for the decomposed time series.

        """
        if minimum is None:
            minimum = 0
        if maximum is None:
            maximum = self.d

        if self.Wcorr is None:
            self.calc_wcorr()

        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0, 1)

        # For plotting purposes:
        if maximum == self.d:
            max_range = self.d - 1
        else:
            max_range = maximum

        plt.xlim(minimum - 0.5, max_range + 0.5)
        plt.ylim(max_range + 0.5, minimum - 0.5)

    @staticmethod
    def singular_value_hard_threshold(singular_values,
                                      rank=None,
                                      threshold=2.858):
        rank = len(singular_values) if rank is None else rank

        median_sv = np.median(singular_values[:rank])
        sv_threshold = threshold * median_sv
        adjusted_rank = np.sum(singular_values >= sv_threshold)
        return adjusted_rank

    @staticmethod
    def ts_vector_to_trajectory_matrix(timeseries, L, K):
        hankelized = hankel(timeseries, np.zeros(L)).T
        hankelized = hankelized[:, :K]
        return hankelized

    def ts_matrix_to_trajectory_matrix(self, timeseries, L, K):
        P, N = timeseries.shape

        trajectory_matrix = [
            self.ts_vector_to_trajectory_matrix(timeseries[p, :], L, K)
            for p in range(P)
        ]

        trajectory_matrix = np.concatenate(trajectory_matrix, axis=1)
        return trajectory_matrix

    @staticmethod
    def decompose_trajectory_matrix(trajectory_matrix,
                                    K=10,
                                    svd_method='exact'):
        # calculate S matrix
        S = trajectory_matrix

        if svd_method == 'exact':
            try:
                U, s, V = np.linalg.svd(S)
            except Exception:
                U, s, V = randomized_svd(S, n_components=K)

        # Valid rank is only where eigenvalues > 0
        rank = np.sum(s > 0.01)

        return U, s, V, rank

    @staticmethod
    def sv_to_explained_variance_ratio(singular_values, N):
        eigenvalues = singular_values ** 2
        explained_variance = eigenvalues / (N - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance
        return explained_variance, explained_variance_ratio
