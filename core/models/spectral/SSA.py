from typing import Union
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from core.utils.Decorators import type_check_decorator

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


class Spectrum:

    def __init__(self,
                 time_series: Union[pd.DataFrame, pd.Series, np.ndarray, list],
                 window_length: int = None,
                 save_memory: bool = True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.

        Parameters
        ----------
        time_series : The original time series, in the form of a Pandas Series, NumPy array or list.
        window_length : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_memory : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """
        self.__time_series = time_series
        self.__window_length = window_length
        self.__save_memory = save_memory
        object_type = type(time_series)
        self.__set_dimensions()
        self.__check_windows_length()
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

    @window_length.setter
    def window_length(self, window_length):
        self.__window_length = window_length

    @property
    def trajectory_matrix(self):
        return self.__trajectory_matrix

    @trajectory_matrix.setter
    def trajectory_matrix(self, trajectory_matrix: np.ndarray):
        self.__trajectory_matrix = trajectory_matrix

    # @type_check_decorator(object_type=pd.Series, types_list=supported_types)
    def decompose(self, return_df=True, correlation_flag=False):
        # Embed the time series in a trajectory matrix
        Components_df = None
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

        if rank > 100000:
            combined_components = self.calc_wcorr(TS_comps, rank)
            Components_df = self.components_to_df(combined_components, len(combined_components))
        else:
            Components_df = self.components_to_df(TS_comps.T, rank)

        n_components = [x / sum(Sigma) * 100 for x in Sigma]
        n_components = list(filter(lambda s: s > 1.0, n_components))
        explained_dispersion = sum(n_components)

        if explained_dispersion > 95:
            dispersion = 0
            for index, elem in enumerate(n_components):
                if dispersion < 95:
                    dispersion += elem
                else:
                    break
        else:
            n_components = len(n_components)

        if type(n_components) is list:
            explained_dispersion = 95.0
            n_components = index

        return TS_comps, X_elem, V, Components_df, Wcorr, n_components, explained_dispersion

    def calc_wcorr(self, TS_comps, rank):
        """
        Calculates the w-correlation matrix for the time series.
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
            corellated_comp = [i for i, v in enumerate(Wcorr[i,]) if v > 0.5]

            if len(corellated_comp) < 2:
                final_component = TS_comps[corellated_comp[0],]
            else:
                final_component = np.sum(TS_comps[corellated_comp,], axis=0)

            for elem in corellated_comp:
                try:
                    components.remove(elem)
                except Exception:
                    _ = 1

            combined_components.append(final_component)

        return combined_components

    def components_to_df(self, TS_comps, rank, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, rank)
        else:
            n = rank

        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        df = pd.DataFrame(TS_comps).T
        df.columns = cols
        return df

    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.

        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]

        ts_vals = self.TS_comps[:, indices].sum(axis=1)
        return pd.Series(ts_vals)

    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d

        if self.Wcorr is None:
            self.calc_wcorr()

        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0, 1)

        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d - 1
        else:
            max_rnge = max

        plt.xlim(min - 0.5, max_rnge + 0.5)
        plt.ylim(max_rnge + 0.5, min - 0.5)
