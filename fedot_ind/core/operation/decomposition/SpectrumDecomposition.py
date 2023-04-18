# -*- coding: utf-8 -*-
from typing import Union

import numpy as np
import pandas as pd

from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold

supported_types = (pd.Series, np.ndarray, list)


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
        self.__convert_ts_to_array()
        self.__window_length = window_length
        if self.__window_length is None:
            self.__window_length = round(self.__time_series.size * 0.35)
        self.__save_memory = save_memory
        self.__set_dimensions()
        self.__check_windows_length()
        self.__trajectory_matrix = self.__get_trajectory_matrix()

    def __check_windows_length(self):
        if not 2 <= self.__window_length <= self.__ts_length / 2:
            raise ValueError("The window length must be in the interval [2, N/2].")

    def __convert_ts_to_array(self):
        if type(self.__time_series) == pd.DataFrame:
            self.__time_series = self.__time_series.values
        elif type(self.__time_series) == list:
            self.__time_series = np.array(self.__time_series)
        else:
            self.__time_series = self.__time_series

    def __set_dimensions(self):
        self.__ts_length = len(self.__time_series)
        self.__subseq_length = self.__ts_length - self.__window_length + 1

    def __get_trajectory_matrix(self):
        # return np.array(
        #     [self.__time_series[i:self.__window_length + i] for i in range(0, self.__subseq_length)]).T
        hankel_transform = HankelMatrix(time_series=self.__time_series, window_size=self.window_length)
        if len(self.__time_series.shape) > 2:
            return hankel_transform.trajectory_matrix
        else:
            return hankel_transform.trajectory_matrix

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

    def decompose(self, rank_hyper: int = None):
        # Embed the time series in a trajectory matrix
        U, Sigma, VT = np.linalg.svd(self.__trajectory_matrix)

        if rank_hyper is None:
            sing_values = singular_value_hard_threshold(singular_values=Sigma)
            rank = len(sing_values)

        else:
            rank = rank_hyper

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

        return TS_comps, Sigma, rank, X_elem, V

    @staticmethod
    def components_to_df(TS_comps: np.ndarray, rank: int, n: int = 0) -> pd.DataFrame:
        """Converts all the time series components in a single Pandas DataFrame object.

        Args:
            TS_comps: The time series components.
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
        df = pd.DataFrame(TS_comps.T).T.iloc[:, :rank]
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
