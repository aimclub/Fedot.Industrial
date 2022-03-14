from typing import Union

import numpy as np
import pandas as pd
from ripser import Rips
import matplotlib.pyplot as plt
import random
import itertools
from functools import reduce


# rips = Rips()
# data = np.random.random((100, 2))
# diagrams = rips.fit_transform(data)
# rips.plot(diagrams)


class Topological:
    def __init__(self,
                 time_series: Union[pd.DataFrame, pd.Series, np.ndarray, list],
                 max_simplex_dim: int = None,
                 epsilon: int = None,
                 persistance_params: dict = None,
                 window_length: int = None):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.

        Parameters
        ----------
        time_series : The original time series, in the form of a Pandas Series, NumPy array or list.
        window_length : integer that will be the length of the window.
        """
        self.__time_series = time_series

        # time series to float type
        self.__time_series = np.array(self.__time_series)
        self.__time_series = self.__time_series.astype(float)

        self.max_simplex_dim = max_simplex_dim

        self.epsilon_range = self.__create_epsilon_range(epsilon)

        self.persistance_params = persistance_params
        if self.persistance_params is None:
            self.persistance_params = {'maxdim': 2,
                                       'thresh': -1,
                                       'coeff': 2,
                                       'do_cocycles': False,
                                       'verbose': False}

        self.__window_length = window_length

    def __create_epsilon_range(self, epsilon):
        return np.array([y * float(1 / epsilon) for y in range(epsilon)])

    def rolling_window(self, array):
        """
        Take in an array and return array of rolling windows of specified length

        Parameters:
        - array: numpy array that will be windowed
        - window: integer that will be the length of the window

        Returns:
        - a_windowed: array where each entry is an array of length window
        """
        shape = array.shape[:-1] + (array.shape[-1] - self.__window_length + 1, self.__window_length)
        strides = array.strides + (array.strides[-1],)
        a_windowed = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
        return a_windowed

    def time_series_to_point_cloud(self, dimension_embed=2):
        """
        Convert a time series into a point cloud in the dimension specified by dimension_embed

        Parameters:
        - dimension_embed: dimension of Euclidean space in which to embed the time series into by taking windows of
        dimension_embed length, e.g. if the time series is [t_1,...,t_n] and dimension_embed is 2,
        then the point cloud would be [(t_0, t_1), (t_1, t_2),...,(t_(n-1), t_n)]

        Returns:
        - point_cloud: collection of points embedded into Euclidean space of dimension = dimension_embed, constructed
        in the manner explained above
        """

        assert len(self.__time_series) >= dimension_embed, 'dimension_embed larger than length of time_series'

        if self.__window_length is None:
            self.__window_length = dimension_embed

        # compute point cloud
        point_cloud = self.rolling_window(array=self.__time_series)

        return np.array(point_cloud)

    def point_cloud_to_persistent_cohomology_ripser(self,
                                                    point_cloud: np.array = None,
                                                    max_simplex_dim: int = None):

        # ensure epsilon_range is a numpy array
        epsilon_range = self.epsilon_range

        # build filtration
        filtration = Rips(**self.persistance_params)

        if point_cloud is None:
            point_cloud = self.time_series_to_point_cloud()

        # initialize persistence diagrams
        diagrams = filtration.fit_transform(point_cloud)

        # normalize epsilon distance in diagrams so max is 1
        diagrams = [np.array([dg for dg in diag if np.isfinite(dg).all()]) for diag in diagrams]
        diagrams = diagrams / max(
            [np.array([dg for dg in diag if np.isfinite(dg).all()]).max() for diag in diagrams if diag.shape[0] > 0])

        ep_ran_len = len(epsilon_range)

        homology = {dimension: np.zeros(ep_ran_len).tolist() for dimension in range(max_simplex_dim + 1)}

        for dimension, diagram in enumerate(diagrams):
            if dimension <= max_simplex_dim and len(diagram) > 0:
                homology[dimension] = np.array(
                    [np.array(((epsilon_range >= point[0]) & (epsilon_range <= point[1])).astype(int))
                     for point in diagram
                     ]).sum(axis=0).tolist()

        return homology

    def time_series_to_persistent_cohomology_ripser(self):
        """
        Wrapper function that takes in a time series and outputs
        the persistent homology object, along with other
        auxiliary objects.

        Parameters:
        - time_series: Numpy array of time series values
        - epsilon_range: Numpy array of epsilon values between 0 and 1 at which to extract betti numbers.
        - max_simplex_dim: Integer denoting the maximum dimension of simplexes to create in filtration

        Returns:
        - homology: dictionary with keys in range(max_simplex_dim) and, the value hom[i] is an array of length
        equal to len(epsilon_range) containing the betti numbers of the i-th homology groups for the Rips filtration
        """

        # create point cloud from time series
        point_cloud = self.time_series_to_point_cloud(dimension_embed=self.max_simplex_dim)

        homology = self.point_cloud_to_persistent_cohomology_ripser(point_cloud=point_cloud,
                                                                    max_simplex_dim=self.max_simplex_dim - 1)
        return homology

    def time_series_rolling_betti_ripser(self, df, channels, max_simplex_dim, window):
        betti_channels = {}
        for channel in channels:
            betti_channels[channel] = [
                self.time_series_to_persistent_cohomology_ripser()
                for wdw in self.rolling_window(df[channel].values)
            ]

            betti_channels[channel] = pd.concat(
                objs=[
                    pd.Series(df[channel].index[window - 1:], index=df[channel].index[window - 1:], name='index'),
                    df[channel][window - 1:],
                    pd.DataFrame(
                        data=betti_channels[channel],
                        index=df[channel].index[window - 1:]
                    ).rename(columns={n: '{CHANNEL}_betti_'.format(CHANNEL=channel) + str(n) for n in
                                      range(max_simplex_dim + 1)})
                ],
                axis=1
            )

        return reduce(lambda left, right: pd.merge(left, right, on='index'),
                      [betti_channels[channel] for channel in channels])
