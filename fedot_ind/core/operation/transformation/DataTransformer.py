from typing import Union

import numpy as np
import pandas as pd
from ripser import Rips, ripser
from scipy import sparse
from scipy.spatial.distance import pdist, squareform


class TopologicalTransformation:
    """Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
    recorded at equal intervals.

    Args:
        time_series: Time series to be decomposed.
        max_simplex_dim: Maximum dimension of the simplices to be used in the Rips filtration.
        epsilon: Maximum distance between two points to be considered connected by an edge in the Rips filtration.
        persistence_params: ...
        window_length: Length of the window to be used in the rolling window function.

    Attributes:
        epsilon_range (np.ndarray): Range of epsilon values to be used in the Rips filtration.

    """

    def __init__(self,
                 time_series: Union[pd.Series, np.ndarray, list] = None,
                 max_simplex_dim: int = None,
                 epsilon: int = None,
                 persistence_params: dict = None,
                 window_length: int = None):
        self.time_series = time_series
        self.time_series = np.array(self.time_series)
        self.time_series = self.time_series.astype(float)
        self.max_simplex_dim = max_simplex_dim
        self.epsilon_range = self.__create_epsilon_range(epsilon)
        self.persistence_params = persistence_params

        if self.persistence_params is None:
            self.persistence_params = {
                'coeff': 2,
                'do_cocycles': False,
                'verbose': False}

        self.__window_length = window_length

    @staticmethod
    def __create_epsilon_range(epsilon):
        return np.array([y * float(1 / epsilon) for y in range(epsilon)])

    @staticmethod
    def __compute_persistence_landscapes(ts):

        N = len(ts)
        I = np.arange(N - 1)
        J = np.arange(1, N)
        V = np.maximum(ts[0:-1], ts[1::])

        # Add vertex birth times along the diagonal of the distance matrix
        I = np.concatenate((I, np.arange(N)))
        J = np.concatenate((J, np.arange(N)))
        V = np.concatenate((V, ts))

        # Create the sparse distance matrix
        D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
        dgm0 = ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]
        dgm0 = dgm0[dgm0[:, 1] - dgm0[:, 0] > 1e-3, :]

        allgrid = np.unique(dgm0.flatten())
        allgrid = allgrid[allgrid < np.inf]

        xs = np.unique(dgm0[:, 0])
        ys = np.unique(dgm0[:, 1])
        ys = ys[ys < np.inf]

    @staticmethod
    def rolling_window(array: np.array, window: int) -> np.array:
        """Takes in an array and return array of rolling windows of specified length.

        Args:
            array: Array to be rolled.
            window: Length of the window.

        Returns:
            Array of rolling windows.

        """
        shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
        strides = array.strides + (array.strides[-1],)
        a_windowed = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
        return a_windowed

    def time_series_to_point_cloud(self, array: np.array = None,
                                   dimension_embed=2) -> np.array:
        """Convert a time series into a point cloud in the dimension specified by dimension_embed.

        Args:
            array: Time series to be converted.
            dimension_embed: dimension of Euclidean space in which to embed the time series into by taking
            windows of dimension_embed length, e.g. if the time series is ``[t_1,...,t_n]`` and dimension_embed
            is ``2``, then the point cloud would be ``[(t_0, t_1), (t_1, t_2),...,(t_(n-1), t_n)]``

        Returns:
            Collection of points embedded into Euclidean space of dimension = dimension_embed, constructed
            in the manner explained above.

        """

        assert len(self.time_series) >= dimension_embed, 'dimension_embed larger than length of time_series'

        if self.__window_length is None:
            self.__window_length = dimension_embed

        # compute point cloud
        if array is None:
            array = self.time_series

        point_cloud = self.rolling_window(array=array, window=dimension_embed)
        return np.array(point_cloud)

    def point_cloud_to_persistent_cohomology_ripser(self, point_cloud: np.array = None,
                                                    max_simplex_dim: int = None):

        # ensure epsilon_range is a numpy array
        epsilon_range = self.epsilon_range

        # build filtration
        self.persistence_params['maxdim'] = max_simplex_dim
        filtration = Rips(**self.persistence_params)

        if point_cloud is None:
            point_cloud = self.time_series_to_point_cloud()

        # initialize persistence diagrams
        diagrams = filtration.fit_transform(point_cloud)
        # Instantiate persistence landscape transformer
        # plot_diagrams(diagrams)

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

    def time_series_to_persistent_cohomology_ripser(self,
                                                    time_series: np.array,
                                                    max_simplex_dim: int) -> dict:
        """Wrapper function that takes in a time series and outputs the persistent homology object, along with other
        auxiliary objects.

        Args:
            time_series: Time series to be converted.
            max_simplex_dim: Maximum dimension of the simplicial complex to be constructed.

        Returns:
            Persistent homology object. Dictionary with keys in ``range(max_simplex_dim)`` and, the value ``hom[i]``
            is an array of length equal to ``len(epsilon_range)`` containing the betti numbers of the ``i-th`` homology
            groups for the Rips filtration.

        """

        homology = self.point_cloud_to_persistent_cohomology_ripser(point_cloud=time_series,
                                                                    max_simplex_dim=max_simplex_dim)
        return homology

    def time_series_rolling_betti_ripser(self, ts):

        point_cloud = self.rolling_window(array=ts, window=self.__window_length)
        homology = self.time_series_to_persistent_cohomology_ripser(point_cloud,
                                                                    max_simplex_dim=self.max_simplex_dim)
        df_features = pd.DataFrame(data=homology)
        cols = ["Betti_{}".format(i) for i in range(df_features.shape[1])]
        df_features.columns = cols
        df_features['Betti_sum'] = df_features.sum(axis=1)
        return df_features


class TSTransformer:
    def __init__(self, time_series, min_signal_ratio, max_signal_ratio, rec_metric):
        self.time_series = time_series
        self.recurrence_matrix = None
        self.threshold_baseline = [1, 5, 10, 15, 20, 25, 30]
        self.min_signal_ratio = min_signal_ratio
        self.max_signal_ratio = max_signal_ratio
        self.rec_metric = rec_metric

    def ts_to_recurrence_matrix(self,
                                eps=0.10,
                                steps=None):
        distance_matrix = pdist(metric=self.rec_metric, X=self.time_series[:, None])
        distance_matrix = np.floor(distance_matrix / eps)
        distance_matrix, steps = self.binarization(distance_matrix, threshold=steps)
        distance_matrix[distance_matrix > steps] = steps
        self.recurrence_matrix = squareform(distance_matrix)
        return self.recurrence_matrix

    def binarization(self, distance_matrix, threshold):
        best_threshold_flag = False
        signal_ratio_list = []
        if threshold is None:
            for threshold_baseline in self.threshold_baseline:
                threshold = threshold_baseline
                tmp_array = np.copy(distance_matrix)
                tmp_array[tmp_array < threshold_baseline] = 0.0
                tmp_array[tmp_array >= threshold_baseline] = 1.0
                signal_ratio = np.where(tmp_array == 0)[0].shape[0] / tmp_array.shape[0]

                if self.min_signal_ratio < signal_ratio < self.max_signal_ratio:
                    best_ratio = signal_ratio
                    distance_matrix = tmp_array
                    best_threshold_flag = True
                    if signal_ratio > best_ratio:
                        distance_matrix = tmp_array
                else:
                    signal_ratio_list.append(abs(self.max_signal_ratio - signal_ratio))

                del tmp_array

        if not best_threshold_flag:
            threshold = self.threshold_baseline[signal_ratio_list.index(min(signal_ratio_list))]
            distance_matrix[distance_matrix < threshold] = 0.0
            distance_matrix[distance_matrix >= threshold] = 1.0
        return distance_matrix, threshold

    def get_recurrence_metrics(self):
        if self.recurrence_matrix is None:
            return self.ts_to_recurrence_matrix()
