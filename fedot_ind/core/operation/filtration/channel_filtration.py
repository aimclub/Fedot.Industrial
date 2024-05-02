import itertools

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.neighbors import NearestCentroid
from sktime.dists_kernels import (
    BasePairwiseTransformerPanel, FlatDist, ScipyDist)
from typing import Optional

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.IndustrialCachableOperation import IndustrialCachableOperationImplementation
from fedot_ind.core.repository.constanst_repository import DISTANCE_METRICS


def _detect_knee_point(values, indices):
    """Find elbow point.The elbow cut method is a method to determine a point in
    a curve where significant change can be observed, e.g., from a steep slope to almost flat curve"""
    n_points = len(values)  # number_of_channels
    # coordinate of each channel projected in chosen centroid
    all_coords = np.vstack((range(n_points), values)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coords - first_point  # line coord from first point to last
    scalar_prod = np.sum(
        vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    # "angle" between each point and line
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    # find distance from all points to line
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    knee_idx = np.argmax(dist_to_line)
    knee = values[knee_idx]
    best_dims = [idx for (elem, idx) in zip(values, indices) if elem > knee]
    if len(best_dims) == 0:
        return [knee_idx], knee_idx

    return best_dims,


class ChannelCentroidFilter(IndustrialCachableOperationImplementation):
    """ChannelCentroidFilter (CCF) transformer to select a subset of channels/variables.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1] by calculating the distance between each class centroid. The
    ECS selects the subset of channels using the elbow method, which maximizes the
    distance between the class centroids by aggregating the distance for every
    class pair across each channel.

    Note: Channels, variables, dimensions, features are used interchangeably in
    literature. E.g., channel selection = variable selection.

    Attributes:
        distance: sktime pairwise panel transform, str, or callable, optional, default=None
            if panel transform, will be used directly as the distance in the algorithm
            default None = euclidean distance on flattened series, FlatDist(ScipyDist())
            if str, will behave as FlatDist(ScipyDist(distance)) = scipy dist on flat series
            if callable, must be univariate nested_univ x nested_univ -> 2D float np.array

        channels_selected : list of integer
            List of variables/channels selected by the estimator
            integers (iloc reference), referring to variables/channels by order
        distance_frame : np.array
            distance matrix of the class centroids pair and channels.
                ``shape = [n_channels, n_class_centroids_pairs]``

    References:

        ..[1]: Bhaskar Dhariyal et al. "Fast Channel Selection for Scalable Multivariate
        Time Series Classification." AALTD, ECML-PKDD, Springer, 2021
    """

    def __init__(self, params: Optional[OperationParameters] = None):

        super().__init__(params)
        self.distance = params.get('distance', None)  # “manhattan” “chebyshev”
        self.shrink = params.get('shrink', 1e-5)
        self.centroid_metric = params.get('centroid_metric', 'euclidean')
        self.sample_metric = params.get('sample_metric', 'euclidean')
        self.sample_metric = DISTANCE_METRICS[self.sample_metric]
        self.channel_selection_strategy = params.get(
            'selection_strategy', 'sum')
        self.channels_selected = []

        if self.distance is None:
            self.distance_ = FlatDist(ScipyDist())
        elif isinstance(self.distance, str):
            self.distance_ = FlatDist(ScipyDist(metric=self.distance))
        elif isinstance(self.distance, BasePairwiseTransformerPanel):
            self.distance_ = self.distance.clone()
        else:
            self.distance_ = self.distance

    def eval_distance_from_centroid(self, centroid_frame):
        """Create distance matrix."""
        # distance from each class to each without repetitions. Number of pairs is n_cls(n_cls-1)/2
        distance_pair = list(itertools.combinations(
            range(0, centroid_frame.shape[0]), 2))
        # distance_metrics = []
        # for metric in DISTANCE_METRICS.values():
        distance_frame = pd.DataFrame()
        for class_ in distance_pair:
            class_pair = []
            # calculate the distance of centroid here
            for _, (q, t) in enumerate(zip(centroid_frame[class_[0], :],
                                           centroid_frame[class_[1], :], )):
                class_pair.append(self.sample_metric(q, t))
                dict_ = {f"Centroid_{[class_[0]]}_{[class_[1]]}": class_pair}

            distance_frame = pd.concat(
                [distance_frame, pd.DataFrame(dict_)], axis=1)
        # distance_metrics.append(distance_frame)

        return distance_frame

    def create_centroid(self, X, y):
        """Create the centroid for each class."""
        n_samples, n_channels, n_points = X.shape
        centroids = []
        for dim in range(n_channels):  # for each channel evaluate distance to class centroid
            # choose channel. Input matrix is n_samples x 1 x n_points
            train = X[:, dim, :]
            clf = NearestCentroid(metric=self.centroid_metric,
                                  shrink_threshold=self.shrink)
            clf.fit(train, y)
            # return matrix n_classes x n_points
            centroids.append(clf.centroids_)

        centroid_frame = np.stack(centroids, axis=1)

        return centroid_frame

    def _channel_sum(self):
        self.distance_frame = pd.Series(self.distance_frame.sum(axis=1))
        distance = self.distance_frame.sort_values(ascending=False).values
        indices = self.distance_frame.sort_values(ascending=False).index
        self.channels_selected = _detect_knee_point(distance, indices)[0]

    def _channel_pairwise(self, centroids_by_channel):
        self.distance_frame = self.eval_distance_from_centroid(
            centroids_by_channel)
        for pairdistance in self.distance_frame.items():
            distance = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index
            self.channels_selected.extend(
                _detect_knee_point(distance, indices)[0])
            self.channels_selected = list(set(self.channels_selected))

    def _transform(self, input_data: InputData):
        """Fit ECS to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame or np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        InputData
        """
        if input_data.features.shape[1] == 1:
            return input_data.features
        else:
            if len(self.channels_selected) == 0:
                if input_data.task.task_type.value == 'regression':
                    bins = [np.quantile(input_data.target, x)
                            for x in np.arange(0, 1, 0.2)]
                    labels = [x for x in range(len(bins) - 1)]
                    input_data.target = pd.cut(input_data.target,
                                               bins=bins,
                                               labels=labels).codes
                # step 1. create channel centroids
                centroids_by_channel = self.create_centroid(
                    input_data.features, input_data.target)
                # step 2. create distance matrix
                self.distance_frame = self.eval_distance_from_centroid(
                    centroids_by_channel)
                # step 3. choose filtration algo
                if self.channel_selection_strategy == 'sum':
                    self._channel_sum()
                elif self.channel_selection_strategy == 'pairwise':
                    self._channel_pairwise(centroids_by_channel)
            return input_data.features[:, self.channels_selected, :]
