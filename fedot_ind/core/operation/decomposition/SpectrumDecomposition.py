# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorly as tl
from pymonad.list import ListMonad
from tensorly.decomposition import parafac

from fedot_ind.core.operation.transformation.data.eigen import combine_eigenvectors
from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold, \
    reconstruct_basis

supported_types = (pd.Series, np.ndarray, list)


class SpectrumDecomposer:
    """Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
    recorded at equal intervals.

    Args:
        time_series: The time series to decompose.
        window_length: The length of the window to use. Defaults to None.
        save_memory: Whether to save memory by not storing the elementary matrices. Defaults to True.

    """

    def __init__(self, data, n_components, ts_length):
        if type(data) == list:
            rank = round(data[0].shape[0] / 10)
            beta = data[0].shape[0] / data[0].shape[1]

        window_size = data.shape[0]

        self.svd = lambda x: ListMonad(np.linalg.svd(x))
        self.threshold = lambda Monoid: ListMonad([Monoid[0],
                                                   singular_value_hard_threshold(singular_values=Monoid[1],
                                                                                 beta=data.shape[0] / data.shape[1],
                                                                                 threshold=None),
                                                   Monoid[2]]) if n_components is None else ListMonad(
            [Monoid[0][:, :n_components],
             Monoid[1][
             :n_components],
             Monoid[2][:n_components, :]])
        self.data_driven_basis = lambda Monoid: ListMonad(reconstruct_basis(Monoid[0],
                                                                            Monoid[1],
                                                                            Monoid[2],
                                                                            ts_length=ts_length))

        self.tensor_decomposition = lambda x: ListMonad(parafac(tl.tensor(x), rank=rank).factors)
        multi_threshold = lambda x: singular_value_hard_threshold(singular_values=x,
                                                                  beta=beta,
                                                                  threshold=None)

        self.multi_threshold = lambda Monoid: ListMonad([Monoid[1],
                                                         list(map(multi_threshold, Monoid[0])),
                                                         Monoid[2].T]) if n_components is None else ListMonad(
            [Monoid[1][
             :,
             :n_components],
             Monoid[0][
             :,
             :n_components],
             Monoid[2][
             :,
             :n_components].T])

        self.combine_components = lambda Monoid: ListMonad(
            combine_eigenvectors(Monoid, window_length=window_size, correlation_level=0.6))
