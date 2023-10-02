# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
import pandas as pd
import tensorly as tl
from pymonad.list import ListMonad
from tensorly.decomposition import parafac

from fedot_ind.core.operation.decomposition.matrix_decomposition.power_iteration_decomposition import RSVDDecomposition
from fedot_ind.core.operation.transformation.regularization.spectrum import reconstruct_basis, \
    singular_value_hard_threshold

supported_types = (pd.Series, np.ndarray, list)


class SpectrumDecomposer:
    """Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
    recorded at equal intervals.

    Args:
        data: hankel matrix of time series
        ts_length: The length of the time series
        threshold: number of components.

    """

    def __init__(self, data, ts_length, threshold: Optional[int] = None):
        if type(data) == list:
            self.rank = round(data[0].shape[0] / 10)
            self.beta = data[0].shape[0] / data[0].shape[1]
        self.ts_length = ts_length
        self.svd_estimator = RSVDDecomposition()
        self.low_rank_approximation = True
        self.thr = threshold

    def svd(self, x):
        return ListMonad(self.svd_estimator.rsvd(tensor=x,
                                                 approximation=self.low_rank_approximation,
                                                 regularized_rank=self.thr))

    def threshold(self, x):
        return ListMonad([x[0],
                          x[1][:self.thr],
                          x[2]])

    def multi_threshold(self, x):
        return ListMonad([x[1],
                          list(map(lambda x: singular_value_hard_threshold(
                              singular_values=x,
                              beta=self.beta,
                              threshold=None), x[0])),
                          x[2].T]) if self.thr is None else ListMonad([x[1][
                                                                       :,
                                                                       :self.thr],
                                                                       x[0][
                                                                       :,
                                                                       :self.thr],
                                                                       x[2][
                                                                       :,
                                                                       :self.thr].T])

    def data_driven_basis(self, x):
        return ListMonad(reconstruct_basis(x[0],
                                           x[1],
                                           x[2],
                                           ts_length=self.ts_length))

    def tensor_decomposition(self, x):
        return ListMonad(parafac(tl.tensor(x), rank=self.rank).factors)
