# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
import pandas as pd
import tensorly as tl
from pymonad.list import ListMonad
from tensorly.decomposition import parafac

from fedot_ind.core.operation.decomposition.matrix_decomposition.fast_svd import RSVDDecomposition
from fedot_ind.core.operation.transformation.data.eigen import combine_eigenvectors
from fedot_ind.core.operation.transformation.regularization.spectrum import reconstruct_basis, \
    singular_value_hard_threshold

supported_types = (pd.Series, np.ndarray, list)


class SpectrumDecomposer:
    """Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
    recorded at equal intervals.

    Args:
        time_series: The time series to decompose.
        window_length: The length of the window to use. Defaults to None.
        save_memory: Whether to save memory by not storing the elementary matrices. Defaults to True.

    """

    def __init__(self, data, ts_length, threshold: Optional[int] = None):
        if type(data) == list:
            rank = round(data[0].shape[0] / 10)
            beta = data[0].shape[0] / data[0].shape[1]

        self.svd_estimator = RSVDDecomposition()
        self.low_rank_approximation = True

        self.svd = lambda x: ListMonad(self.svd_estimator.rsvd(tensor=x,
                                                               approximation=self.low_rank_approximation,
                                                               regularized_rank=threshold))

        self.threshold = lambda Monoid: ListMonad([Monoid[0],
                                                   Monoid[1][:threshold],
                                                   Monoid[2]])

        self.multi_threshold = lambda Monoid: ListMonad([Monoid[1],
                                                         list(map(lambda x: singular_value_hard_threshold(
                                                             singular_values=x,
                                                             beta=beta,
                                                             threshold=None), Monoid[0])),
                                                         Monoid[2].T]) if threshold is None else ListMonad([Monoid[1][
                                                                                                            :,
                                                                                                            :threshold],
                                                                                                            Monoid[0][
                                                                                                            :,
                                                                                                            :threshold],
                                                                                                            Monoid[2][
                                                                                                            :,
                                                                                                            :threshold].T])

        self.data_driven_basis = lambda x: ListMonad(reconstruct_basis(x[0],
                                                                       x[1],
                                                                       x[2],
                                                                       ts_length=ts_length))

        self.tensor_decomposition = lambda x: ListMonad(parafac(tl.tensor(x), rank=rank).factors)

