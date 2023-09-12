import numpy as np
import pandas as pd
import copy


def weighted_inner_product(F_i, F_j, window_length, ts_length):
    # Calculate the weights
    first = list(np.arange(window_length) + 1)
    second = [window_length] * (ts_length - 2*window_length)
    third = list(np.arange(window_length) + 1)[::-1]
    w = np.array(first + second + third)
    return w.dot(F_i * F_j)


def calculate_matrix_norms(TS_comps, window_length, ts_length):
    r = []
    for i in range(TS_comps.shape[1]):
        r.append(weighted_inner_product(TS_comps[:, i], TS_comps[:, i], window_length, ts_length))
    F_wnorms = np.array(r)
    F_wnorms = F_wnorms ** -0.5
    return F_wnorms


def calculate_corr_matrix(TS_comps, F_wnorms, window_length, ts_length):
    Wcorr = np.identity(TS_comps.shape[1])
    for i in range(Wcorr.shape[0]):
        for j in range(i + 1, Wcorr.shape[0]):
            Wcorr[i, j] = abs(
                weighted_inner_product(TS_comps[:, i], TS_comps[:, j], window_length, ts_length) *
                F_wnorms[i] * F_wnorms[j])
            Wcorr[j, i] = Wcorr[i, j]
    return Wcorr, [i for i in range(Wcorr.shape[0])]


def combine_eigenvectors(TS_comps, window_length,  correlation_level: float = 0.8):
    """Calculates the w-correlation matrix for the time series.

    Args:
        TS_comps (np.ndarray): The time series components.
        correlation_level (float): threshold value of Pearson correlation, using for merging eigenvectors.
        ts_length (int): The length of TS .
        window_length (int): The length of TS window.


    """
    combined_components = []
    ts_length = TS_comps.shape[0]
    # Calculated weighted norms
    F_wnorms = calculate_matrix_norms(TS_comps, window_length, ts_length)

    # Calculate Wcorr.
    Wcorr, components = calculate_corr_matrix(TS_comps, F_wnorms, window_length, ts_length)

    combined_components = []
    current_group = []
    for i in range(len(components)):
        if i == 0 or Wcorr[i, i-1] > correlation_level:
            current_group.append(TS_comps[:, i])
        else:
            combined_components.append(np.array(current_group).sum(axis=0))
            current_group = [TS_comps[:, i]]



    return combined_components
