import numpy as np
import pandas as pd
import copy


def weighted_inner_product(F_i, F_j, window_length, ts_length, subseq_length):
    # Calculate the weights
    first = list(np.arange(window_length) + 1)
    second = [ts_length] * (subseq_length - window_length - 1)
    third = list(np.arange(window_length) + 1)[::-1]
    w = np.array(first + second + third)
    return w.dot(F_i * F_j)


def calculate_matrix_norms(rank, TS_comps, window_length, ts_length, subseq_length):
    F_wnorms = np.array(
        [weighted_inner_product(TS_comps[:, i], TS_comps[:, i], window_length, ts_length, subseq_length) for i in
         range(rank)])
    F_wnorms = F_wnorms ** -0.5
    return F_wnorms


def calculate_corr_matrix(rank, TS_comps, F_wnorms, window_length, ts_length, subseq_length):
    Wcorr = np.identity(rank)
    components = [i for i in range(rank)]
    for i in components:
        for j in range(i + 1, rank):
            Wcorr[i, j] = abs(
                weighted_inner_product(TS_comps[:, i], TS_comps[:, j], window_length, ts_length, subseq_length) *
                F_wnorms[i] * F_wnorms[j])
            Wcorr[j, i] = Wcorr[i, j]
    return Wcorr, components


def combine_eigenvectors(TS_comps, rank, window_length, ts_length, subseq_length, correlation_level: float = 0.8):
    """Calculates the w-correlation matrix for the time series.

    Args:
        TS_comps (np.ndarray): The time series components.
        rank (int): The rank of the time series.
        correlation_level (float): threshold value of Pearson correlation, using for merging eigenvectors.
        subseq_length (int): The length of TS subseq.
        ts_length (int): The length of TS .
        window_length (int): The length of TS window.


    """
    combined_components = []

    # Calculated weighted norms
    F_wnorms = calculate_matrix_norms(rank, TS_comps, window_length, ts_length, subseq_length)

    # Calculate Wcorr.
    Wcorr, components = calculate_corr_matrix(rank, TS_comps, F_wnorms, window_length, ts_length, subseq_length)

    # Calculate Wcorr. and Select Correlated Eigenvectors.
    corr_dict = {i: [i for i, v in enumerate(Wcorr[:, i]) if v > correlation_level] for i in components}
    # copy of the dictionary for deleting keys
    filtred_dict = copy.deepcopy(corr_dict)

    # Select Correlated Eigenvectors.
    for list_of_corr_vectors in list(corr_dict.values()):
        final_component = None
        if len(list_of_corr_vectors) < 2:
            final_component = TS_comps[:, list_of_corr_vectors[0]]
            combined_components.append(final_component)
        else:
            for corr_vector in list_of_corr_vectors:
                if corr_vector in filtred_dict.keys():
                    if final_component is None:
                        final_component = np.sum(TS_comps[:, list_of_corr_vectors], axis=1)
                        combined_components.append(final_component)
                    del filtred_dict[corr_vector]

    combined_components = pd.DataFrame(combined_components).T

    return combined_components
