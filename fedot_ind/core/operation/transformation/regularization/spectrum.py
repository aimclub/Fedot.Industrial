from typing import Union

import numpy as np


def sv_to_explained_variance_ratio(singular_values, rank):
    """Calculate the explained variance ratio of the singular values.

    Args:
        singular_values (array-like, shape (n_components,)): Singular values.
        rank (int): Number of singular values to use.

    Returns:
        explained_variance (float): Explained variance ratio.
        n_components (int): Number of singular values to use.

    """
    n_components = [x / sum(singular_values) * 100 for x in singular_values][:rank]
    explained_variance = sum(n_components)
    n_components = rank
    return explained_variance, n_components


def singular_value_hard_threshold(singular_values: np.array,
                                  rank: Union[int, None] = None,
                                  beta: Union[float, None] = None,
                                  threshold: Union[float, None] = None) -> list:
    """Calculate the hard threshold for the singular values.

    Args:
        singular_values (array-like, shape (n_components,)): Singular values.
        rank (int): Number of singular values to use.
        beta (float): Beta value is a ratio of initial matrix dimensions.
        threshold (float): Desirable Threshold value.

    Returns:
        adjusted_rank (int): Adjusted rank.

    """
    if rank is not None:
        return singular_values[:rank]
    else:
        # Find the median of the singular values
        singular_values = [s_val for s_val in singular_values if s_val > 0.001]

        if len(singular_values) == 1:
            return singular_values[:1]
        median_sv = np.median(singular_values[:rank])
        # Find the adjusted rank
        if threshold is None:
            threshold = 0.56 * np.power(beta, 3) - 0.95 * np.power(beta, 2) + 1.82 * beta + 1.43
        sv_threshold = threshold * median_sv
        # Find the threshold value
        adjusted_rank = np.sum(singular_values >= sv_threshold)
        # If the adjusted rank is 0, recalculate the threshold value
        if adjusted_rank == 0:
            # sv_threshold = 2.31 * median_sv
            # adjusted_rank = np.sum(singular_values >= sv_threshold)
            adjusted_rank = 1
        return singular_values[:adjusted_rank]


def reconstruct_basis(U, Sigma, VT, ts_length):
    """Reconstruct the basis of the matrix.

    Args:
        U (array-like, shape (n_samples, n_components)): Left singular vectors.
        Sigma (array-like, shape (n_components,)): Singular values.
        VT (array-like, shape (n_components, n_features)): Right singular vectors.
        ts_length (int): Length of the time series.

    Returns:
        TS_comps (array-like, shape (ts_length, n_components)): Reconstructed basis.

    """

    if len(Sigma.shape) > 1:
        multi_reconstruction = lambda x: reconstruct_basis(U=U, Sigma=x, VT=VT, ts_length=ts_length)
        TS_comps = list(map(multi_reconstruction, Sigma))
    else:
        rank = Sigma.shape[0]
        TS_comps = np.zeros((ts_length, rank))
        for i in range(rank):
            X_elem = Sigma[i] * np.outer(U[:, i], VT[i, :])
            X_rev = X_elem[::-1]
            eigenvector = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]
            TS_comps[:, i] = eigenvector
    return TS_comps
