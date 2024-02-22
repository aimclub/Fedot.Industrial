from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.repository.constanst_repository import SINGULAR_VALUE_BETA_THR, SINGULAR_VALUE_MEDIAN_THR


def sv_to_explained_variance_ratio(singular_values, dispersion_by_component):
    """Calculate the explained variance ratio of the singular values.

    Args:
        singular_values (array-like, shape (n_components,)): Singular values.
        rank (int): Number of singular values to use.

    Returns:
        explained_variance (int): Explained variance percent.
        n_components (int): Number of singular values to use.

    """
    singular_values = [abs(x) for x in singular_values]
    n_components = [x / sum(singular_values) * 100 for x in singular_values]
    n_components = [x for x in n_components if x > dispersion_by_component]
    explained_variance = sum(n_components)
    n_components = len(n_components)
    return explained_variance, n_components


def singular_value_hard_threshold(singular_values,
                                  rank=None,
                                  beta=None,
                                  threshold=SINGULAR_VALUE_MEDIAN_THR) -> list:
    """Calculate the hard threshold for the singular values.

    Args:
        singular_values (array-like, shape (n_components,)): Singular values.
        rank (int): Number of singular values to use.
        beta (float): Beta value.
        threshold (float): Threshold value.

    Returns:
        adjusted singular values array (array-like, shape (n_components,)): Adjusted array of singular values.

    """
    if rank is not None:
        return singular_values[:rank]
    else:
        # Find the median of the singular values
        singular_values = [s_val for s_val in singular_values if s_val > 0.01]
        if len(singular_values) == 1:
            return singular_values[:1]
        median_sv = np.median(singular_values[:rank])
        # Find the adjusted rank
        if threshold is None:
            threshold = SINGULAR_VALUE_BETA_THR(beta)
        sv_threshold = threshold * median_sv
        # Find the threshold value
        adjusted_rank = np.sum(singular_values >= sv_threshold)
        # If the adjusted rank is 0, recalculate the threshold value
        if adjusted_rank == 0:
            sv_threshold = 2.31 * median_sv
            adjusted_rank = max(np.sum(singular_values >= sv_threshold), 1)
        return singular_values[:adjusted_rank]


def reconstruct_basis(U, Sigma, VT, ts_length):
    if len(Sigma.shape) > 1:
        def multi_reconstruction(x):
            return reconstruct_basis(U=U, Sigma=x, VT=VT, ts_length=ts_length)

        TS_comps = list(map(multi_reconstruction, Sigma))
    else:
        rank = Sigma.shape[0]
        TS_comps = np.zeros((ts_length, rank))
        for i in range(rank):
            X_elem = Sigma[i] * np.outer(U[:, i], VT[i, :])
            X_rev = X_elem[::-1]
            eigenvector = [X_rev.diagonal(
                j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]
            TS_comps[:, i] = eigenvector
    return TS_comps
