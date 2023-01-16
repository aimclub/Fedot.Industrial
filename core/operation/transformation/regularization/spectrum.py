import numpy as np
from sklearn.preprocessing import MinMaxScaler


def sv_to_explained_variance_ratio(singular_values, rank):
    """
    Calculate the explained variance ratio of the singular values.

    Parameters
    ----------
    singular_values : array-like, shape (n_components,)
        Singular values.
    rank : int
        Number of singular values to use.

    Returns
    -------
    explained_variance : float
        Explained variance ratio.
    n_components : int
        Number of singular values to use.
    """
    n_components = [x / sum(singular_values) * 100 for x in singular_values]
    n_components = n_components[:rank]
    explained_variance = sum(n_components)
    n_components = rank
    return explained_variance, n_components


def singular_value_hard_threshold(singular_values, rank=None, threshold=2.858):
    """
    Calculate the hard threshold for the singular values.

    Parameters
    ----------
    singular_values : array-like, shape (n_components,)
        Singular values.
    rank : int
        Number of singular values to use.
    threshold : float
        Threshold value.

    Returns
    -------
    adjusted_rank : int
        Adjusted rank.
    """

    rank = len(singular_values) if rank is None else rank
    # Scale the singular values between 0 and 1.
    singular_values = MinMaxScaler(feature_range=(0, 1)).fit_transform(singular_values.reshape(-1, 1))[:, 0]
    # Find the median of the singular values.
    median_sv = np.median(singular_values[:rank])
    # Find the adjusted rank.
    sv_threshold = threshold * median_sv
    # Find the threshold value.
    adjusted_rank = np.sum(singular_values >= sv_threshold)
    # If the adjusted rank is 0, set it to 2.
    if adjusted_rank == 0:
        adjusted_rank = 2
    return adjusted_rank
