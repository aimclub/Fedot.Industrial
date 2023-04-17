import numpy as np


def quantile(column, q: str):
    return np.quantile(a=column, q=q)


def softmax(w, theta=1.0) -> np.ndarray:
    """Takes a vector w of S N-element and returns a vectors where each column
        of the vector sums to 1, with elements exponentially proportional to the
        respective elements in N.

        Args:
            w: array of shape = [N,  M].
            theta: float, parameter, used as a multiplier prior to exponentiation (default = 1.0).

        Returns:
            array of shape = [N, M]. Which the sum of each row sums to 1 and the elements are exponentially
            proportional to the respective elements in N

        """
    w = np.atleast_2d(w)
    e = np.exp(np.array(w) / theta)
    dist = e / np.sum(e, axis=1).reshape(-1, 1)
    return dist


stat_methods_default = {
    'mean_': np.mean,
    'median_': np.median,
    'std_': np.std,
    'var_': np.var,
    'q5_': quantile,
    'q25_': quantile,
    'q75_': quantile,
    'q95_': quantile,
}

stat_methods_ensemble = {
    'MeanEnsemble': np.mean,
    'MedianEnsemble': np.median,
    'MinEnsemble': np.min,
    'MaxEnsemble': np.max,
    'ProductEnsemble': np.prod
}

stat_methods_full = {
    'mean_': np.mean,
    'median_': np.median,
    'lambda_less_zero': lambda x: x < 0.01,
    'std_': np.std,
    'var_': np.var,
    'max': np.max,
    'min': np.min,
    'q5_': quantile,
    'q25_': quantile,
    'q75_': quantile,
    'q95_': quantile,
    'sum_': np.sum,
    'dif_': np.diff
}

hyper_param_dict = {'statistical_methods': stat_methods_default,
                    'statistical_methods_extra': stat_methods_full,
                    'stat_methods_ensemble': stat_methods_ensemble}


def select_hyper_param(param_name):
    return hyper_param_dict[param_name]
