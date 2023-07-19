import numpy as np


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


stat_methods_ensemble = {
    'MeanEnsemble': np.mean,
    'MedianEnsemble': np.median,
    'MinEnsemble': np.min,
    'MaxEnsemble': np.max,
    'ProductEnsemble': np.prod
}
