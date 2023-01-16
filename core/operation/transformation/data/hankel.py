import numpy as np
from scipy.linalg import hankel


class HankelMatrix:
    """
    This class implements an algorithm for converting an original time series into a Hankel matrix.
    """

    def __init__(self, data, subseq_length):
        """
        Initialize the class with the data and subseq_length parameters.

        Parameters
        ----------
        data : np.array
            An array of data that is either a one-dimensional or multidimensional time series.
        subseq_length : int
            The length of the subsequence.
        """
        self.data = data
        self.subseq_length = subseq_length

    def one_dimensional_transform(self):
        """
        Transform a one-dimensional time series into a Hankel matrix.

        Returns
        -------
        np.array
            The Hankel matrix.
        """
        hankel_matrix = []
        for i in range(len(self.data) - self.subseq_length + 1):
            hankel_matrix.append(self.data[i:i + self.subseq_length])
        return np.array(hankel_matrix)

    def multidimensional_transform(self):
        """
        Transform a multidimensional time series into a Hankel matrix.

        Returns
        -------
        np.array
            The Hankel matrix.
        """
        hankel_matrix = []
        for i in range(len(self.data) - self.subseq_length + 1):
            hankel_matrix.append(self.data[i:i + self.subseq_length, :])
        return np.array(hankel_matrix)


    # def ts_vector_to_trajectory_matrix(timeseries, L, K):
    #     hankelized = hankel(timeseries, np.zeros(L)).T
    #     hankelized = hankelized[:, :K]
    #     return hankelized
    #
    # def ts_matrix_to_trajectory_matrix(self, timeseries, L, K):
    #     P, N = timeseries.shape
    #
    #     trajectory_matrix = [
    #         self.ts_vector_to_trajectory_matrix(timeseries[p, :], L, K)
    #         for p in range(P)
    #     ]
    #
    #     trajectory_matrix = np.concatenate(trajectory_matrix, axis=1)
    #     return trajectory_matrix