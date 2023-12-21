from typing import Tuple

import numpy as np


class CURDecomposition:
    def __init__(self, rank):
        self.selection_rank = None
        self.rank = rank
        self.column_indices = None
        self.row_indices = None

    def _get_selection_rank(self, rank, matrix):
        """
        Compute the selection rank for the CUR decomposition. It must be at least 4 times the rank of the matrix but not
        greater than the number of rows or columns of the matrix.

        Args:
            rank: the rank of the matrix.
            matrix: the matrix to decompose.

        Returns:
            the selection rank
        """

        return min(4 * rank, min(matrix.shape))

    def fit_transform(self, matrix: np.ndarray) -> tuple:
        self.selection_rank = self._get_selection_rank(self.rank, matrix)

        array = np.array(matrix.copy())
        c, w, r = self.select_rows_cols(array)

        X, Sigma, y_T = np.linalg.svd(w)
        Sigma_plus = np.linalg.pinv(np.diag(Sigma))

        u = y_T.T @ Sigma_plus @ Sigma_plus @ X.T
        return c, u, r

    def reconstruct_basis(self, C, U, R, ts_length):
        rank = U.shape[1]
        TS_comps = np.zeros((ts_length, rank))
        for i in range(rank):
            X_elem = np.outer(C @ U[:, i], R[i, :])
            X_rev = X_elem[::-1]
            eigenvector = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]
            TS_comps[:, i] = eigenvector
        return TS_comps

    def select_rows_cols(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        col_norms = np.sum(matrix ** 2, axis=0)
        row_norms = np.sum(matrix ** 2, axis=1)

        matrix_norm = np.sum(np.power(matrix, 2))

        # Compute the probabilities for selecting columns and rows
        col_probs = col_norms / matrix_norm
        row_probs = row_norms / matrix_norm

        # Select k columns and rows based on the probabilities p and q
        # selected_cols = np.random.choice(matrix.shape[1], size=self.rank, replace=False, p=col_probs)
        # selected_rows = np.random.choice(matrix.shape[0], size=self.rank, replace=False, p=row_probs)
        #
        selected_cols = np.sort(np.argsort(col_probs)[-self.selection_rank:])
        selected_rows = np.sort(np.argsort(row_probs)[-self.selection_rank:])
        # selected_cols = np.argsort(col_probs)[-self.rank:]
        # selected_rows = np.argsort(row_probs)[-self.rank:]

        row_scale_factors = 1 / np.sqrt(self.selection_rank * row_probs[selected_rows])
        col_scale_factors = 1 / np.sqrt(self.selection_rank * col_probs[selected_cols])

        C_matrix = matrix[:, selected_cols] * col_scale_factors
        R_matrix = matrix[selected_rows, :] * row_scale_factors[:, np.newaxis]
        W_matrix = matrix[selected_rows, :][:, selected_cols]
        return C_matrix, W_matrix, R_matrix

    @staticmethod
    def ts_to_matrix(time_series: np.ndarray, window: int) -> np.ndarray:
        """Make matrix from ts using window"""

        matrix = np.zeros((len(time_series) - window + 1, window))
        for i in range(len(time_series) - window + 1):
            matrix[i] = time_series[i:i + window]
        return matrix

    @staticmethod
    def matrix_to_ts(matrix: np.ndarray) -> np.ndarray:
        """Make ts from matrix"""

        ts = np.zeros(matrix.shape[0] + matrix.shape[1] - 1)
        for i in range(matrix.shape[0]):
            ts[i:i + matrix.shape[1]] += matrix[i]
        return ts


def get_random_sparse_matrix(size: tuple):
    """Generate random sparse matrix with size = size"""

    matrix = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            if np.random.rand() < 0.1:
                matrix[i, j] = np.random.rand()
    return matrix
