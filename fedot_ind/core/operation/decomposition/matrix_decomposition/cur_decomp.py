from typing import Tuple
from cur import cur_decomposition
import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import TruncatedSVD


class CURDecomposition:
    def __init__(self, rank):
        self.rank = rank

    def fit_transform(self, matrix: np.ndarray) -> tuple:

        array = np.array(matrix.copy())
        c, w, r = self.select_rows_cols(array)
        x, sigma, y = np.linalg.svd(w)
        u = y.T @ np.diag(np.square(1/sigma)) @ x.T

        return c, u, r

    def select_rows_cols(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        col_norms = np.sum(matrix ** 2, axis=0)
        row_norms = np.sum(matrix ** 2, axis=1)

        # Compute the probabilities for selecting columns and rows
        p = col_norms / np.sum(col_norms)
        q = row_norms / np.sum(row_norms)

        # Select k columns and rows based on the probabilities p and q
        selected_cols = np.random.choice(matrix.shape[1], size=self.rank, replace=False, p=p)
        selected_rows = np.random.choice(matrix.shape[0], size=self.rank, replace=False, p=q)

        row_scale_factors = 1 / np.sqrt(self.rank * q[selected_rows])
        col_scale_factors = 1 / np.sqrt(self.rank * p[selected_cols])

        C_matrix = matrix[:, selected_cols] * col_scale_factors
        R_matrix = matrix[selected_rows, :] * row_scale_factors[:, np.newaxis]
        W_matrix = matrix[selected_rows, :][:, selected_cols]
        return C_matrix, W_matrix, R_matrix

    @staticmethod
    def make_matrix_from_ts(time_series: np.ndarray, window: int) -> np.ndarray:
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


if __name__ == '__main__':
    rank_ = 10
    cur = CURDecomposition(rank=rank_)
    # arr = np.array([[1, 1, 1, 0, 0],
    #                 [3, 3, 3, 0, 0],
    #                 [4, 4, 4, 0, 0],
    #                 [5, 5, 5, 0, 0],
    #                 [0, 0, 0, 4, 4],
    #                 [0, 0, 0, 5, 5],
    #                 [0, 0, 0, 2, 2]])
    ts = np.random.rand(100)
    arr = cur.make_matrix_from_ts(ts, 10)
    # arr = np.random.rand(100, 100)

    C, U, R = cur.fit_transform(arr)
    arr_approx_my = C @ U @ R
    ts_approx_my = cur.matrix_to_ts(arr_approx_my)

    c_, u_, r_ = cur_decomposition(arr, rank_)
    arr_approx_other = c_ @ u_ @ r_
    ts_approx_other = cur.matrix_to_ts(arr_approx_other)

    # L, S, R = np.linalg.svd(arr)
    # arr_svd_approx = L @ np.diag(S) @ R
    arr_svd_approx = TruncatedSVD(n_components=rank_).fit_transform(arr)
    ts_svd_approx = cur.matrix_to_ts(arr_svd_approx)
    # arr_svd_approx = L[:, :rank_] @ np.diag(S[:rank_]) @ R[:rank_, :]

    error_svd = np.linalg.norm(arr - arr_svd_approx)
    error_cur = np.linalg.norm(arr - arr_approx_my)
    error_other_cur = np.linalg.norm(arr - arr_approx_other)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(ts, label='original')
    ax.plot(ts_approx_my, label='my cur')
    ax.plot(ts_approx_other, label='other cur')
    ax.plot(ts_svd_approx, label='svd')
    ax.legend()
    ax.set_title(f'rank = {rank_}, error_svd = {error_svd}, error_cur = {error_cur}, error_other_cur = {error_other_cur}')
    plt.show()

    _ = 1
