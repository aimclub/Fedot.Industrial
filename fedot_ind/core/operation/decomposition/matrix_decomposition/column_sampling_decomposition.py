from typing import Tuple

from numpy import linalg as LA

from fedot_ind.core.architecture.settings.computational import backend_methods as np


class CURDecomposition:
    def __init__(self, rank,
                 return_samples: bool = True):
        self.selection_rank = None
        self.return_samples = return_samples
        if not self.return_samples:
            self.rank = min(20000, rank)
        else:
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

        return min(rank, max(matrix.shape))

    def get_aproximation_error(self, original_tensor, cur_matrices: tuple):
        C, U, R = cur_matrices
        return np.linalg.norm(original_tensor - C @ U @ R)

    def _plot_aproximation_error(self):
        pass
        # f,a = plt.subplots(2, 1, figsize=(10, 10))
        # # a[0].plot(ranks, svd_errors, label='svd')
        # a[1].plot(ranks, cur_errors, label='cur')
        # a[0].set_title('svd')
        # a[1].set_title('cur')
        # plt.legend()
        # plt.show()

    def fit_transform(self, feature_tensor: np.ndarray, target: np.ndarray = None) -> tuple:
        self.selection_rank = self._get_selection_rank(self.rank, feature_tensor)
        # create sub matrices for CUR-decompostion
        array = np.array(feature_tensor.copy())
        c, w, r = self.select_rows_cols(array)
        if self.return_samples:
            sampled_tensor = feature_tensor[:, self.column_indices]
            sampled_tensor = sampled_tensor[self.row_indices, :]
        else:
            # evaluate pseudoinverse for W - U^-1
            X, Sigma, y_T = np.linalg.svd(w, full_matrices=False)
            Sigma_plus = np.linalg.pinv(np.diag(Sigma))
            # aprox U using pseudoinverse
            u = y_T.T @ Sigma_plus @ Sigma_plus @ X.T
            sampled_tensor = (c, u, r)
            error = self.get_aproximation_error(feature_tensor, sampled_tensor)
        if target is not None:
            target = target[self.row_indices]
        return sampled_tensor, target

    def reconstruct_basis(self, C, U, R, ts_length):
        # if len(U.shape) > 1:
        #     multi_reconstruction = lambda x: self.reconstruct_basis(C=C, U=U, R=x, ts_length=ts_length)
        #     TS_comps = list(map(multi_reconstruction, R))
        # else:
        rank = U.shape[1]
        TS_comps = np.zeros((ts_length, rank))
        for i in range(rank):
            X_elem = np.outer(C @ U[:, i], R[i, :])
            X_rev = X_elem[::-1]
            eigenvector = [X_rev.diagonal(
                j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]
            TS_comps[:, i] = eigenvector
        return TS_comps

    def select_rows_cols(
            self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Evaluate norms for columns and rows
        matrix = np.nan_to_num(matrix)
        col_norms = LA.norm(matrix, axis=0)
        row_norms = LA.norm(matrix, axis=1)
        col_norms = np.nan_to_num(col_norms)
        row_norms = np.nan_to_num(row_norms)
        matrix_norm = LA.norm(matrix, 'fro')  # np.sum(np.power(matrix, 2))

        # Compute the probabilities for selecting columns and rows
        col_probs = col_norms / matrix_norm
        row_probs = row_norms / matrix_norm

        # Select k columns and rows based on the probabilities p and q
        # selected_cols = np.random.choice(matrix.shape[1], size=self.rank, replace=False, p=col_probs)
        # selected_rows = np.random.choice(matrix.shape[0], size=self.rank, replace=False, p=row_probs)
        selected_cols = np.sort(np.argsort(col_probs)[-self.selection_rank:])
        selected_rows = np.sort(np.argsort(row_probs)[-self.selection_rank:])

        self.row_indices = selected_rows
        self.column_indices = selected_cols
        row_scale_factors = 1 / \
                            np.sqrt(self.selection_rank * row_probs[selected_rows])
        col_scale_factors = 1 / \
                            np.sqrt(self.selection_rank * col_probs[selected_cols])

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


if __name__ == '__main__':
    from fedot_ind.tools.loader import DataLoader

    arr = np.array([[1, 1, 1, 0, 0],
                    [3, 3, 3, 0, 0],
                    [4, 4, 4, 0, 0],
                    [5, 5, 5, 0, 0],
                    [0, 0, 0, 4, 4],
                    [0, 0, 0, 5, 5],
                    [0, 0, 0, 2, 2]])

    (X_train, y_train), (X_test, y_test) = DataLoader('Lightning7').load_data()

    # init_ts = train[0].iloc[0, :].values
    # scaler = MinMaxScaler()
    # scaler.fit(init_ts.reshape(-1, 1))
    # single_ts = scaler.transform(init_ts.reshape(-1, 1)).reshape(-1)

    cur = CURDecomposition(rank=20)
    # M = cur.ts_to_matrix(single_ts, 30)
    C, U, R = cur.fit_transform(X_train)
    basis = cur.reconstruct_basis(C, U, R, X_train.shape[1])

    # rec_ts = cur.matrix_to_ts(C @ U @ R)
    # err = np.linalg.norm(single_ts - rec_ts)

    # plt.plot(init_ts, label='init_ts')
    # plt.plot(scaler.inverse_transform(rec_ts.reshape(-1, 1)), label='rec_ts')
    # plt.legend()
    # plt.show()
    _ = 1
