from typing import Tuple

from numpy import linalg as LA
from sklearn import preprocessing
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.repository.constanst_repository import DEFAULT_SVD_SOLVER


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
        self.column_space = 'Full'

    @staticmethod
    def _get_selection_rank(matrix):
        """
        Compute the selection rank for the CUR decomposition. It must be at least 4 times the rank of the matrix but not
        greater than the number of rows or columns of the matrix.

        Args:
            matrix: the matrix to decompose.

        Returns:
            the selection rank
        """
        tol = [0.5, 0.1, 0.05]
        n_samples = max(matrix.shape)
        min_num_samples = johnson_lindenstrauss_min_dim(n_samples, eps=tol).tolist()
        return max([x if x < n_samples else n_samples for x in min_num_samples])

    def get_aproximation_error(self, original_tensor, cur_matrices: tuple):
        C, U, R = cur_matrices
        return np.linalg.norm(original_tensor - C @ U @ R)

    def _balance_target(self, target):
        classes = np.unique(target)
        self.classes_idx = [np.where(target == cls)[0] for cls in classes]

    def fit_transform(self, feature_tensor: np.ndarray,
                      target: np.ndarray = None) -> tuple:
        feature_tensor = feature_tensor.squeeze()
        # transformer = random_projection.SparseRandomProjection().fit_transform(target)
        self.selection_rank = self._get_selection_rank(feature_tensor)
        self._balance_target(target)
        # create sub matrices for CUR-decompostion
        array = np.array(feature_tensor.copy())
        c, w, r = self.select_rows_cols(array)
        if self.return_samples:
            sampled_tensor = feature_tensor[:, self.column_indices]
            sampled_tensor = sampled_tensor[self.row_indices, :]
        else:
            # evaluate pseudoinverse for W - U^-1
            X, Sigma, y_T = DEFAULT_SVD_SOLVER(w, full_matrices=False)
            Sigma_plus = np.linalg.pinv(np.diag(Sigma))
            # aprox U using pseudoinverse
            u = y_T.T @ Sigma_plus @ Sigma_plus @ X.T
            sampled_tensor = (c, u, r)
            self.get_aproximation_error(feature_tensor, sampled_tensor)
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
        matrix = preprocessing.MinMaxScaler().fit_transform(np.nan_to_num(matrix))
        col_norms, row_norms = np.nan_to_num(LA.norm(matrix, axis=0)), np.nan_to_num(LA.norm(matrix, axis=1))
        matrix_norm = LA.norm(matrix, 'fro')  # np.sum(np.power(matrix, 2))

        # Compute the probabilities for selecting columns and rows
        col_probs, row_probs = col_norms / matrix_norm, row_norms / matrix_norm

        is_matrix_tall = self.selection_rank > matrix.shape[1]
        col_rank = self.selection_rank if not is_matrix_tall or self.column_space == 'Full' \
            else len([prob for prob in col_probs if prob > 0.01])
        row_rank = round(self.selection_rank / len(self.classes_idx)) if is_matrix_tall else col_rank

        self.column_indices = np.sort(np.argsort(col_probs)[-col_rank:])
        self.row_indices = np.concatenate([np.sort(np.argsort(row_probs[cls_idx])[-row_rank:])
                                           for cls_idx in self.classes_idx])

        row_scale_factors = 1 / \
            np.sqrt(self.selection_rank * row_probs[self.row_indices])
        col_scale_factors = 1 / \
            np.sqrt(self.selection_rank * col_probs[self.column_indices])

        C_matrix = matrix[:, self.column_indices] * col_scale_factors
        R_matrix = matrix[self.row_indices, :] * row_scale_factors[:, np.newaxis]
        W_matrix = matrix[self.row_indices, :][:, self.column_indices]
        # Select k columns and rows based on the probabilities p and q
        # row_probs = preprocessing.Normalizer(norm='l1').fit_transform(row_probs.reshape(1, -1)).flatten()
        # selected_cols = np.random.choice(matrix.shape[1], size=self.rank, replace=False, p=col_probs)
        # selected_rows = np.random.choice(matrix.shape[0], size=row_rank, replace=False, p=row_probs)
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
