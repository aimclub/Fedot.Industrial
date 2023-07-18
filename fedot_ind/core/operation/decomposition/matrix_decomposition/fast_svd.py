import numpy as np
from scipy.linalg import qr
from sklearn.preprocessing import MinMaxScaler
from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold
import math


class RSVDDecomposition:
    def __init__(self, rank: int = None):
        self.rank = rank

    def _init_random_params(self, tensor):
        # Rank of the decomposition
        projection_rank = math.ceil(min(tensor.shape) / 1.5)
        # Power of the matrix
        self.poly_deg = 3
        self.random_projection = np.random.randn(tensor.shape[1], projection_rank)

    def _compute_matrix_approximation(self, Ut, block, tensor, rank):
        Ut_ = Ut[:, :rank]
        tensor_approx = block @ Ut_
        reconstr_m = tensor_approx @ tensor_approx.T @ tensor
        return reconstr_m

    def _regularize_rank(self, low_rank, Ut, block, tensor, l_reg: float = 0.2):
        spectral_norms, fro_norms = [], []
        list_of_rank = list(range(1, low_rank + 1, 1))
        for rank in list_of_rank:
            reconstr_m = self._compute_matrix_approximation(Ut, block, tensor, rank)
            spectral_norms.append(abs(np.linalg.norm(tensor - reconstr_m, 2)))
            fro_norms.append(abs(np.linalg.norm(tensor - reconstr_m, 'fro')))
        scaled_spectral = MinMaxScaler().fit_transform(np.array(spectral_norms).reshape(-1, 1))
        scaled_fro = MinMaxScaler().fit_transform(np.array(fro_norms).reshape(-1, 1))
        # scaled_rank = MinMaxScaler().fit_transform(np.array(list_of_rank).reshape(-1, 1))
        aprox_error = (1 - l_reg) * scaled_spectral + l_reg * scaled_fro
        aprox_error = aprox_error.reshape(-1)
        deriviate_of_error = abs(np.diff(aprox_error))
        deriviate_of_error = deriviate_of_error[deriviate_of_error > 0.01]
        #first_gap_idx = np.where(deriviate_of_error == deriviate_of_error.max())[0][0]
        #regularized_rank = first_gap_idx+2
        error_threshold = np.median(deriviate_of_error)
        regularized_rank = np.sum(deriviate_of_error <= error_threshold)
        return regularized_rank

    def _evaluate_regularized_rank(self, low_rank, Ut, sampled_tensor_orto, tensor):
        if low_rank > 1:
            regularized_rank = self._regularize_rank(low_rank, Ut, sampled_tensor_orto, tensor)
        else:
            regularized_rank = 1
        return regularized_rank

    def rsvd(self, tensor, approximation: bool = False, regularized_rank: int = None) -> list:
        """Block Krylov subspace method for computing the SVD of a matrix with a low computational cost.

        Args:
            tensor (array (M, N) array_like):
            k (int): rank of the decomposition
            block_size (int): size of the block
            num_iter (int): number of iterations

        Returns:
            u, s, vt (array_like): decomposition

        Notes:
        :param regularized_rank:
        :param approximation:

        """

        if not approximation:
            Ut, St, Vt = np.linalg.svd(tensor, full_matrices=False)
            low_rank = len(singular_value_hard_threshold(St, beta=tensor.shape[0] / tensor.shape[1]))
            if regularized_rank is not None:
                low_rank = regularized_rank
            U_, S_, V_ = Ut[:, :low_rank], St[:low_rank], Vt[:low_rank, :]
            return [U_, S_, V_]
        else:
            self._init_random_params(tensor)
            AAT = tensor @ tensor.T
            sampled_tensor = np.linalg.matrix_power(AAT, self.poly_deg) @ tensor @ self.random_projection
            # Orthogonalization of the sampled_tensor
            sampled_tensor_orto, _ = qr(sampled_tensor, mode='economic')

            M = sampled_tensor_orto.T @ AAT @ sampled_tensor_orto
            Ut, St, Vt = np.linalg.svd(M, full_matrices=False)
            low_rank = len(singular_value_hard_threshold(St, beta=tensor.shape[0] / tensor.shape[1]))

            if regularized_rank is None:
                regularized_rank = self._evaluate_regularized_rank(low_rank, Ut, sampled_tensor_orto, tensor)

            reconstr_tensor = self._compute_matrix_approximation(Ut, sampled_tensor_orto, tensor, regularized_rank)
            U_, S_, V_ = np.linalg.svd(reconstr_tensor, full_matrices=False)
            return [U_, S_, V_]
