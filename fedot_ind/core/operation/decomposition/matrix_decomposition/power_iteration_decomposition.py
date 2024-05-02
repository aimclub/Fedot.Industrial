import math

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.filtration.channel_filtration import _detect_knee_point
from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold, \
    sv_to_explained_variance_ratio


class RSVDDecomposition:
    def __init__(self,
                 rank: int = None
                 ):
        self.rank = rank

    def _init_random_params(self, tensor):
        # Percent of sampling columns. By default - 70%
        projection_rank = math.ceil(min(tensor.shape) / 1.5)
        # Polynom degree for power iteration procedure.
        self.poly_deg = 3
        # Create random matrix for projection/
        self.random_projection = np.random.randn(
            tensor.shape[1], projection_rank)

    def _compute_matrix_approximation(self, Ut, block, tensor, rank):
        Ut_ = Ut[:, :rank]
        tensor_approx = block @ Ut_
        reconstr_m = tensor_approx @ tensor_approx.T @ tensor
        return reconstr_m

    def _spectrum_regularization(self,
                                 spectrum: np.array,
                                 reg_type: str = 'hard_thresholding'):
        if reg_type == 'explained_dispersion':
            low_rank = sv_to_explained_variance_ratio(spectrum, 3)[1]
        elif reg_type == 'hard_thresholding':
            low_rank = len(singular_value_hard_threshold(spectrum))
        return max(low_rank, 2)

    def _matrix_approx_regularization(self, low_rank, Ut, block, tensor):
        if low_rank == 1:
            return low_rank
        else:
            list_of_rank = list(range(1, low_rank + 1, 1))
            reconstr_matrix = [self._compute_matrix_approximation(
                Ut, block, tensor, rank) for rank in list_of_rank]
            fro_norms = [abs(np.linalg.norm(tensor - reconstr_m, 'fro') / np.linalg.norm(tensor) * 100)
                         for reconstr_m in reconstr_matrix]
            regularized_rank = _detect_knee_point(
                values=fro_norms, indices=list(range(len(fro_norms))))
            regularized_rank = len(regularized_rank)
            # deriviate_of_error = abs(np.diff(fro_norms))
            # regularized_rank = len(
            #     deriviate_of_error[deriviate_of_error > 1]) + 1
        return regularized_rank

    def rsvd(self,
             tensor: np.array,
             approximation: bool = False,
             regularized_rank: int = None,
             reg_type: str = 'hard_thresholding') -> list:
        """Block Krylov subspace method for computing the SVD of a matrix with a low computational cost.

        Args:
            tensor: matrix to decompose
            approximation: if True, the matrix approximation will be computed
            regularized_rank: rank of the matrix approximation
            reg_type: type of regularization. 'hard_thresholding' or 'explained_dispersion'

        Returns:
            u, s, vt: decomposition

        """
        # Return classic svd decomposition with chosen type of spectrum thresholding
        if not approximation:
            # classic svd decomposition
            Ut, St, Vt = np.linalg.svd(tensor, full_matrices=False)
            # Compute low rank.
            low_rank = self._spectrum_regularization(St, reg_type=reg_type)
            if regularized_rank is not None:
                low_rank = regularized_rank
            # Return first n eigen components.
            U_, S_, V_ = Ut[:, :low_rank], St[:low_rank], Vt[:low_rank, :]
            return [U_, S_, V_]
        else:
            # First step. Initialize random matrix params.
            self._init_random_params(tensor)
            # Second step. Transform initial matrix to Gram. matrix
            AAT = tensor @ tensor.T
            # Third step. Power iteration procedure.First we raise the Gram matrix to the chosen degree.
            # This step is necessary in order to obtain a more "pronounced" spectrum (in which the eigenvalues
            # are well separated from each other). The important point is that the exponentiation procedure only changes
            # the eigenvalues but does not change the eigenvectors. Next, the resulting matrix is multiplied with the
            # original matrix ("overweighing" the column space) and then multiplied with a random matrix
            # in order to reduce the dimension and facilitate the procedure for "large" matrices.
            sampled_tensor = np.linalg.matrix_power(
                AAT, self.poly_deg) @ tensor @ self.random_projection
            # Fourth step. Orthogonalization of the resulting "sampled" matrix creates for us a basis of eigenvectors.
            sampled_tensor_orto, _ = np.linalg.qr(
                sampled_tensor, mode='reduced')
            # Fifth step. Project initial Gramm matrix on new basis obtained from "sampled matrix".
            M = sampled_tensor_orto.T @ AAT @ sampled_tensor_orto
            # Six step. Classical svd decomposition with choosen type of spectrum thresholding
            Ut, St, Vt = np.linalg.svd(M, full_matrices=False)
            # Compute low rank.
            low_rank = self._spectrum_regularization(St, reg_type=reg_type)
            # Seven step. Compute matrix approximation and choose new low_rank
            if regularized_rank is None:
                regularized_rank = self._matrix_approx_regularization(
                    low_rank, Ut, sampled_tensor_orto, tensor)
            # Eight step. Return matrix approximation.
            reconstr_tensor = self._compute_matrix_approximation(
                Ut, sampled_tensor_orto, tensor, regularized_rank)
            U_, S_, V_ = np.linalg.svd(reconstr_tensor, full_matrices=False)

            return [U_, S_, V_]
