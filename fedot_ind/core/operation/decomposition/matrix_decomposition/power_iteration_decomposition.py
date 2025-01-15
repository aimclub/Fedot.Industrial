import math
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.filtration.channel_filtration import _detect_knee_point
from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold, \
    sv_to_explained_variance_ratio, eigencorr_matrix
from fedot_ind.core.repository.constanst_repository import DEFAULT_SVD_SOLVER, DEFAULT_QR_SOLVER


class RSVDDecomposition:
    """Randomized SVD decomposition with power iteration method.
    Implements the block Krylov subspace method for computing the SVD of a matrix with a low computational cost.
    The method is based on the power iteration procedure, which allows us to obtain a low-rank approximation of the
    matrix. The method is based on the following steps:
    1. Random projection of the matrix.
    2. Transformation of the initial matrix to the Gram matrix.
    3. Power iteration procedure.
    4. Orthogonalization of the resulting "sampled" matrix.
    5. Projection of the initial Gram matrix on the new basis obtained from the "sampled matrix".
    6. Classical svd decomposition with the chosen type of spectrum thresholding.
    7. Compute matrix approximation and choose a new low_rank.
    8. Return matrix approximation.

    Args:
        params: dictionary with parameters for the operation:
            rank: rank of the matrix approximation
            power_iter: polynom degree for power iteration procedure
            sampling_share: percent of sampling columns. By default - 70%

    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.rank = params.get('rank', 1)
        # Polynom degree for power iteration procedure.
        self.poly_deg = params.get('power_iter', 3)
        # Percent of sampling columns. By default - 70%
        self.projection_rank = params.get('sampling_share', 0.7)
        self.tolerance = params.get('tolerance', [0.5, 0.1])

    def _init_random_params(self, tensor):
        # Create random matrix for projection
        max_num_rows = 10000
        self.is_matrix_big = any([tensor.shape[0] > max_num_rows, tensor.shape[1] > max_num_rows])
        if self.is_matrix_big:
            self.projection_rank = self._get_selection_rank(tensor)
        else:
            self.projection_rank = math.ceil(min(tensor.shape) * self.projection_rank)
        self.is_matrix_tall = self.projection_rank > tensor.shape[1]
        if self.is_matrix_tall:
            self.random_projection = np.random.randn(self.projection_rank, tensor.shape[0])
        else:
            self.random_projection = np.random.randn(tensor.shape[1], self.projection_rank)

    def _get_selection_rank(self, matrix):
        """
        Compute the selection rank for the CUR decomposition.
        It must be at least 4 times the rank of the matrix but not
        greater than the number of rows or columns of the matrix.

        Args:
            matrix: the matrix to decompose.

        Returns:
            the selection rank
        """
        n_samples = max(matrix.shape)
        min_num_samples = johnson_lindenstrauss_min_dim(n_samples, eps=self.tolerance).tolist()
        return max([x if x < n_samples else n_samples for x in min_num_samples])

    def _compute_matrix_approximation(self, Ut, block, tensor, rank):
        Ut_ = Ut[:, :rank]
        tensor_approx = block @ Ut_
        reconstr_m = tensor_approx @ tensor_approx.T @ tensor
        return reconstr_m

    def _spectrum_regularization(self,
                                 spectrum: np.array,
                                 reg_type: str = 'hard_thresholding'):
        if reg_type == 'explained_dispersion':
            explained_disperesion, low_rank = sv_to_explained_variance_ratio(spectrum, 3)
            if explained_disperesion < 90 and low_rank < 3:
                low_rank = 'ill_conditioned'
        elif reg_type == 'hard_thresholding':
            low_rank = len(singular_value_hard_threshold(spectrum))
        else:
            regularized_rank = _detect_knee_point(
                values=spectrum, indices=list(range(len(spectrum))))
            low_rank = min(len(regularized_rank), 4)
        return low_rank

    def _matrix_approx_regularization(self, low_rank, Ut, block, tensor):
        if low_rank == 1:
            return low_rank
        else:
            list_of_rank = list(range(1, low_rank + 1, 1))
            reconstr_matrix = [self._compute_matrix_approximation(
                Ut, block, tensor, rank) for rank in list_of_rank]
            fro_norms = [abs(np.linalg.norm(tensor -
                                            reconstr_m, 'fro') /
                             np.linalg.norm(tensor) *
                             100) for reconstr_m in reconstr_matrix]
            regularized_rank = _detect_knee_point(
                values=fro_norms, indices=list(range(len(fro_norms))))
            regularized_rank = len(regularized_rank)
        return regularized_rank

    def _column_sampling(self, S, V, tensor, low_rank):
        column_basis = S @ V
        top_cols_idx = column_basis.argsort()[-low_rank:][::-1]
        return tensor[:, top_cols_idx]

    def _power_iteration_procedure(self, tensor, reg_type, regularized_rank, return_svd: bool = True,
                                   sampling_regime: str = None):
        # Second step. Transform initial matrix to Gram. matrix
        big_tall_matrix = all([self.is_matrix_big, self.is_matrix_tall])
        if big_tall_matrix:
            tensor_row_sampled = self.random_projection @ tensor  # For tall and big matrix we use "row-sampling" operator
            grammian = tensor_row_sampled @ tensor_row_sampled.T
            grammian_with_good_spectrum = np.linalg.matrix_power(grammian, self.poly_deg)
            sampled_tensor = grammian_with_good_spectrum @ tensor_row_sampled
        else:
            # Third step. Power iteration procedure. First we raise the Gram matrix to the chosen degree.
            # This step is necessary in order to obtain a more "pronounced" spectrum (in which the eigenvalues
            # are well separated from each other). The important point is that the exponentiation procedure only changes
            # the eigenvalues but does not change the eigenvectors. Next, the resulting matrix is multiplied with the
            # original matrix ("overweightning" the column space) and then multiplied with a random matrix
            # in order to reduce the dimension and facilitate the procedure for
            # "large" matrices.
            grammian = tensor @ tensor.T
            sampled_tensor = np.linalg.matrix_power(grammian, self.poly_deg) @ tensor @ self.random_projection
        # Fourth step. Orthogonalization of the resulting "sampled" matrix
        # creates for us a basis of eigenvectors.
        sampled_tensor_orto, _ = DEFAULT_QR_SOLVER(sampled_tensor, mode='reduced')
        # Fifth step. Project initial Gramm matrix on new basis obtained
        # from "sampled matrix".
        M = sampled_tensor_orto.T @ grammian @ sampled_tensor_orto
        # Six step. Classical svd decomposition with choosen type of
        # spectrum thresholding
        Ut, St, Vt = DEFAULT_SVD_SOLVER(M, full_matrices=False)
        # Compute low rank.
        low_rank = self._spectrum_regularization(St, reg_type=reg_type)
        # Seven step. Compute matrix approximation and choose new low_rank
        if regularized_rank is None and big_tall_matrix:
            self.regularized_rank = low_rank
        elif regularized_rank is not None:
            self.regularized_rank = min(low_rank, regularized_rank)
        else:
            self.regularized_rank = self._matrix_approx_regularization(
                low_rank, Ut, sampled_tensor_orto, tensor)
        # Eight step. Return matrix approximation.
        if big_tall_matrix and sampling_regime == 'column_sampling':
            reconstr_tensor = self._column_sampling(St, Vt, tensor, self.regularized_rank)
        else:
            reconstr_tensor = self._compute_matrix_approximation(Ut, sampled_tensor_orto, tensor, self.regularized_rank)
        return reconstr_tensor if not return_svd else DEFAULT_SVD_SOLVER(reconstr_tensor, full_matrices=False)

    def rsvd(self,
             tensor: np.array,
             approximation: bool = False,
             regularized_rank: int = None,
             reg_type: str = 'hard_thresholding',
             return_svd: bool = True,
             sampling_regime: str = None) -> list:
        """Block Krylov subspace method for computing the SVD of a matrix with a low computational cost.

        Args:
            tensor: matrix to decompose
            approximation: if True, the matrix approximation will be computed
            regularized_rank: rank of the matrix approximation
            reg_type: type of regularization. 'hard_thresholding' or 'explained_dispersion'

        Returns:
            u, s, vt: decomposition

        """
        # Return classic svd decomposition with chosen type of spectrum
        # thresholding
        if not approximation:
            tensor_approx = tensor
            # classic svd decomposition
            Ut, St, Vt = DEFAULT_SVD_SOLVER(tensor, full_matrices=False)
            # Compute low rank.
            low_rank = self._spectrum_regularization(St, reg_type=reg_type)
            is_rank_too_low = regularized_rank is not None and regularized_rank > low_rank
            if is_rank_too_low:
                low_rank = regularized_rank
            if low_rank == 'ill_conditioned':
                U_ = [Ut, St, Vt]
                V_ = eigencorr_matrix(Ut, St, Vt)
                S_ = low_rank  # spectrum # noise
            else:
                # Return first n eigen components.
                U_, S_, V_ = Ut[:, :low_rank], St[:low_rank], Vt[:low_rank, :]
            if sampling_regime == 'column_sampling':
                tensor_approx = self._column_sampling(S_, V_, tensor, low_rank)
            self.regularized_rank = low_rank
            return [U_, S_, V_] if return_svd else tensor_approx
        else:
            # First step. Initialize random matrix params.
            self._init_random_params(tensor)
            return self._power_iteration_procedure(tensor, reg_type, regularized_rank, return_svd, sampling_regime)
