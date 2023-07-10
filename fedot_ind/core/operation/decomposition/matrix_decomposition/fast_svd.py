import numpy as np
from scipy.linalg import qr

from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold


def bksvd(tensor) -> list:
    """Block Krylov subspace method for computing the SVD of a matrix with a low computational cost.

    Args:
        tensor (array (M, N) array_like):
        k (int): rank of the decomposition
        block_size (int): size of the block
        num_iter (int): number of iterations

    Returns:
        u, s, vt (array_like): decomposition

    Notes:

    """
    import math

    # Rank of the decomposition
    k = math.ceil(min(tensor.shape) / 3)
    # Power of the matrix
    q = 3
    block_size = k

    _block = np.random.randn(tensor.shape[1], block_size)

    AAT = tensor @ tensor.T
    block = np.linalg.matrix_power(AAT, q) @ tensor @ _block
    # Orthogonalization of the block
    block, _ = qr(block, mode='economic')

    M = block.T @ AAT @ block
    Ut, St, Vt = np.linalg.svd(M, full_matrices=False)
    low_rank = len(singular_value_hard_threshold(St, beta=tensor.shape[0] / tensor.shape[1]))

    Ut_ = Ut[:, :low_rank]
    tensor_approx = block @ Ut_
    reconstr_m = tensor_approx @ tensor_approx.T @ tensor
    diff_norm = np.linalg.norm(reconstr_m) - np.linalg.norm(tensor)

    U_, S_, V_ = np.linalg.svd(reconstr_m, full_matrices=False)

    return [U_, S_, V_]
