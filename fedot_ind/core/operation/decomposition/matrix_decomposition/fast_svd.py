import numpy as np
from scipy.linalg import qr
from sklearn.preprocessing import MinMaxScaler
from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold


def compute_matrix_approximation(Ut, block, tensor, rank):
    Ut_ = Ut[:, :rank]
    tensor_approx = block @ Ut_
    reconstr_m = tensor_approx @ tensor_approx.T @ tensor
    return reconstr_m


def regularize_rank(low_rank, Ut, block, tensor, l_reg: float = 0.2):
    spectral_norms, fro_norms = [], []
    list_of_rank = list(range(1, low_rank + 1, 1))
    for rank in list_of_rank:
        reconstr_m = compute_matrix_approximation(Ut, block, tensor, rank)
        spectral_norms.append(abs(np.linalg.norm(tensor - reconstr_m, 2)))
        fro_norms.append(abs(np.linalg.norm(tensor - reconstr_m, 'fro')))
    scaled_spectral = MinMaxScaler().fit_transform(np.array(spectral_norms).reshape(-1, 1))
    scaled_fro = MinMaxScaler().fit_transform(np.array(fro_norms).reshape(-1, 1))
    #scaled_rank = MinMaxScaler().fit_transform(np.array(list_of_rank).reshape(-1, 1))
    aprox_error = (1 - l_reg) * scaled_spectral + l_reg * scaled_fro
    aprox_error = aprox_error.reshape(-1)
    deriviate_of_error = np.diff(aprox_error)
    error_threshold = np.median(deriviate_of_error)
    regularized_rank = np.sum(deriviate_of_error <= error_threshold)
    return regularized_rank


def bksvd(tensor) -> list:
    """Block Krylov subspace method for computing the SVD of a matrix with a low computational cost.

    Args:
        tensor (array (M, N) array_like):
        k (int): rank of the decomposition
        block_size (int): size of the block
        num_iter (int): number of iterations

    Returns:
        u, s, vt (array_like): decomposition

    """
    import math

    # Rank of the decomposition
    k = math.ceil(min(tensor.shape) / 1.5)
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
    if low_rank > 1:
        regularized_rank = regularize_rank(low_rank, Ut, block, tensor)
    else:
        regularized_rank = 1
    reconstr_m = compute_matrix_approximation(Ut, block, tensor, regularized_rank)
    U_, S_, V_ = np.linalg.svd(reconstr_m, full_matrices=False)

    return [U_, S_, V_]
