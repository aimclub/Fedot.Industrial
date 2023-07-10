import math

import numpy as np
from scipy.linalg import qr

from fedot_ind.core.operation.transformation.regularization.spectrum import singular_value_hard_threshold


def bksvd(tensor, k=6, block_size=None, num_iter=3) -> tuple:
    """Block Krylov SVD

    Args:
        tensor (array (M, N) array_like):
        k (int): rank of the decomposition
        block_size (int): size of the block
        num_iter (int): number of iterations

    Returns:
        u, s, vt (array_like): decomposition

    Notes:
        This function is taken from https://github.com/whistlebee/bksvd-py/tree/master

    """
    import math

    k = math.ceil(min(tensor.shape) / 3)
    # q = math.ceil(math.log(tensor.shape[1], 10)/0.1)
    q = 3
    # if k == 'full':
    #     k = min(tensor.shape)

    # if block_size is None:
    #     block_size = k
    block_size = k

    # block_size = 2

    # k = min(k, min(tensor.shape))
    # u = np.zeros((1, tensor.shape[1]))
    #
    # l = np.ones((tensor.shape[0], 1))

    # K = np.zeros((tensor.shape[0], block_size * num_iter))
    # K = np.zeros((tensor.shape[1], block_size * num_iter))
    _block = np.random.randn(tensor.shape[1], block_size)

    AAT = tensor @ tensor.T
    block = np.linalg.matrix_power(AAT, q) @ tensor @ _block

    # block, _ = qr(block, mode='full')
    block, _ = qr(block, mode='economic')

    M = block.T @ AAT @ block
    Ut, St, Vt = np.linalg.svd(M, full_matrices=False)
    # U, S, V = np.linalg.svd(tensor, full_matrices=False)
    low_rank = len(singular_value_hard_threshold(St, beta=tensor.shape[0] / tensor.shape[1]))

    Ut_ = Ut[:, :low_rank]
    tensor_approx = block @ Ut_
    reconstr_m = tensor_approx @ tensor_approx.T @ tensor
    # diff_norm = np.linalg.norm(reconstr_m) - np.linalg.norm(tensor)

    U_, S_, V_ = np.linalg.svd(reconstr_m, full_matrices=False)

    return U_, S_, V_
    # _ = 1

    # _, SS, _ = np.linalg.svd(block @ Ut @ (block @ Ut).T @ tensor)
    # import matplotlib.pyplot as plt
    #
    # plt.plot(SS, label='SS')
    # plt.plot(S, label='S')
    # plt.legend()
    # plt.show()
    #
    # diff_norm = np.linalg.norm(block @ Ut @ (block @ Ut).T @ tensor) - np.linalg.norm(tensor)

    # T = np.zeros((tensor.shape[1], block_size))
    #
    # for i in range(num_iter):
    #     T = tensor @ block - l * (u @ block)
    #     block = tensor.T @ T - (u.T * (l.T @ T))
    #     block, _ = qr(block, mode='economic')
    #     K[:, i * block_size: (i + 1) * block_size] = block
    # Q, _ = qr(K, mode='economic')
    #
    # # Rayleigh-Ritz
    # T = tensor @ Q - l @ (u @ Q)
    #
    # Ut, St, Vt = np.linalg.svd(T, full_matrices=False)
    # U = Ut[:, :k]
    # S = St[:k]
    # V = Q @ Vt.T
