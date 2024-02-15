import numpy as np
import pytest

from fedot_ind.core.operation.transformation.regularization.spectrum import reconstruct_basis, \
    singular_value_hard_threshold, sv_to_explained_variance_ratio
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator


@pytest.fixture()
def matrix_from_ts():
    window = 30
    ts_config = {
        'ts_type': 'sin',
        'length': 300,
        'amplitude': 10,
        'period': 500
    }
    time_series = TimeSeriesGenerator(params=ts_config).get_ts()
    matrix = []
    for i in range(len(time_series) - window):
        matrix.append(time_series[i:i + window])
    return np.array(matrix)


@pytest.fixture()
def singular_values_rank_threshold_beta():
    return [0.1, 0.2, 0.3, 0.4, 0.5], 3, 0.5, 0.5


def test_sv_to_explained_variance_ratio(singular_values_rank_threshold_beta):
    singular_values, rank, _, _ = singular_values_rank_threshold_beta
    explained_variance, n_components = sv_to_explained_variance_ratio(
        singular_values, rank)
    assert 0 < explained_variance <= 100


def test_singular_value_hard_threshold(singular_values_rank_threshold_beta):
    singular_values, rank, beta, threshold = singular_values_rank_threshold_beta
    adjusted_sv = singular_value_hard_threshold(
        singular_values, rank, beta, threshold)
    assert len(adjusted_sv) == 3


def test_reconstruct_basis(matrix_from_ts):
    U, S, VT = np.linalg.svd(matrix_from_ts)
    reconstructed_basis = reconstruct_basis(U=U,
                                            Sigma=S,
                                            VT=VT,
                                            ts_length=299)
    assert isinstance(reconstructed_basis, np.ndarray)
    assert reconstructed_basis.shape == (299, 30)
