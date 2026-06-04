from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.decomposition.matrix_decomposition.decomposer import MatrixDecomposer


def test_knee_point_spectrum_regularization_accepts_single_spectrum_argument():
    decomposer = MatrixDecomposer({
        'decomposition_params': {'spectrum_regularization': 'knee_point'},
    })
    spectrum = np.array([5.0, 4.5, 1.0, 0.3, 0.1])

    selected = decomposer.spectrum_regularization(spectrum, reg_type='knee_point')

    assert selected
    assert all(isinstance(index, int) for index in selected)
    assert all(0 <= index < len(spectrum) for index in selected)


def test_matrix_decomposer_apply_supports_knee_point_rank_regularization():
    decomposer = MatrixDecomposer({
        'decomposition_type': 'svd',
        'decomposition_params': {'spectrum_regularization': 'knee_point'},
    })
    tensor = np.diag(np.array([5.0, 2.0, 0.2, 0.1]))

    result = decomposer.apply(tensor)

    assert 1 <= result['rank'] <= len(result['spectrum'])
