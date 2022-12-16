from core.operation.decomposition.SpectrumDecomposition import *
import pytest


@pytest.fixture()
def basic_spectral_data():
    x0 = 1 * np.ones(100) + np.random.rand(100) * 1
    x1 = 3 * np.ones(100) + np.random.rand(100) * 2
    x2 = 5 * np.ones(100) + np.random.rand(100) * 1.5
    x = np.hstack([x0, x1, x2])
    x += np.random.rand(x.size)

    return x


def test_SpectrDecompose_property(basic_spectral_data):
    spectral = SpectrumDecomposer(time_series=basic_spectral_data)
    assert spectral.ts_length is not None
    assert spectral.window_length is not None
    assert spectral.trajectory_matrix is not None


def test_SpectrDecompose_methods(basic_spectral_data):
    spectral = SpectrumDecomposer(time_series=basic_spectral_data,
                                  save_memory=False)
    TS_comps, X_elem, V, components_df, n_components, explained_dispersion = spectral.decompose()
    combined_components = spectral.combine_eigenvectors(TS_comps,rank=n_components)
    assert V.shape[0] == spectral.sub_seq_length
    assert TS_comps.shape[0] == spectral.ts_length
    assert X_elem.shape[1] == spectral.window_length
    assert n_components == components_df.shape[1]
    assert explained_dispersion > 0
    assert combined_components is not None
