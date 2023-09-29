import numpy as np
import pytest

from fedot_ind.core.operation.decomposition.spectrum_decomposition import SpectrumDecomposer

WINDOW_LENGTH = 10
TS_LENGTH = 100

@pytest.fixture()
def basic_spectral_data():
    x0 = 1 * np.ones(TS_LENGTH) + np.random.rand(TS_LENGTH) * 1
    x1 = 3 * np.ones(TS_LENGTH) + np.random.rand(TS_LENGTH) * 2
    x2 = 5 * np.ones(TS_LENGTH) + np.random.rand(TS_LENGTH) * 1.5
    x = np.hstack([x0, x1, x2])
    x += np.random.rand(x.size)

    return x


def test_SpectrDecompose_property(basic_spectral_data):
    spectral = SpectrumDecomposer(data=basic_spectral_data,
                                  ts_length=TS_LENGTH)
    assert spectral.ts_length is not None
