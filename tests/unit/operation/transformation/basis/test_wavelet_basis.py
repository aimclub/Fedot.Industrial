import numpy as np
import pytest
import pywt
from fedot.core.data.data import OutputData

from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator

WAVELETS = ['mexh', 'shan', 'morl', 'cmor', 'fbsp', 'db5', 'sym5']
N_COMPONENTS = list(range(2, 12, 2))


def wavelet_components_combination():
    return [(w, c) for w in WAVELETS for c in N_COMPONENTS]


@pytest.fixture
def dataset():
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=20,
                                                                       max_ts_len=50,
                                                                       n_classes=2,
                                                                       test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


@pytest.fixture
def input_train(dataset):
    X_train, y_train, X_test, y_test = dataset
    input_train_data = init_input_data(X_train, y_train)
    return input_train_data


@pytest.mark.parametrize('wavelet, n_components', wavelet_components_combination())
def test_transform(input_train, wavelet, n_components):
    basis = WaveletBasisImplementation({"wavelet": wavelet,
                                        "n_components": n_components})
    train_features = basis.transform(input_data=input_train)
    assert isinstance(train_features, OutputData)
    assert train_features.features.shape[0] == input_train.features.shape[0]


@pytest.mark.parametrize('wavelet, n_components', wavelet_components_combination())
def test_decompose_signal(input_train, wavelet, n_components):
    basis = WaveletBasisImplementation({"wavelet": wavelet,
                                        "n_components": n_components})
    sample = input_train.features[0]
    transformed_sample = basis._decompose_signal(sample)
    assert isinstance(transformed_sample, tuple)
    assert len(transformed_sample) == 2


@pytest.mark.parametrize('wavelet, n_components', wavelet_components_combination())
def test_decomposing_level(input_train, wavelet, n_components):
    basis = WaveletBasisImplementation({"wavelet": wavelet,
                                        "n_components": n_components})
    sample = input_train.features[0]
    discrete_wavelets = pywt.wavelist(kind='discrete')
    basis.time_series = sample
    basis.wavelet = np.random.choice(discrete_wavelets)
    decomposing_level = basis._decomposing_level()
    assert isinstance(decomposing_level, int)


@pytest.mark.parametrize('wavelet, n_components', wavelet_components_combination())
def test_transform_one_sample(input_train, wavelet, n_components):
    basis = WaveletBasisImplementation({"wavelet": wavelet,
                                        "n_components": n_components})
    sample = input_train.features[0]
    transformed_sample = basis._transform_one_sample(sample)
    assert isinstance(transformed_sample, np.ndarray)


@pytest.mark.parametrize('wavelet, n_components', wavelet_components_combination())
def test_get_1d_bassis(input_train, wavelet, n_components):
    basis = WaveletBasisImplementation({"wavelet": wavelet,
                                        "n_components": n_components})
    sample = input_train.features[0]
    extracted_basis = basis._get_1d_basis(sample)
    assert isinstance(extracted_basis, np.ndarray)
