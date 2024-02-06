import numpy as np
from tensorly import tensor

from fedot_ind.core.operation.decomposition.spectrum_decomposition import SpectrumDecomposer, supported_types


class TestSpectrumDecomposer:

    def test_constructor_valid_types(self):
        for data_type in supported_types:
            data = np.random.rand(10, 5) if data_type == np.ndarray else [
                np.random.rand(10, 5)]
            ts_length = 10
            decomposer = SpectrumDecomposer(data, ts_length)
            assert decomposer.ts_length == ts_length

    def test_svd(self):
        data = np.random.rand(10, 5)
        decomposer = SpectrumDecomposer(data, 10)
        result = decomposer.svd(data)
        assert isinstance(result.value, list)
        assert len(result.value[0]) == 3

    def test_threshold(self):
        data = np.random.rand(10, 5)
        decomposer = SpectrumDecomposer(data, 10)
        result = decomposer.svd(data)
        result = decomposer.threshold(result.value[0])
        assert isinstance(result.value, list)
        assert len(result.value[0]) == 3
        assert result.value[0][1].shape[0] <= decomposer.thr

    def test_multi_threshold(self):
        data = np.random.rand(10, 5)
        decomposer = SpectrumDecomposer(data, 10)
        result = decomposer.svd(data)
        decomposer.beta = round(
            result.value[0][0].shape[0] / result.value[0][0].shape[1])
        result = decomposer.multi_threshold(result.value[0])
        assert isinstance(result.value, list)
        assert len(result.value[0]) == 3

    def test_data_driven_basis(self):
        data = np.random.rand(10, 5)
        decomposer = SpectrumDecomposer(data, 10)
        result = decomposer.svd(data)
        decomposer.ts_length = 14
        result = decomposer.data_driven_basis(result.value[0])
        assert isinstance(result.value[0], np.ndarray)

    def test_tensor_decomposition(self):
        data = tensor(np.random.rand(10, 5, 3))
        decomposer = SpectrumDecomposer(list(data), 10)
        decomposer.rank = 2
        result = decomposer.tensor_decomposition(data)
        assert isinstance(result.value[0], list)
        assert len(result.value[0]) == 3
