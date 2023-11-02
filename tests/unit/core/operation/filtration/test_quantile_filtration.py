import numpy as np

from fedot_ind.core.operation.filtration.quantile_filtration import quantile_filter


def test_quantile_filter():
    input_data = np.random.rand(10, 10)
    predicted_data = np.random.rand(10, 10)
    result = quantile_filter(input_data=input_data, predicted_data=predicted_data)
    assert isinstance(result[0], np.int64)
