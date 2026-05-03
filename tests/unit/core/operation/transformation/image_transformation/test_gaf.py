import pytest
import torch
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from pyts.image import GramianAngularField
from fedot_ind.core.operation.transformation.torch_backend.image_transformation.methods.gaf_transformation import GAF


@pytest.fixture
def data():
    return np.random.rand(100, 100)


def test_gaf(data):
    gaf_np = GramianAngularField(method='s', overlapping=True, image_size=0.7)
    res_np = gaf_np.fit_transform(data)
    gaf_torch = GAF({"method": 'summation', "overlapping": True, "image_size":0.7})
    res_torch = gaf_torch.transform(torch.tensor(data, dtype=torch.float64))
    print(res_np.shape, res_torch.shape)
    res_torch_np = np.asarray(res_torch)
    rmse = np.power((res_np - res_torch_np), 2).mean() ** 0.5
    print(rmse)

if __name__ == "__main__":
    test_gaf(np.random.rand(100, 100))
