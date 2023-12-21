import numpy as np
import pytest
from torch import Size, Tensor

from fedot_ind.core.architecture.datasets.classification_datasets import NumpyImageDataset

N_SAMPLES = 5
IMAGE_SIZE = (50, 50)


@pytest.fixture
def images_n_target():
    images = np.array([np.random.randint(low=0, high=255, size=IMAGE_SIZE) for _ in range(N_SAMPLES)])
    target = np.array([[np.random.randint(0, 3)] for _ in range(N_SAMPLES)])
    return images, target


def test_get_item(images_n_target):
    images, target = images_n_target
    dataset = NumpyImageDataset(images=images, targets=target)
    for sample_idx in range(N_SAMPLES + 1):
        if sample_idx == N_SAMPLES:
            with pytest.raises(IndexError) as execution_info:
                dataset.__getitem__(sample_idx)
            assert str(
                execution_info.value) == f'index {sample_idx} is out of bounds for dimension 0 with size {N_SAMPLES}'
        else:
            item = dataset.__getitem__(sample_idx)
            img, target_value = item[0], item[1]
            assert isinstance(item, tuple)
            assert isinstance(img, Tensor)
            assert isinstance(target_value, Tensor)
            assert np.all(np.array(img.size()) == IMAGE_SIZE)
            assert target_value.size() == Size((1,))
