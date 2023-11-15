import os

import numpy as np
import pytest
from torch import Tensor
from torchvision.transforms import transforms

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.architecture.datasets.prediction_datasets import PredictionFolderDataset, PredictionNumpyDataset

IMAGE_SIZE = (50, 50)
N_SAMPLES = 5
yolo_path = os.path.join(PROJECT_PATH, 'tests', 'data', 'datasets', 'ALET10', 'test')


@pytest.fixture
def images():
    return np.array([np.random.randint(low=0,
                                       high=255,
                                       size=IMAGE_SIZE) for _ in range(N_SAMPLES)])


def test_prediction_numpy_dataset(images):
    dataset = PredictionNumpyDataset(images=images)
    sample = dataset[0]
    assert len(dataset) == N_SAMPLES
    assert isinstance(sample, tuple)
    assert isinstance(sample[0], Tensor)


def test_prediction_folder_dataset():
    dataset = PredictionFolderDataset(image_folder=yolo_path,
                                      transform=transforms.ToTensor())
    tensor, filename = dataset[0]
    assert len(dataset) == 3
    assert isinstance(tensor, Tensor)
    assert isinstance(filename, str)
