import os

import numpy as np
import pytest
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from core.architecture.datasets.splitters import k_fold, split_data
from core.architecture.utils.utils import PROJECT_PATH

DATASETS_PATH = os.path.join(PROJECT_PATH, 'tests/data/datasets/')


@pytest.fixture()
def dataset():
    yield ImageFolder(root=DATASETS_PATH + 'Agricultural/train', transform=ToTensor())


def test_split_data(dataset):
    fold_indices = split_data(dataset, n=3)
    assert np.array_equal(np.sort(np.concatenate(fold_indices)), np.arange(len(dataset)))
    assert fold_indices[0].size == 21
    assert fold_indices[1].size == 20
    assert fold_indices[2].size == 20


def test_k_fold(dataset):
    for train_ds, val_ds in k_fold(dataset, 3):
        assert len(train_ds) + len(val_ds) == len(dataset)