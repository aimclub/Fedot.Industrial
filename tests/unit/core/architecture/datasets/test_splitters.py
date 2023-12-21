import os

import numpy as np
import pytest
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from fedot_ind.core.architecture.datasets.splitters import k_fold, split_data, undersampling, dataset_info, get_dataset_mean_std, train_test_split
from fedot_ind.api.utils.path_lib import PROJECT_PATH

DATASETS_PATH = os.path.abspath(PROJECT_PATH + '/tests/data/datasets')


@pytest.fixture()
def dataset():
    path = os.path.join(DATASETS_PATH, 'Agricultural/train')

    yield ImageFolder(root=path, transform=ToTensor())


def test_train_test_split(dataset):
    train_ds, test_ds = train_test_split(dataset, p=0.2)
    assert len(train_ds) + len(test_ds) == len(dataset)


def test_split_data(dataset):
    fold_indices = split_data(dataset, n=3, verbose=True)
    assert np.array_equal(np.sort(np.concatenate(fold_indices)), np.arange(len(dataset)))
    assert fold_indices[0].size == 21
    assert fold_indices[1].size == 20
    assert fold_indices[2].size == 20


def test_k_fold(dataset):
    for train_ds, val_ds in k_fold(dataset, 3):
        assert len(train_ds) + len(val_ds) == len(dataset)


def test_undersampling(dataset):
    balanced = undersampling(dataset=dataset, n=3, verbose=True)
    assert len(balanced) == 9


def test_dataset_info(dataset):
    result = dataset_info(dataset=dataset, verbose=True)
    assert isinstance(result, dict)


def test_get_dataset_mean_std(dataset):
    mean, std = get_dataset_mean_std(dataset=dataset)
    assert isinstance(mean, tuple)
    assert isinstance(std, tuple)
    assert len(mean) == 3
    assert len(std) == 3