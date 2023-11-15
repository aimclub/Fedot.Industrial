import os

import pytest
from torchvision import transforms

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.architecture.datasets.object_detection_datasets import COCODataset, YOLODataset

yolo_path = os.path.join(PROJECT_PATH, 'tests', 'data', 'datasets', 'minerals', 'minerals.yaml')
coco_path = os.path.join(PROJECT_PATH, 'tests', 'data', 'datasets', 'ALET10', 'test.json')
coco_img_path = os.path.join(PROJECT_PATH, 'tests', 'data', 'datasets', 'ALET10', 'test')


@pytest.fixture
def synthetic_coco_dataset():
    return COCODataset(coco_img_path, coco_path, transform=transforms.ToTensor())


@pytest.fixture
def yolo_dataset():
    return YOLODataset(yolo_path, transform=transforms.ToTensor())


def test_coco_dataset_sample(synthetic_coco_dataset):
    sample = synthetic_coco_dataset[0]
    image, label = sample
    assert len(synthetic_coco_dataset) == 3
    assert image is not None
    assert label is not None


def test_yolo_dataset_sample(yolo_dataset):
    sample = yolo_dataset[0]
    image, label = sample
    assert len(yolo_dataset) == 47
    assert image is not None
    assert label is not None
