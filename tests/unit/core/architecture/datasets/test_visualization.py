import os

import pytest
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.architecture.datasets.object_detection_datasets import COCODataset
from fedot_ind.core.architecture.datasets.visualization import draw_sample_with_bboxes, draw_sample_with_masks

coco_path = os.path.join(PROJECT_PATH, 'tests', 'data', 'datasets', 'ALET10', 'test.json')
coco_img_path = os.path.join(PROJECT_PATH, 'tests', 'data', 'datasets', 'ALET10', 'test')


@pytest.fixture
def synthetic_coco_dataset():
    return COCODataset(coco_img_path, coco_path, transform=transforms.ToTensor())


def test_draw_sample_with_bboxes(synthetic_coco_dataset):
    sample = synthetic_coco_dataset[0]
    image, label = sample
    figure = draw_sample_with_bboxes(image=image, target=label)

    assert isinstance(figure, plt.Figure)


def test_draw_sample_with_masks(synthetic_coco_dataset):
    sample = synthetic_coco_dataset[0]
    image, _ = sample
    figure = draw_sample_with_masks(image=image, target=image)

    assert isinstance(figure, plt.Figure)
