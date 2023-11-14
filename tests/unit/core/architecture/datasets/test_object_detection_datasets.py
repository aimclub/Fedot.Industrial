import json
from pathlib import Path

import pytest
import yaml
from torchvision import transforms

from fedot_ind.core.architecture.datasets.object_detection_datasets import COCODataset, YOLODataset

synthetic_coco_data = {
    "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
    "images": [
        {"id": 1, "file_name": "image1.jpg"},
        {"id": 2, "file_name": "image2.jpg"}
    ],
    "annotations": [
        {"image_id": 1, "category_id": 1, "area": 100, "bbox": [10, 20, 30, 40], "iscrowd": 0},
        {"image_id": 2, "category_id": 2, "area": 150, "bbox": [15, 25, 35, 45], "iscrowd": 1}
    ]
}


synthetic_yolo_data = {
    "train": "train/images",
    "val": "val/images",
    "names": ["cat", "dog"]
}


@pytest.fixture
def synthetic_coco_dataset():
    tmp_path = Path('.')
    coco_json_path = tmp_path / "synthetic_coco.json"
    coco_json_path.write_text(json.dumps(synthetic_coco_data))
    images_path = tmp_path / "images"
    images_path.mkdir(exist_ok=True)
    (images_path / "image1.jpg").write_text("")
    (images_path / "image2.jpg").write_text("")
    return COCODataset(str(images_path), str(coco_json_path), transform=transforms.ToTensor())


@pytest.fixture
def synthetic_yolo_dataset():
    tmp_path = Path('.')
    yolo_yaml_path = tmp_path / "synthetic_yolo.yaml"
    yolo_yaml_path.write_text(yaml.dump(synthetic_yolo_data))
    root_path = tmp_path / "train" / "images"
    root_path.mkdir(exist_ok=True, parents=True)
    (root_path / "image1.jpg").write_text("")  # Create empty files for images
    (root_path / "image2.jpg").write_text("")
    return YOLODataset(str(yolo_yaml_path), transform=transforms.ToTensor())


def test_coco_dataset_length(synthetic_coco_dataset):
    assert len(synthetic_coco_dataset) == 2


def test_coco_dataset_sample(synthetic_coco_dataset):
    sample = synthetic_coco_dataset.samples[0]
    image, label = sample['image'], sample['labels']
    assert image is not None
    assert label is not None


def test_yolo_dataset_length(synthetic_yolo_dataset):
    assert len(synthetic_yolo_dataset) == 2


def test_yolo_dataset_sample(synthetic_yolo_dataset):
    sample = synthetic_yolo_dataset.samples[0]
    image, label = sample
    assert image is not None
    assert label is not None
