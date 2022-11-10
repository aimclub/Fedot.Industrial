import json
import os
from typing import Tuple, Callable, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def collate_fn(batch):
    return tuple(zip(*batch))


class COCODataset(Dataset):
    """Class-loader for COCO json.

    Args:
        images_path: Image folder path.
        json_path: Json file path.
        transform: A function/transform that takes in an PIL image and returns a
            transformed version.
    """

    def __init__(
        self,
        images_path: str,
        json_path: str,
        transform: Callable,
    ) -> None:
        self.transform = transform
        self.classes = {}
        self.samples = []

        with open(json_path) as f:
            data = json.load(f)

        for category in data['categories']:
            self.classes[category['id']] = category['name']

        samples = {}
        for image in data['images']:
            samples[image['id']] = {
                'image': os.path.join(images_path, image['file_name']),
                'area': [],
                'iscrowd': [],
                'labels': [],
                'boxes': [],
            }

        for annotation in tqdm(data['annotations']):
            if annotation['area'] > 0:
                bbox = np.array(annotation['bbox'])
                bbox[2:] += bbox[:2]  # x, y, w, h -> x1, y1, x2, y2
                samples[annotation['image_id']]['labels'].append(
                    annotation['category_id']
                )
                samples[annotation['image_id']]['boxes'].append(bbox)
                samples[annotation['image_id']]['area'].append(annotation['area'])
                samples[annotation['image_id']]['iscrowd'].append(annotation['iscrowd'])

        for sample in samples.values():
            if len(sample['labels']) > 0:
                self.samples.append(sample)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns list of images and list of targets.

        Args:
            idx: Index of sample.

        Returns:
            A tuple ``(image, targets)``, where image is image tensor,
                and targets is dict with keys: ``'boxes'``, ``'labels'``,
                ``'image_id'``, ``'area'``, ``'iscrowd'``.
        """
        sample = self.samples[idx]
        image = Image.open(sample['image']).convert('RGB')
        image = self.transform(image)
        target = {
            'boxes': torch.tensor(np.stack(sample['boxes']), dtype=torch.float32),
            'labels': torch.tensor(sample['labels'], dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor(sample['area'], dtype=torch.float32),
            'iscrowd': torch.tensor(sample['iscrowd'], dtype=torch.int64),
        }
        return image, target

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.samples)


def get_detection_dataloaders(
    dataset_name: str,
    datasets_folder: str,
    train_transforms: Callable,
    val_transforms: Callable,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Get dataloaders.

    Args:
        dataset_name: The dataset name must match the folder name.
        datasets_folder: Path to folder with datasets.
        train_transforms: A transformation applied to train images that takes in an PIL
            image and returns a transformed version.
        val_transforms: A transformation applied to validation images that takes in an
            PIL image and returns a transformed version.
        batch_size: How many samples per batch to load (default: ``1``).
        num_workers: How many subprocesses to use for data loading. ``0`` means that
            the data will be loaded in the main process. (default: ``0``)

    Returns:
        A tuple (train_dataloader, test_dataloader, num_classes)
    """
    train_dataloader = DataLoader(
        dataset=COCODataset(
            images_path=os.path.join(datasets_folder, dataset_name, 'train'),
            json_path=os.path.join(datasets_folder, dataset_name, 'train.json'),
            transform=train_transforms,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=COCODataset(
            images_path=os.path.join(datasets_folder, dataset_name, 'val'),
            json_path=os.path.join(datasets_folder, dataset_name, 'val.json'),
            transform=val_transforms,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    num_classes = len(train_dataloader.dataset.classes) + 1
    return train_dataloader, val_dataloader, num_classes
