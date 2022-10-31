import json
import os
from typing import Tuple, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm


def collate_fn(batch):
    return tuple(zip(*batch))


class COCODataset(Dataset):
    """Class-loader for COCO json."""

    def __init__(
        self,
        datasets_folder: str,
        image_folder: str,
        json_file: str,
        transform: Callable,
    ) -> None:
        self.transform = transform
        self.samples = []
        self.classes = {}
        path_to_image_folder = os.path.join(datasets_folder, image_folder)
        path_to_json_file = os.path.join(datasets_folder, json_file)
        self._read_json(path_to_image_folder, path_to_json_file)

    def __getitem__(self, idx):
        """Get item method. Return list of images and list of targets."""
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        image = self.transform(image)
        target = {
            "boxes": torch.tensor(np.stack(sample["boxes"]), dtype=torch.float32),
            "labels": torch.tensor(sample["labels"], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(sample["area"], dtype=torch.float32),
            "iscrowd": torch.tensor(sample["iscrowd"], dtype=torch.int64),
        }
        return image, target

    def __len__(self):
        """Return length of dataset"""
        return len(self.samples)

    def _read_json(self, images_path: str, json_path: str) -> None:
        """Read annotations from json file"""

        samples = {}
        with open(json_path) as f:
            data = json.load(f)

        for category in data["categories"]:
            self.classes[category["id"]] = category["name"]

        for image in data["images"]:
            samples[image["id"]] = {
                "image": os.path.join(images_path, image["file_name"]),
                "area": [],
                "iscrowd": [],
                "labels": [],
                "boxes": [],
            }

        for annotation in tqdm(data["annotations"]):
            if annotation["area"] > 0:
                bbox = np.array(annotation["bbox"])
                bbox[2:] += bbox[:2]  # x, y, w, h -> x1, y1, x2, y2
                samples[annotation["image_id"]]["labels"].append(
                    annotation["category_id"]
                )
                samples[annotation["image_id"]]["boxes"].append(bbox)
                samples[annotation["image_id"]]["area"].append(annotation["area"])
                samples[annotation["image_id"]]["iscrowd"].append(annotation["iscrowd"])

        for sample in samples.values():
            if len(sample["labels"]) > 0:
                self.samples.append(sample)


DATASETS_PARAMETERS = {
    "ALET": {
        "getter": COCODataset,
        "num_classes": 50,
        "train": {
            "image_folder": "ALET/trainv4",
            "json_file": "ALET/trainv4.json",
            "transform": ToTensor(),
        },
        "val": {
            "image_folder": "ALET/valv4",
            "json_file": "ALET/valv4.json",
            "transform": ToTensor(),
        },
        "test": {
            "image_folder": "ALET/testv4",
            "json_file": "ALET/testv4.json",
            "transform": ToTensor(),
        },
    }
}


def get_detection_dataloaders(
    dataset_name: str,
    datasets_folder: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, int]:
    """Get dataloaders and return train and validation dataloaders, number of classes."""
    if dataset_name not in DATASETS_PARAMETERS.keys():
        raise ValueError(
            "dataset_name must be one of {}, but got dataset_name='{}'".format(
                DATASETS_PARAMETERS.keys(), dataset_name
            )
        )
    params = DATASETS_PARAMETERS[dataset_name]
    train_dataloader = DataLoader(
        dataset=params["getter"](datasets_folder=datasets_folder, **params["train"]),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=params["getter"](datasets_folder=datasets_folder, **params["val"]),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader, params["num_classes"]
