import json
import os
from typing import Tuple, Callable, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
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

        for category in data["categories"]:
            self.classes[category["id"]] = category["name"]

        samples = {}
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

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns a sample from a dataset.

        Args:
            idx: Index of sample.

        Returns:
            A tuple ``(image, targets)``, where image is image tensor,
                and targets is dict with keys: ``'boxes'``, ``'labels'``,
                ``'image_id'``, ``'area'``, ``'iscrowd'``.
        """
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

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.samples)
