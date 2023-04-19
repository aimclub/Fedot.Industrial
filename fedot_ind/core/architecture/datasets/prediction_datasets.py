"""This module contains classes for wrapping data of various types
for passing it to the prediction method of computer vision models.
"""

import os
from typing import Tuple, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class PredictionNumpyDataset(Dataset):
    """Class for prediction on numpy arrays.

    Args:
        images: Numpy matrix of images.
    """

    def __init__(
            self,
            images: np.ndarray,
    ) -> None:
        self.images = torch.from_numpy(images).float()

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """Returns a sample from a dataset.

        Args:
            idx: Index of sample.

        Returns:
            A tuple ``(image, id)``, where image is image tensor,
                and id is integer.
        """
        return self.images[idx], idx

    def __len__(self) -> int:
        """Return length of dataset"""
        return self.images.size()[0]


class PredictionFolderDataset(Dataset):
    """Class for prediction on images from folder.

    Args:
        image_folder: Path to image folder.
        transform: A function/transform that takes in an PIL image and returns a
            transformed version.
    """

    def __init__(
            self,
            image_folder: str,
            transform: Callable,
    ) -> None:
        self.root = image_folder
        self.images = []
        for address, dirs, files in os.walk(image_folder):
            for name in files:
                if name.lower().endswith(IMG_EXTENSIONS):
                    self.images.append(os.path.join(address, name))
        self.transform = transform

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        """Returns a sample from a dataset.

        Args:
            idx: Index of sample.

        Returns:
            A tuple ``(image, id)``, where image is image tensor,
                and id is file name.
        """

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        image = self.transform(image)
        return image, self.images[idx]

    def __len__(self) -> int:
        """Return length of dataset"""
        return len(self.images)
