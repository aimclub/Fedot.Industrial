from os import path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageNet
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

DATASETS_PARAMETERS = {
    "CIFAR100": {
        "getter": CIFAR100,
        "num_classes": 100,
        "train": {
            "train": True,
            "download": True,
            "transform": Compose(
                [
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
        "val": {
            "train": False,
            "download": False,
            "transform": Compose(
                [
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
    },
    "CIFAR10": {
        "getter": CIFAR10,
        "num_classes": 10,
        "train": {
            "train": True,
            "download": True,
            "transform": Compose(
                [
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
        "val": {
            "train": False,
            "download": False,
            "transform": Compose(
                [
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
    },
    "MNIST": {
        "getter": MNIST,
        "num_classes": 10,
        "train": {
            "train": True,
            "download": True,
            "transform": Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                ]
            ),
        },
        "val": {
            "train": False,
            "download": False,
            "transform": Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                ]
            ),
        },
    },
    "ImageNet": {
        "getter": ImageNet,
        "num_classes": 1000,
        "train": {
            "split": "train",
            "transform": Compose(
                [
                    Resize((256, 265)),
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        },
        "val": {
            "split": "val",
            "transform": Compose(
                [
                    Resize((256, 265)),
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        },
    },
}


def get_classification_dataloaders(
    dataset_name: str,
    datasets_folder: str,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Get dataloaders.

    Args:
        dataset_name: ``'MNIST'``, ``'CIFAR10'``, ``'CIFAR100'`` or ``'ImageNet'``.
        datasets_folder: Path to folder with datasets.
        batch_size: How many samples per batch to load (default: ``1``).
        num_workers: How many subprocesses to use for data loading. ``0`` means that
            the data will be loaded in the main process. (default: ``0``)

    Returns:
        A tuple ``(train_dataloader, test_dataloader, num_classes)``

    Raises:
        ValueError: If ``dataset_name`` not in valid values.
    """
    if dataset_name not in DATASETS_PARAMETERS.keys():
        raise ValueError(
            "dataset_name must be one of {}, but got dataset_name='{}'".format(
                DATASETS_PARAMETERS.keys(), dataset_name
            )
        )
    params = DATASETS_PARAMETERS[dataset_name]

    train_dataloader = DataLoader(
        dataset=params["getter"](
            root=path.join(datasets_folder, dataset_name), **params["train"]
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=params["getter"](
            root=path.join(datasets_folder, dataset_name), **params["val"]
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return train_dataloader, test_dataloader, params["num_classes"]
