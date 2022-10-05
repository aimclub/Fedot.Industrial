from os import path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageNet
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

DATASETS_PARAMETERS = {
    "CIFAR100": {
        "getter": CIFAR100,
        "num_classes": 100,
        "transform": Compose(
            [
                ToTensor(),
                Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
            ]
        ),
        "download": True,
    },
    "CIFAR10": {
        "getter": CIFAR10,
        "num_classes": 10,
        "transform": Compose(
            [
                ToTensor(),
                Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
            ]
        ),
        "mean": (0.5074, 0.4867, 0.4411),
        "std": (0.2011, 0.1987, 0.2025),
        "download": True,
    },
    "MNIST": {
        "getter": MNIST,
        "num_classes": 10,
        "transform": Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        ),
        "download": True,
    },
    "ImageNet": {
        "getter": ImageNet,
        "num_classes": 1000,
        "transform": Compose(
            [
                Resize(256),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        "download": False,
    },
}


def get_dataloaders(
    dataset_name: str, datasets_folder: str, batch_size: int
) -> Tuple[DataLoader, DataLoader, int]:

    if dataset_name not in DATASETS_PARAMETERS.keys():
        raise ValueError(
            "dataset_name must be one of {}, but got dataset_name='{}'".format(
                DATASETS_PARAMETERS.keys(), dataset_name
            )
        )
    params = DATASETS_PARAMETERS[dataset_name]

    train_dataloader = DataLoader(
        dataset=params["getter"](
            root=path.join(datasets_folder, dataset_name),
            train=True,
            transform=params["transform"],
            download=params["download"],
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=params["getter"](
            root=path.join(datasets_folder, dataset_name),
            train=False,
            transform=params["transform"],
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_dataloader, test_dataloader, params["num_classes"]
