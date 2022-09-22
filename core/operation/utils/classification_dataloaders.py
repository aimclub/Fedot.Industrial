from typing import Tuple
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

DATASETS_PARAMETERS = {
    "CIFAR100": {
        "getter": CIFAR100,
        "num_classes": 100,
        "mean": (0.5074, 0.4867, 0.4411),
        "std": (0.2011, 0.1987, 0.2025)
    },
    "CIFAR10": {
        "getter": CIFAR10,
        "num_classes": 10,
        "mean": (0.5074, 0.4867, 0.4411),
        "std": (0.2011, 0.1987, 0.2025)
    },
    "MNIST": {
        "getter": MNIST,
        "num_classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,)
    }
}


def get_dataloaders(
        dataset_name: str,
        ds_path: str,
        batch_size: int
) -> Tuple[DataLoader, DataLoader, int]:

    if dataset_name not in DATASETS_PARAMETERS.keys():
        raise ValueError(
            "dataset_name must be one of {}, but got dataset_name='{}'".format(
                DATASETS_PARAMETERS.keys(), dataset_name
            )
        )
    params = DATASETS_PARAMETERS[dataset_name]

    transform = Compose([ToTensor(), Normalize(params["mean"], params["std"])])

    train_dataloader = DataLoader(
        dataset=params["getter"](
            root=ds_path,
            train=True,
            transform=transform,
            download=True
        ),
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=params["getter"](
            root=ds_path,
            train=False,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=False
    )
    return train_dataloader, test_dataloader, params["num_classes"]
