from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageNet
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


DATASETS_PARAMETERS = {
    "CIFAR100": {
        "getter": CIFAR100,
        "num_classes": 100,
        "train_ds_params": {
            "train": True,
            "download": True,
            "transform": Compose(
                [
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
        "val_ds_params": {
            "train": False,
            "download": False,
            "transform": Compose(
                [
                    Resize((32, 32)),
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
    },
    "CIFAR10": {
        "getter": CIFAR10,
        "num_classes": 10,
        "train_ds_params": {
            "train": True,
            "download": True,
            "transform": Compose(
                [
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
        "val_ds_params": {
            "train": False,
            "download": False,
            "transform": Compose(
                [
                    Resize((32, 32)),
                    ToTensor(),
                    Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
                ]
            ),
        },
    },
    "MNIST": {
        "getter": MNIST,
        "num_classes": 10,
        "train_ds_params": {
            "train": True,
            "download": True,
            "transform": Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                ]
            ),
        },
        "val_ds_params": {
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
        "train_ds_params": {
            "split": "train",
            "transform": Compose(
                [
                    Resize((256, 265)),
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        },
        "val_ds_params": {
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
