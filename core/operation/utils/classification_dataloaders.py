import os
from typing import Tuple, Dict

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageNet, ImageFolder

TYPICAL_DATASETS = {
    'CIFAR100':  {'getter': CIFAR100, 'num_classes': 100},
    'CIFAR10': {'getter': CIFAR10, 'num_classes': 10},
    'MNIST':  {'getter': MNIST, 'num_classes': 10},
    'ImageNet': {'getter': ImageNet, 'num_classes': 1000},
}


def get_classification_dataloaders(
    dataset_name: str,
    datasets_folder: str,
    train_ds_params: Dict,
    val_ds_params: Dict,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Get dataloaders using ImageFolder.

    Args:
        dataset_name: The dataset name must match the folder name. May be: ``'MNIST'``,
            ``'CIFAR10'``, ``'CIFAR100'``, ``'ImageNet'`` or custom folder passed to
            ``ImageFolder``.
        datasets_folder: Path to folder with datasets.
        train_ds_params:  Parameter dictionary passed to train dataset initialization.
        val_ds_params:  Parameter dictionary passed to validation dataset initialization.
        batch_size: How many samples per batch to load (default: ``1``).
        num_workers: How many subprocesses to use for data loading. ``0`` means that
            the data will be loaded in the main process. (default: ``0``)

    Returns:
        A tuple ``(train_dataloader, test_dataloader, num_classes)``
    """
    root = os.path.join(datasets_folder, dataset_name)
    if dataset_name in TYPICAL_DATASETS.keys():
        getter = TYPICAL_DATASETS[dataset_name]['getter']
        num_classes = TYPICAL_DATASETS[dataset_name]['num_classes']
        train_dataset = getter(root=root, **train_ds_params)
        val_dataset = getter(root=root, **val_ds_params)
    else:
        train_root = os.path.join(root, 'train')
        num_classes = len(os.listdir(train_root))
        train_dataset = ImageFolder(root=train_root, **train_ds_params)
        val_dataset = ImageFolder(root=os.path.join(root, 'val'), **val_ds_params)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return train_dataloader, test_dataloader, num_classes
