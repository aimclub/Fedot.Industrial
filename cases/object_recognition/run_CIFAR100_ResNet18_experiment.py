import torch
from torchvision.models import resnet18

from utils import run_experiment, get_CIFAR100_dataloaders


if __name__ == "__main__":
    run_experiment(
        dataloaders=[
            get_CIFAR100_dataloaders("/home/storage/datasets/CIFAR100_dataset")
        ],
        models=[("ResNet18", resnet18)],
        coefficients=[0.01, 0.005, 0.001, 0.0005],
        train_parameters={
            "loss_fn": torch.nn.CrossEntropyLoss,
            "optimizer": torch.optim.Adam,
            "learning_rate": 0.001,
            "num_epochs": 5,
            "progress": True,
        },
    )
