import torch

from core.models.cnn.classification_models import SimpleConvNet2
from utils import run_experiment, get_MNIST_dataloaders


if __name__ == "__main__":
    run_experiment(
        dataloaders=[get_MNIST_dataloaders("/home/storage/datasets/MNIST")],
        models=[("SimpleConvNet2", SimpleConvNet2)],
        coefficients=[0.01, 0.005, 0.001, 0.0005],
        train_parameters={
            "loss_fn": torch.nn.CrossEntropyLoss,
            "optimizer": torch.optim.Adam,
            "learning_rate": 0.001,
            "num_epochs": 20,
            "progress": True,
        },
    )
