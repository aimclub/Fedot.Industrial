import copy
import time

import torch
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from core.operation.utils.pruning_tools import decompose_module, prune_model, EnergyThresholdPruning
from core.metrics.svd_loss import OrthogonalLoss, HoyerLoss


def train_loop(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.nn.Module,
    model_losses: List[Tuple[torch.nn.Module, float]] = [],
    progress: bool = True,
) -> (float, List[float]):

    batches = tqdm(dataloader) if progress else dataloader
    train_loss = 0
    additional_losses = torch.zeros(len(model_losses))
    for i, (x, y) in enumerate(batches):
        x = x.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        for j, (model_loss, coefficient) in enumerate(model_losses):
            m_loss = model_loss(model)
            additional_losses[j] += m_loss.item()
            loss += coefficient * m_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if progress:
            batches.set_postfix({"train loss": train_loss / (i + 1)})
    return train_loss / len(dataloader), additional_losses / len(dataloader)


def test_loop(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    progress: bool = True,
) -> (float, float, float):

    batches = tqdm(dataloader) if progress else dataloader
    size = len(dataloader.dataset)
    test_loss = 0
    accuracy = 0

    start = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(batches):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

            if progress:
                batches.set_postfix({"val loss": test_loss / (i + 1)})
    total_time = time.time() - start
    return test_loss / len(dataloader), accuracy / size, total_time / len(dataloader)


def train(
    exp_description: str,
    num_epochs: int,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.nn.Module,
    learning_rate: float,
    hoyer_loss_coef: Optional[float] = None,
    decomposing_mode: Optional[str] = None,
    progress: bool = True,
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("runs/" + exp_description)
    loss_fn = loss_fn()
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    if hoyer_loss_coef is not None:
        model_losses = [(OrthogonalLoss(device), 1), (HoyerLoss(), hoyer_loss_coef)]
    else:
        model_losses = []

    print("Train {}, using device: {}".format(exp_description, device))
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        if progress:
            print("Epoch {}".format(epoch))
        train_loss, additional_losses = train_loop(
            model=model,
            device=device,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            model_losses=model_losses,
            progress=progress,
        )
        test_loss, accuracy, _ = test_loop(
            model=model,
            device=device,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            progress=progress,
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/accuracy", accuracy, epoch)
        if len(model_losses) > 0:
            writer.add_scalar("train/orthogonal loss", additional_losses[0], epoch)
            writer.add_scalar("train/hoyer loss", additional_losses[1], epoch)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    if decomposing_mode in {"channel", "spatial"}:
        auto_pruning(
            model=model,
            device=device,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            writer=writer,
        )


def run_experiment(
    dataloaders: List[Tuple[str, DataLoader, DataLoader]],
    models: List[Tuple[str, torch.nn.Module]],
    coefficients: List[float],
    train_parameters: Dict,
) -> None:

    for dataset_name, train_dl, test_dl in dataloaders:
        for model_name, model in models:
            description = dataset_name + "/" + model_name
            tmp_model = model()
            train(
                exp_description=description,
                model=tmp_model,
                train_dataloader=train_dl,
                test_dataloader=test_dl,
                **train_parameters
            )
            for decomposing_mode in ["channel", "spatial"]:
                dec_description = description + "_" + decomposing_mode
                tmp_model = model()
                decompose_module(tmp_model, decomposing_mode)
                train(
                    exp_description=dec_description,
                    model=tmp_model,
                    train_dataloader=train_dl,
                    test_dataloader=test_dl,
                    decomposing_mode=decomposing_mode,
                    **train_parameters
                )

                for coef in coefficients:
                    tmp_model = model()
                    decompose_module(tmp_model, decomposing_mode)
                    train(
                        exp_description=dec_description + "_{:.4f}".format(coef),
                        model=tmp_model,
                        train_dataloader=train_dl,
                        test_dataloader=test_dl,
                        hoyer_loss_coef=coef,
                        decomposing_mode=decomposing_mode,
                        **train_parameters
                    )


def auto_pruning(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    writer: SummaryWriter,
) -> None:

    _, default_accuracy, default_time = test_loop(
        dataloader=dataloader,
        model=model,
        device=device,
        loss_fn=loss_fn,
        progress=False,
    )

    for e in [0.1, 0.3, 0.5, 0.7, 0.9,
              0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
              0.992, 0.994, 0.996, 0.998, 0.999,
              0.9999, 1]:
        int_e = e * 10000
        new_model = copy.deepcopy(model)
        start = time.time()
        old_params, new_params = prune_model(new_model, EnergyThresholdPruning(e))
        pruning_time = time.time() - start

        _, accuracy, val_time = test_loop(
            dataloader=dataloader,
            model=new_model,
            device=device,
            loss_fn=loss_fn,
            progress=False,
        )
        size_percentage = new_params / old_params * 100
        time_percentage = val_time / default_time * 100
        accuracy_percentage = accuracy / default_accuracy * 100

        writer.add_scalar("e/accuracy %)", accuracy_percentage, int_e)
        writer.add_scalar("e/size %", size_percentage, int_e)
        writer.add_scalar("e/inference time %", time_percentage, int_e)
        writer.add_scalar("e/pruning time", pruning_time, int_e)
        writer.add_scalar("acc-compr/division)",
                          accuracy_percentage / size_percentage, int_e)
        writer.add_scalar("acc-compr/subtraction)",
                          accuracy_percentage - size_percentage, int_e)


def get_MNIST_dataloaders(
        ds_path: str,
        batch_size: int = 100
) -> Tuple[str, DataLoader, DataLoader]:
    mnist_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root=ds_path, train=True, transform=mnist_transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=ds_path, train=False, transform=mnist_transform
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return "MNIST", train_dataloader, test_dataloader


def get_CIFAR100_dataloaders(
        ds_path: str,
        batch_size: int = 100
) -> Tuple[str, DataLoader, DataLoader]:

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5074, 0.4867, 0.4411),
                (0.2011, 0.1987, 0.2025)
            ),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(
        root=ds_path, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=ds_path, train=False, transform=transform
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return "CIFAR100", train_dataloader, test_dataloader
