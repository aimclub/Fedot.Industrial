import torch
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from core.operation.utils.decomposing_tools import decompose_module, prune_model
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
) -> (float, float):

    batches = tqdm(dataloader) if progress else dataloader
    size = len(dataloader.dataset)
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(batches):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

            if progress:
                batches.set_postfix({"val loss": test_loss / (i + 1)})
    return test_loss / len(dataloader), accuracy / size


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
    for epoch in range(num_epochs):
        train_loss, additional_losses = train_loop(
            model=model,
            device=device,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            model_losses=model_losses,
            progress=progress,
        )
        test_loss, accuracy = test_loop(
            model=model,
            device=device,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            progress=progress,
        )

        writer.add_scalar("train loss", train_loss, epoch)
        writer.add_scalar("test loss", test_loss, epoch)
        writer.add_scalar("test accuracy", accuracy, epoch)
        if len(model_losses) > 0:
            writer.add_scalar("orthogonal loss", additional_losses[0], epoch)
            writer.add_scalar("hoyer loss", additional_losses[1], epoch)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    if decomposing_mode in {"channel", "spatial"}:
        auto_pruning(
            model=model,
            device=device,
            decomposing_mode=decomposing_mode,
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
    decomposing_mode: str,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    writer: SummaryWriter,
) -> None:

    for e10 in range(1, 11):
        e = e10 / 10
        pruned_model = type(model)()
        decompose_module(model=pruned_model, decomposing_mode=decomposing_mode)
        pruned_model.to(device)
        pruned_model.load_state_dict(model.state_dict())
        compression = prune_model(pruned_model, e) * 100
        test_loss, accuracy = test_loop(
            dataloader=dataloader,
            model=pruned_model,
            device=device,
            loss_fn=loss_fn,
            progress=False,
        )
        accuracy *= 100
        writer.add_scalar("accuracy(e)", accuracy, e10)
        writer.add_scalar("compression(e)", compression, e10)
        writer.add_scalar("acc/(100-compr)(e)", accuracy / (100 - compression), e10)
        writer.add_scalar("acc - (100-compr)(e)", accuracy - (100 - compression), e10)


def get_MNIST_dataloaders(batch_size: int = 100) -> Tuple[str, DataLoader, DataLoader]:
    mnist_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="MNIST_dataset", train=True, transform=mnist_transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="MNIST_dataset", train=False, transform=mnist_transform
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return "MNIST", train_dataloader, test_dataloader
