import copy
import time

import torch

from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Type
from tqdm import tqdm

from core.operation.utils.pruning_tools import decompose_module, prune_model
from core.operation.utils.pruning_tools import EnergyThresholdPruning, SoftFilterPruning
from core.operation.utils.classification_dataloaders import get_dataloaders
from core.metrics.svd_loss import OrthogonalLoss, HoyerLoss
from core.models.cnn.classification_models import MODELS


class Experimenter:
    """Class of experiment on compression of convolutional classification models."""

    def __init__(
        self,
        dataset: str,
        dataset_params: Dict,
        model: str,
        model_params: Dict,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict,
        loss_fn: Type[torch.nn.Module],
        loss_params: Dict,
        compression_mode: str,
        compression_params: Dict,
        progress: bool = True,
    ) -> None:

        self.exp_description = "{}/{}/".format(dataset, model)
        self.train_dl, self.val_dl, num_classes = get_dataloaders(
            dataset, **dataset_params
        )

        valid_compression_modes = {"none", "SVD", "SFP"}
        if compression_mode not in valid_compression_modes:
            raise ValueError(
                "compression_mode must be one of {}, but got compression_mode='{}'".format(
                    valid_compression_modes, compression_mode
                )
            )
        self.compression_mode = compression_mode
        self.compression_params = compression_params

        if model not in MODELS.keys():
            raise ValueError(
                "model must be one of {}, but got model='{}'".format(
                    MODELS.keys(), model
                )
            )
        self.model = MODELS[model](num_classes=num_classes, **model_params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = loss_fn(**loss_params)
        self.progress = progress

        if self.compression_mode == "SVD":
            decompose_module(
                model=self.model,
                decomposing_mode=self.compression_params["decomposing_mode"],
            )
            self.exp_description += "{}_O:{:.1f}_H:{:.6f}".format(
                self.compression_params["decomposing_mode"],
                self.compression_params["orthogonal_loss"],
                self.compression_params["hoer_loss"],
            )
            self.compression_params["hoer_loss"] = HoyerLoss(
                factor=self.compression_params["hoer_loss"]
            )
            self.compression_params["orthogonal_loss"] = OrthogonalLoss(
                device=self.device, factor=self.compression_params["orthogonal_loss"]
            )
        elif self.compression_mode == "SFP":
            self.compression_params["optimizer"] = SoftFilterPruning(
                pruning_ratio=self.compression_params["pruning_ratio"]
            )
            self.exp_description += "{}_P:{:.2f}".format(
                self.compression_mode,
                self.compression_params["pruning_ratio"],
            )

        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.writer = SummaryWriter("runs/" + self.exp_description)
        print("{}, using device: {}".format(self.exp_description, self.device))

    def run(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            if self.progress:
                print("Epoch {}".format(epoch))
            train_loss, svd_losses = self.train_loop()
            val_loss, accuracy, _ = self.val_loop()

            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/accuracy", accuracy, epoch)

            if self.compression_mode == "SVD":
                for key in svd_losses.keys():
                    self.writer.add_scalar("train/" + key, svd_losses[key], epoch)
            elif self.compression_mode == "SFP":
                prune_model(
                    model=self.model, optimizer=self.compression_params["optimizer"]
                )

        print("Accuracy: {:.2f}%".format(accuracy * 100))
        if self.compression_mode == "SVD":
            self.auto_pruning()

    def train_loop(self) -> (float, Dict):
        batches = tqdm(self.train_dl) if self.progress else self.train_dl
        train_loss = 0
        svd_losses = {"orthogonal_loss": 0, "hoer_loss": 0}
        for x, y in batches:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            train_loss += loss.item()

            if self.compression_mode == "SVD":
                orthogonal_loss = self.compression_params["orthogonal_loss"](self.model)
                hoer_loss = self.compression_params["hoer_loss"](self.model)
                svd_losses["orthogonal_loss"] += orthogonal_loss.item()
                svd_losses["hoer_loss"] += hoer_loss.item()
                loss += orthogonal_loss + hoer_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        n = len(self.train_dl)
        train_loss /= n
        svd_losses = {key: svd_losses[key] / n for key in svd_losses.keys()}
        return train_loss, svd_losses

    def val_loop(self) -> (float, float, float):
        batches = tqdm(self.val_dl) if self.progress else self.val_dl
        val_loss = 0
        accuracy = 0
        start = time.time()
        with torch.no_grad():
            for x, y in batches:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                val_loss += self.loss_fn(pred, y).item()
                accuracy += (pred.argmax(1) == y).type(torch.float).mean().item()

        total_time = time.time() - start
        n = len(self.val_dl)
        return val_loss / n, accuracy / n, total_time / n

    def auto_pruning(self) -> None:

        _, default_accuracy, default_time = self.val_loop()

        for e in self.compression_params["e"]:
            int_e = e * 100000
            new_model = copy.deepcopy(self.model)
            start = time.time()
            old_params, new_params = prune_model(
                model=new_model, optimizer=EnergyThresholdPruning(e)
            )
            pruning_time = time.time() - start

            _, accuracy, val_time = self.val_loop()

            size_p = new_params / old_params * 100
            time_p = val_time / default_time * 100
            accuracy_p = accuracy / default_accuracy * 100

            self.writer.add_scalar("e/accuracy %", accuracy_p, int_e)
            self.writer.add_scalar("e/size %", size_p, int_e)
            self.writer.add_scalar("e/inference time %", time_p, int_e)
            self.writer.add_scalar("e/pruning time", pruning_time, int_e)
            self.writer.add_scalar("acc-compr/division", accuracy_p / size_p, int_e)
            self.writer.add_scalar("acc-compr/subtraction)", accuracy_p - size_p, int_e)
