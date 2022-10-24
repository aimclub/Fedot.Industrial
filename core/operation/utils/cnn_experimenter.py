import copy
import os
import time
from typing import Dict, Type, Set, Union

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.metrics.svd_loss import OrthogonalLoss, HoyerLoss
from core.models.cnn.classification_models import MODELS, QUA_MODELS
from core.operation.utils.classification_dataloaders import get_dataloaders
from core.operation.utils.sfp_tools import EnergyThresholdPruning, SoftFilterPruning
from core.operation.utils.sfp_tools import decompose_module, prune_model


def parameter_value_check(parameter: str, value: str, valid_values: Set[str]) -> None:
    if value not in valid_values:
        raise ValueError(
            "{} must be one of {}, but got {}='{}'".format(
                parameter, valid_values, parameter, value
            )
        )


class Experimenter:
    """Class of experiment on compression of convolutional classification models."""

    def __init__(
        self,
        dataset: str,
        dataset_params: Dict,
        model: str,
        model_params: Dict,
        models_saving_path: str,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict,
        loss_fn: Type[torch.nn.Module],
        loss_params: Dict,
        compression_mode: str,
        compression_params: Dict,
        target_metric: str = "f1",
        summary_path: str = "runs",
        summary_per_class: bool = False,
        progress: bool = True,
    ) -> None:

        parameter_value_check(
            parameter="model",
            value=model,
            valid_values=MODELS.keys())
        parameter_value_check(
            parameter="compression_mode",
            value=compression_mode,
            valid_values={"none", "SVD", "SFP", "PTQ"})
        parameter_value_check(
            parameter="target_metric",
            value=target_metric,
            valid_values={"f1", "accuracy", "precision", "recall", "roc_auc"},
        )

        self.model_name = model
        self.exp_description = "{}/{}/{}".format(dataset, model, compression_mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_score = 0
        self.train_dl, self.val_dl, self.num_classes = get_dataloaders(
            dataset, **dataset_params
        )
        self.model = MODELS[model](num_classes=self.num_classes, **model_params)
        self.default_size = self.size_of_model()
        print("Default size: {:.2f} MB".format(self.default_size))
        self.loss_fn = loss_fn(**loss_params)
        self.compression_mode = compression_mode
        self.compression_params = compression_params
        self.models_path = models_saving_path
        self.summary_per_class = summary_per_class
        self.progress = progress
        self.target_metric = target_metric

        if self.compression_mode == "SVD":
            decompose_module(
                model=self.model,
                decomposing_mode=self.compression_params["decomposing_mode"],
            )
            print("SVD decomposed size: {:.2f} MB".format(self.size_of_model()))
            self.exp_description += "_{}_O-{:.1f}_H-{:.6f}".format(
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
            self.exp_description += "_P-{:.2f}".format(
                self.compression_params["pruning_ratio"],
            )
        elif self.compression_mode == "PTQ":
            parameter_value_check(
                parameter="model",
                value=model,
                valid_values=QUA_MODELS.keys())

        self.model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.writer = SummaryWriter(os.path.join(summary_path, self.exp_description))
        print("{}, using device: {}".format(self.exp_description, self.device))

    def run(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            if self.progress:
                print("Epoch {}".format(epoch))

            train_accuracy, train_loss, svd_losses = self.train_loop()

            self.writer.add_scalar("train/accuracy", train_accuracy, epoch)
            self.writer.add_scalar("train/loss", train_loss, epoch)

            if self.compression_mode == "SVD":
                for key in svd_losses.keys():
                    self.writer.add_scalar("train/" + key, svd_losses[key], epoch)
            elif self.compression_mode == "SFP":
                prune_model(
                    model=self.model, optimizer=self.compression_params["optimizer"]
                )

            val_scores = self.val_loop()

            if val_scores[self.target_metric] > self.best_score:
                self.best_score = val_scores[self.target_metric]
                self.save_model()

            for key in ("loss", "accuracy", "precision", "recall", "f1", "roc_auc"):
                self.writer.add_scalar("val/" + key, val_scores[key], epoch)

            if self.summary_per_class:
                for key in range(self.num_classes):
                    self.writer.add_scalar(
                        "classes/{}".format(key),
                        val_scores[key],
                        epoch
                    )

        if self.compression_mode == "SVD":
            self.auto_pruning()
        elif self.compression_mode == "PTQ":
            self.post_training_static_quantization()

        self.writer.close()

    def train_loop(self) -> (float, float, Dict):
        self.model.train()
        batches = tqdm(self.train_dl) if self.progress else self.train_dl
        train_accuracy = 0
        train_loss = 0
        svd_losses = {"orthogonal_loss": 0, "hoer_loss": 0}
        for x, y in batches:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            train_loss += loss.item()
            train_accuracy += (pred.argmax(1) == y).type(torch.float).mean().item()

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
        train_accuracy /= n
        train_loss /= n
        svd_losses = {key: svd_losses[key] / n for key in svd_losses.keys()}
        return train_accuracy, train_loss, svd_losses

    def val_loop(self) -> Dict[Union[str, int], float]:
        self.model.eval()
        batches = tqdm(self.val_dl) if self.progress else self.val_dl
        val_loss = 0
        y_true = []
        y_pred = []
        y_score = []
        start = time.time()
        with torch.no_grad():
            for x, y in batches:
                y_true.extend(y)
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                y_pred.extend(pred.cpu().argmax(1))
                y_score.extend(softmax(pred, dim=1).tolist())
                val_loss += self.loss_fn(pred, y).item()

        total_time = time.time() - start
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )
        val_scores = {
            "loss": val_loss / len(self.val_dl),
            "time": total_time / len(self.val_dl),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc_score(y_true, y_score, multi_class="ovo"),
        }
        if self.summary_per_class:
            val_scores.update(enumerate(f1_score(y_true, y_pred, average=None)))
        return val_scores

    def auto_pruning(self) -> None:
        self.load_model()
        default_model = copy.deepcopy(self.model)
        default_scores = self.val_loop()

        for e in self.compression_params["e"]:
            int_e = e * 100000
            self.model = copy.deepcopy(default_model)

            start = time.time()
            old_params, new_params = prune_model(
                model=self.model, optimizer=EnergyThresholdPruning(e)
            )
            pruning_time = time.time() - start

            val_scores = self.val_loop()

            size_p = new_params / old_params * 100
            torch.save(self.model.state_dict(), "runs/{}".format(e))

            self.writer.add_scalar("abs(e)/pruning time", pruning_time, int_e)
            self.writer.add_scalar("abs(e)/size MB", self.size_of_model(), int_e)
            self.writer.add_scalar("percentage(e)/size", size_p, int_e)

            for score in ("time", "accuracy", "precision", "recall", "f1", "roc_auc"):
                score_p = val_scores[score] / default_scores[score] * 100
                delta_score = val_scores[score] - default_scores[score]
                self.writer.add_scalar("percentage(e)/{}".format(score), score_p, int_e)
                self.writer.add_scalar("delta(e)/{}".format(score), delta_score, int_e)

    def post_training_static_quantization(self) -> None:
        self.model = QUA_MODELS[self.model_name](num_classes=self.num_classes)
        file_path = os.path.join(self.models_path, self.exp_description)
        self.model.load_state_dict(torch.load(file_path))
        self.device = "cpu"
        self.model.to(self.device)

        print("Size before quantization: {:.2f} MB".format(self.size_of_model()))
        self.model.fuse_model()
        self.model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        torch.ao.quantization.prepare(self.model, inplace=True)
        default_scores = self.val_loop()
        torch.ao.quantization.convert(self.model, inplace=True)
        print("Size after quantization: {:.2f} MB".format(self.size_of_model()))
        val_scores = self.val_loop()

        for score in ("time", "accuracy", "precision", "recall", "f1", "roc_auc"):
            score_p = val_scores[score] / default_scores[score] * 100
            delta_score = val_scores[score] - default_scores[score]
            print("{}: {}%, delta: {}".format(score, score_p, delta_score))

    def save_model(self) -> None:
        file_path = os.path.join(self.models_path, self.exp_description)
        dir_path = "/".join(file_path.split("/")[:-1])
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.model.state_dict(), file_path + ".pt")
        if self.progress:
            print("Model saved.")

    def load_model(self) -> None:
        file_path = os.path.join(self.models_path, self.exp_description)
        self.model.load_state_dict(torch.load(file_path + ".pt"))
        self.model.to(self.device)
        if self.progress:
            print("Model loaded.")

    def size_of_model(self) -> float:
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1e6