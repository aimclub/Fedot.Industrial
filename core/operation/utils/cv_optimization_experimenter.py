import copy
import os
import time
from typing import Dict, Type, Set, Union, List

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.metrics.svd_loss import OrthogonalLoss, HoyerLoss
from core.models.cnn.classification_models import MODELS
from core.models.cnn.decomposed_conv import DecomposedConv2d
from core.operation.utils.classification_dataloaders import get_dataloaders
from core.operation.utils.sfp_tools import zerolize_filters
from core.operation.utils.svd_tools import energy_threshold_pruning, decompose_module


def parameter_value_check(parameter: str, value: str, valid_values: Set[str]) -> None:
    if value not in valid_values:
        raise ValueError(
            "{} must be one of {}, but got {}='{}'".format(
                parameter, valid_values, parameter, value
            )
        )


class OptimizationExperimenter:
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
        target_loss: Type[torch.nn.Module],
        loss_params: Dict,
        model_optimizer_params: Dict,
        target_metric: str = "f1",
        summary_path: str = "runs",
        summary_per_class: bool = False,
        progress: bool = True,
    ) -> None:

        parameter_value_check(
            parameter="model",
            value=model,
            valid_values=set(MODELS.keys()))
        parameter_value_check(
            parameter="target_metric",
            value=target_metric,
            valid_values={"f1", "accuracy", "precision", "recall", "roc_auc"},
        )

        self.model_name = model
        self.exp_description = "{}/{}/".format(dataset, model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dl, self.val_dl, self.num_classes = get_dataloaders(
            dataset, **dataset_params
        )
        self.model = MODELS[model](num_classes=self.num_classes, **model_params)
        self.default_size = self.size_of_model()
        self.default_n_params = self.number_of_params()
        print("Default size: {:.2f} MB".format(self.default_size))
        self.target_loss = target_loss(**loss_params)
        self.models_path = models_saving_path
        self.summary_per_class = summary_per_class
        self.progress = progress
        self.target_metric = "val/" + target_metric
        self.best_score = 0
        self.epoch = 0
        self.init_model_optimizer(**model_optimizer_params)
        self.model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.writer = SummaryWriter(os.path.join(summary_path, self.exp_description))
        print("{}, using device: {}".format(self.exp_description, self.device))

    def init_model_optimizer(self) -> None:
        self.exp_description += "base"

    def optimize_during_training(self) -> None:
        pass

    def final_optimize(self) -> None:
        pass

    def run(self, num_epochs: int) -> None:
        for _ in range(num_epochs):
            self.epoch += 1
            if self.progress:
                print("Epoch {}".format(self.epoch))
            train_scores = self.train_loop()
            for key, score in train_scores.items():
                self.writer.add_scalar(key, score, self.epoch)
            self.optimize_during_training()
            val_scores = self.val_loop()
            for key, score in val_scores.items():
                self.writer.add_scalar(key, score, self.epoch)
            if val_scores[self.target_metric] > self.best_score:
                self.best_score = val_scores[self.target_metric]
                self.save_model()
        self.final_optimize()
        self.writer.close()

    def train_loop(self) -> (Dict[str, float]):
        self.model.train()
        batches = tqdm(self.train_dl) if self.progress else self.train_dl
        train_scores = {"train/accuracy": 0, "train/loss": 0}
        for x, y in batches:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            loss = self.target_loss(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_scores["train/loss"] += loss.item()
            train_scores["train/accuracy"] += (pred.argmax(1) == y).type(torch.float).mean().item()
        for key in train_scores:
            train_scores[key] /= len(self.train_dl)
        return train_scores

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
                val_loss += self.target_loss(pred, y).item()
        total_time = time.time() - start
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )
        val_scores = {
            "val/loss": val_loss / len(self.val_dl),
            "val/inference_time": total_time / len(self.val_dl),
            "val/accuracy": accuracy_score(y_true, y_pred),
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
            "val/roc_auc": roc_auc_score(y_true, y_score, multi_class="ovo"),
        }
        if self.summary_per_class:
            f1s = f1_score(y_true, y_pred, average=None)
            val_scores.update({"per_class/{}".format(i): s for i, s in enumerate(f1s)})
        return val_scores

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

    def number_of_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class SVDOptimizationExperimenter(OptimizationExperimenter):

    def init_model_optimizer(
            self,
            decomposing_mode: str,
            hoer_loss_factor: float,
            orthogonal_loss_factor: float,
            energy_thresholds: List[float],
    ) -> None:
        decompose_module(model=self.model, decomposing_mode=decomposing_mode)
        print("SVD decomposed size: {:.2f} MB".format(self.size_of_model()))
        self.hoer_loss = HoyerLoss(hoer_loss_factor)
        self.orthogonal_loss = OrthogonalLoss(self.device, orthogonal_loss_factor)
        self.energy_thresholds = energy_thresholds
        self.exp_description += "SVD_{}_O-{:.1f}_H-{:.6f}".format(
            decomposing_mode, orthogonal_loss_factor, hoer_loss_factor
        )

    def final_optimize(self) -> None:
        self.summary_per_class = False
        self.load_model()
        default_model = copy.deepcopy(self.model)
        default_scores = self.val_loop()

        for e in self.energy_thresholds:
            int_e = e * 100000
            self.model = copy.deepcopy(default_model)

            start = time.time()
            self.prune_model(e)
            pruning_time = time.time() - start

            val_scores = self.val_loop()
            size = self.size_of_model()
            n_params = self.number_of_params()

            self.writer.add_scalar("abs(e)/pruning_time", pruning_time, int_e)
            self.writer.add_scalar("abs(e)/size_MB", size, int_e)

            size_p = size / self.default_size * 100
            n_params_p = n_params / self.default_n_params * 100

            self.writer.add_scalar("percentage(e)/size", size_p, int_e)
            self.writer.add_scalar("percentage(e)/number_of_params", n_params_p, int_e)

            for score in val_scores:
                score_p = val_scores[score] / default_scores[score] * 100
                delta_score = val_scores[score] - default_scores[score]
                self.writer.add_scalar("percentage(e)/{}".format(score), score_p, int_e)
                self.writer.add_scalar("delta(e)/{}".format(score), delta_score, int_e)

    def prune_model(self, e):
        for module in self.model.modules():
            if isinstance(module, DecomposedConv2d):
                energy_threshold_pruning(conv=module, energy_threshold=e)

    def train_loop(self) -> (Dict[str, float]):
        self.model.train()
        batches = tqdm(self.train_dl) if self.progress else self.train_dl
        train_scores = {
            "train/accuracy": 0,
            "train/loss": 0,
            "train/hoer_loss": 0,
            "train/orthogonal_loss": 0,
        }
        for x, y in batches:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            loss = self.target_loss(pred, y)
            o_loss = self.orthogonal_loss(self.model)
            h_loss = self.hoer_loss(self.model)
            loss += o_loss + h_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_scores["train/loss"] += loss.item()
            train_scores["train/hoer_loss"] += h_loss.item()
            train_scores["train/orthogonal_loss"] += o_loss.item()
            train_scores["train/accuracy"] += (pred.argmax(1) == y).type(torch.float).mean().item()
        for key in train_scores:
            train_scores[key] /= len(self.train_dl)
        return train_scores


class SFPOptimizationExperimenter(OptimizationExperimenter):

    def init_model_optimizer(self, pruning_ratio: float) -> None:
        self.pruning_ratio = pruning_ratio
        self.exp_description += "SFP_P-{:.2f}".format(pruning_ratio)

    def optimize_during_training(self) -> None:
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                zerolize_filters(conv=module, pruning_ratio=self.pruning_ratio)


def run_experiment(
        mode: str,
        experiment_params:  Dict,
        optimizer_params: Dict,
        num_epochs: int
) -> None:
    modes = {
        "base": OptimizationExperimenter,
        "SVD": SVDOptimizationExperimenter,
        "SFP": SFPOptimizationExperimenter
    }
    parameter_value_check("mode", mode, set(modes.keys()))
    experiment_params["model_optimizer_params"] = optimizer_params
    experimenter = modes[mode](**experiment_params)
    experimenter.run(num_epochs)
