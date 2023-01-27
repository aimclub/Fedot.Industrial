import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Type, Union

import torch
from fedot.core.log import default_log as Logger
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from core.architecture.abstraction.сheckers import parameter_value_check
from core.metrics.cv_metrics import LossesAverager, ClassificationMetricCounter


# lambda x: tuple(zip(*x))


@dataclass
class FitParameters:
    dataset_name: str
    train_dl: DataLoader
    val_dl: DataLoader
    num_epochs: int
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    optimizer_params: Dict = field(default_factory=dict)
    lr_scheduler: Optional[Type] = None
    lr_scheduler_params: Dict = field(default_factory=dict)
    models_path: Union[Path, str] = 'models'
    summary_path: Union[Path, str] = 'summary'
    class_metrics: bool = False


def write_scores(
        writer: SummaryWriter,
        phase: str,
        scores: Dict[str, float],
        x: int,
):
    """Write scores from dictionary by SummaryWriter.

    Args:
        writer: SummaryWriter object for writing scores.
        phase: Experiment phase for grouping records, e.g. 'train'.
        scores: Dictionary {metric_name: value}.
        x: The independent variable.
    """
    for key, score in scores.items():
        writer.add_scalar(f"{phase}/{key}", score, x)


class NNExperimenter:

    def __init__(
            self,
            model: torch.nn.Module,
            metric: str,
            metric_counter,
            name: Optional[str],
            weights: Optional[str],
            gpu: bool,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.model = model
        if weights is not None:
            self.model.load_state_dict(torch.load(weights))
        self.device = torch.device('cuda' if gpu else 'cpu')
        self.model.to(self.device)
        self.name = name if name is not None else type(model).__name__
        self.best_score = -1
        self.metric = metric
        self.metric_counter = metric_counter

    def fit(
            self,
            p: FitParameters,
            phase: str = 'train',
            model_losses: Optional[Callable] = None,
            start_epoch: int = 0
    ) -> None:
        """Run experiment.

        Args:
            num_epochs: Number of epochs.
        """
        model_path = os.path.join(p.models_path, p.dataset_name, self.name, phase)
        summary_path = os.path.join(p.summary_path, p.dataset_name, self.name, phase)
        writer = SummaryWriter(summary_path)

        init_scores = self.val_loop(dataloader=p.val_dl, class_metrics=p.class_metrics)
        write_scores(writer, 'val', init_scores, start_epoch)
        start_epoch += 1

        optimizer = p.optimizer(self.model.parameters(), **p.optimizer_params)
        self.logger.info(f"{self.name}, using device: {self.device}")
        for epoch in range(start_epoch, start_epoch + p.num_epochs):
            self.logger.info(f"Epoch {epoch}")
            train_scores = self.train_loop(
                dataloader=p.train_dl,
                optimizer=optimizer,
                model_losses=model_losses
            )
            write_scores(writer, 'train', train_scores, epoch)
            val_scores = self.val_loop(
                dataloader=p.val_dl,
                class_metrics=p.class_metrics
            )
            write_scores(writer, 'val', val_scores, epoch)
            self.save_model_sd_if_best(val_scores=val_scores, file_path=model_path)
        self.load_model(model_path)
        writer.close()

    def save_model_sd_if_best(self, val_scores: Dict, file_path):
        if val_scores[self.metric] > self.best_score:
            self.best_score = val_scores[self.metric]
            self.logger.info(f'Best {self.metric} score: {self.best_score}')
            self.save_model(file_path=file_path)

    def save_model(
            self,
            file_path: str,
            state_dict: bool = True,
    ) -> None:
        """Save the model or its state dict.

        Args:
            file_path: Path to the file without extension.
            state_dict: If ``True`` save state_dict with extension ".sd.pt",
                else save all model with extension ".model.pt".
        """
        dir_path, file_name = os.path.split(file_path)
        os.makedirs(dir_path, exist_ok=True)
        file_name = f"{file_name}.{'sd' if state_dict else 'model'}.pt"
        file_path = os.path.join(dir_path, file_name)
        data = self.model.state_dict() if state_dict else self.model
        try:
            torch.save(data, file_path)
        except Exception:
            torch.save(data, file_name)
            shutil.move(file_name, dir_path)
        self.logger.info(f"Saved to {os.path.abspath(file_path)}.")

    def load_model(
            self,
            file_path: str,
            state_dict: bool = True,
    ) -> None:
        """Load the model or its state dict.

        Args:
            file_path: Path to the file without extension.
            state_dict: If ``True`` load state_dict with extension ".sd.pt",
                else load all model with extension ".model.pt".
        """
        file_path = f"{file_path}.{'sd' if state_dict else 'model'}.pt"
        data = torch.load(file_path)
        if state_dict:
            self.model.load_state_dict(data)
            self.model.to(self.device)
            self.logger.info("Model state dict loaded.")
        else:
            self.model = data
            self.logger.info("Model loaded.")

    def size_of_model(self) -> float:
        """Returns size of model in Mb."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1e6

    def number_of_model_params(self) -> int:
        """Returns number of model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def apply_func(
            self,
            func: Callable,
            func_params: Dict = {},
            condition: Optional[Callable] = None
    ):
        for module in filter(condition, self.model.modules()):
            func(module, **func_params)

    def forward(self, x):
        """Have to implement the forward method of the model and return predictions."""
        pass

    def forward_with_loss(self, x, y) -> Dict[str, torch.Tensor]:
        """Have to implement the train forward method and return dictionary of losses."""
        pass

    def predict_on_batch(self, x, proba: bool) -> List:
        """Have to implement the prediction method on batch."""
        pass

    def predict(
            self,
            dataloader: DataLoader,
            proba: bool = False,
    ) -> Dict:
        ids = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            for x, id in tqdm(dataloader):
                ids.extend(id)
                preds.extend(self.predict_on_batch(x, proba=proba))
        return dict(zip(ids, preds))

    def predict_proba(self, dataloader: DataLoader) -> Dict:
        return self.predict(dataloader, proba=True)

    def train_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            model_losses: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Training method of the model.

        Returns:
            Dictionary {metric_name: value}.
        """
        self.model.train()
        train_scores = LossesAverager()
        batches = tqdm(dataloader)
        for x, y in batches:
            losses = self.forward_with_loss(x, y)
            if model_losses is not None:
                losses.update(model_losses(self.model))
            train_scores.update(losses)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batches.set_postfix(train_scores.compute())
        return train_scores.compute()

    def val_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            class_metrics: bool = False,
    ) -> Dict[str, float]:
        """Validation method of the model. Returns val_scores

        Returns:
            Dictionary {metric_name: value}.
        """
        self.model.eval()
        metric = self.metric_counter(class_metrics=class_metrics)
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                preds = self.forward(x)
                metric.update(preds, y)
        return metric.compute()


class ClassificationExperimenter(NNExperimenter):

    def __init__(
            self,
            model: torch.nn.Module,
            metric: str = 'f1',
            loss: Callable = torch.nn.CrossEntropyLoss(),
            name: Optional[str] = None,
            weights: Optional[str] = None,
            gpu: bool = True,
    ):
        parameter_value_check(
            parameter='metric',
            value=metric,
            valid_values={'f1', 'accuracy', 'precision', 'recall', 'roc_auc'},
        )
        super().__init__(
            model=model,
            metric=metric,
            metric_counter=ClassificationMetricCounter,
            name=name,
            weights=weights,
            gpu=gpu
        )
        self.loss = loss

    def forward(self, x):
        """Have to implement the forward method of the model and return predictions."""
        x = x.to(self.device)
        return self.model(x)

    def forward_with_loss(self, x, y) -> Dict[str, torch.Tensor]:
        """Have to implement the train forward method and return loss."""
        y = y.to(self.device)
        preds = self.forward(x)
        return {'loss': self.loss(preds, y)}

    def predict_on_batch(self, x, proba: bool) -> List:
        """Returns prediction for sample."""
        assert not self.model.training, "model must be in eval mode"
        x = x.to(self.device)
        pred = self.model(x)
        if proba:
            pred = softmax(pred, dim=1).cpu().detach().tolist()
        else:
            pred = pred.argmax(1).cpu().detach().tolist()
        return pred


class FasterRCNNExperimenter(NNExperimenter):

    def __init__(
            self,
            num_classes: int,
            model_params: Dict = {},
            metric: str = 'map',
            name: Optional[str] = None,
            weights: Optional[str] = None,
            gpu: bool = True,
    ):
        parameter_value_check(
            parameter='metric',
            value=metric,
            valid_values={'map', 'map_50', 'map_75'},
        )
        model = fasterrcnn_resnet50_fpn(**model_params)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        super().__init__(
            model=model,
            metric=metric,
            metric_counter=MeanAveragePrecision,
            name=name,
            weights=weights,
            gpu=gpu
        )

    def forward(self, x):
        """Have to implement the forward method of the model and return predictions."""
        assert not self.model.training
        images = list(image.to(self.device) for image in x)
        preds = self.model(images)
        return [{k: v.to('cpu').detach() for k, v in p.items()} for p in preds]

    def forward_with_loss(self, x, y) -> Dict[str, torch.Tensor]:
        """Have to implement the train forward method and return loss."""
        assert self.model.training, "model must be in training mode"
        images = [image.to(self.device) for image in x]
        targets = [{k: v.to(self.device) for k, v in target.items()} for target in y]
        return self.model(images, targets)

    def predict_on_batch(self, x, proba: bool) -> List:
        """Returns prediction for sample."""
        assert not self.model.training, "model must be in eval mode"
        images = [image.to(self.device) for image in x]
        preds = self.model(images)
        if not proba:
            for pred in preds:
                not_thresh = pred['scores'] > 0.5
                pred['boxes'] = pred['boxes'][not_thresh]
                pred['labels'] = pred['labels'][not_thresh]
                pred.pop('scores')
        preds = [{k: v.tolist() for k, v in p.items()} for p in preds]
        return preds
