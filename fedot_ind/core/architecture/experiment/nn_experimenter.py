"""This module contains classes for working with neural networks using pytorch."""
import logging
import os
import shutil
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union
from abc import ABC, abstractmethod

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from tqdm import tqdm

from fedot_ind.core.architecture.abstraction.writers import CSVWriter, TFWriter, WriterComposer
from fedot_ind.core.architecture.abstraction.сheckers import parameter_value_check
from fedot_ind.core.metrics.cv_metrics import MetricCounter, ClassificationMetricCounter, LossesAverager, \
    ObjectDetectionMetricCounter, SegmentationMetricCounter


@dataclass(frozen=True)
class FitParameters:
    """The data class containing the training parameters.

    Args:
        dataset_name: Name of dataset.
        train_dl: Train dataloader.
        val_dl: Validation dataloader.
        num_epochs: Number of training epochs.
        optimizer: Type of model optimizer, e.g. ``torch.optim.Adam``.
        lr_scheduler: Type of learning rate scheduler, e.g ``torch.optim.lr_scheduler.StepLR``.
        models_path: Path to folder for saving models.
        summary_path: Path to folder for writing experiment summary info.
        validation_period: Validation frequency in epochs.
        class_metrics: If ``True``, calculates validation metrics for each class.
        description: Additional line describing the experiment.

    """

    dataset_name: str
    train_dl: DataLoader
    val_dl: DataLoader
    num_epochs: int
    optimizer: Union[Type[torch.optim.Optimizer], partial] = torch.optim.Adam
    lr_scheduler: Optional[Union[Type[torch.optim.lr_scheduler.LRScheduler], partial]] = None
    models_path: Union[Path, str] = 'models'
    summary_path: Union[Path, str] = 'summary'
    validation_period: int = 1
    class_metrics: bool = False
    description: str = ''


class NNExperimenter(ABC):
    """Generalized class for working with neural models.

    Args:
        model: Trainable model.
        metric: Target metric by which models are compared.
        metric_counter: Class for calculating metrics.
            Must implement ``update`` and ``compute`` methods.
        name: Name of the model.
        weights: Path to the model state_dict to load weights.
        device: String passed to ``torch.device`` initialization.

    """

    def __init__(
            self,
            model: torch.nn.Module,
            metric: str,
            metric_counter: Type[MetricCounter],
            name: Optional[str],
            weights: Optional[str],
            device: str,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.device = torch.device(device)
        if weights is not None:
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.to(self.device)
        self.name = name if name is not None else type(model).__name__
        self.best_score = -1
        self.metric = metric
        self.metric_counter = metric_counter

    def fit(self,
            p: FitParameters,
            phase: str = 'train',
            model_losses: Optional[Callable] = None,
            filter_pruning: Optional[Dict] = None,
            start_epoch: int = 0,
            initial_validation: bool = False
            ) -> None:
        """Run model training.

        Args:
            p: An object containing training parameters.
            phase: String explanation of training.
            model_losses: Function for calculating losses from model weights.
            filter_pruning: Parameters (pruning function and condition) passed to ``apply_func`` function.
            start_epoch: Initial training epoch.
            initial_validation: If ``True`` run validation loop before training.

        """
        model_path = os.path.join(p.models_path, p.dataset_name, self.name, p.description, phase)
        summary_path = os.path.join(p.summary_path, p.dataset_name, self.name, p.description, phase)
        writer = WriterComposer(summary_path, [TFWriter, CSVWriter])
        self.logger.info(f"{phase}: {self.name}, using device: {self.device}")

        if initial_validation:
            init_scores = self.val_loop(dataloader=p.val_dl, class_metrics=p.class_metrics)
            writer.write_scores('val', init_scores, start_epoch)
            self._save_model_sd_if_best(val_scores=init_scores, file_path=model_path)
        start_epoch += 1

        optimizer = p.optimizer(self.model.parameters())
        lr_scheduler = None
        if p.lr_scheduler is not None:
            lr_scheduler = p.lr_scheduler(optimizer)

        for epoch in range(start_epoch, start_epoch + p.num_epochs):
            self.logger.info(f"Epoch {epoch}")
            train_scores = self.train_loop(
                dataloader=p.train_dl,
                optimizer=optimizer,
                model_losses=model_losses
            )
            writer.write_scores('train', train_scores, epoch)

            if filter_pruning is not None:
                self._apply_function(**filter_pruning)

            if epoch % p.validation_period == 0:
                val_scores = self.val_loop(
                    dataloader=p.val_dl,
                    class_metrics=p.class_metrics
                )
                writer.write_scores('val', val_scores, epoch)
                self._save_model_sd_if_best(val_scores=val_scores, file_path=model_path)
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    lr_scheduler.step(val_scores[self.metric])

            if isinstance(lr_scheduler, LRScheduler) and not isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step()
        self.load_model(model_path)
        self.logger.info(f'{self.metric} score: {self.best_score}')
        writer.close()

    def _save_model_sd_if_best(self, val_scores: Dict, file_path):
        """Save the model state dict if the best result on the target metric is achieved.

        Args:
            val_scores: Validation metric dictionary.
            file_path: Path to the file without extension.

        """
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
        data = torch.load(file_path, map_location=self.device)
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

    def _apply_function(
            self,
            func: Callable,
            condition: Optional[Callable] = None
    ):
        """Applies the passed function to model layers by condition.

        Args:
            func: Applicable function.
            condition: Condition function for layer filtering.
        """
        for module in filter(condition, self.model.modules()):
            func(module)

    @abstractmethod
    def _forward(self, x: torch.Tensor):
        """Have to implement the forward method of the model and return predictions."""
        raise NotImplementedError

    @abstractmethod
    def _forward_with_loss(self, x: torch.Tensor, y) -> Dict[str, torch.Tensor]:
        """Have to implement the train forward method and return dictionary of losses."""
        raise NotImplementedError

    @abstractmethod
    def _predict_on_batch(self, x: torch.Tensor, proba: bool) -> List:
        """Have to implement the prediction method on batch."""
        raise NotImplementedError

    def predict(
            self,
            dataloader: DataLoader,
            proba: bool = False,
    ) -> Dict:
        """Computes predictions for data in dataloader.

        Args:
            dataloader: Data loader with prediction dataset.
            proba: If ``True`` computes probabilities.
        """
        self.logger.info('Computing predictions')
        ids = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            for x, id in tqdm(dataloader, desc='predict'):
                ids.extend(id)
                preds.extend(self._predict_on_batch(x, proba=proba))
        return dict(zip(ids, preds))

    def predict_proba(self, dataloader: DataLoader) -> Dict:
        """Computes probability predictions for data in dataloader.

        Args:
            dataloader: Data loader with prediction dataset.
        """
        return self.predict(dataloader, proba=True)

    def train_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            model_losses: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Training method of the model.

        Args:
            dataloader: Training data loader.
            optimizer: Model optimizer.
            model_losses: Function for calculating losses from model weights.

        Returns:
            Dict: {metric_name: value}.
        """
        self.model.train()
        train_scores = LossesAverager()
        batches = tqdm(dataloader, desc='train')
        for x, y in batches:
            losses = self._forward_with_loss(x, y)
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

        Args:
            dataloader: Validation data loader.
            class_metrics: If ``True``, calculates validation metrics for each class.

        Returns:
            Dict: {metric_name: value}.
        """
        self.model.eval()
        metric = self.metric_counter(class_metrics=class_metrics)
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc='val'):
                preds = self._forward(x)
                metric.update(preds, y)
        return metric.compute()


class ClassificationExperimenter(NNExperimenter):
    """Class for working with classification models.

    Args:
        model: Trainable model.
        metric: Target metric by which models are compared.
            One of ``'f1'``, ``'accuracy'``, ``'precision'``, ``'recall'``, ``'roc_auc'``.
        loss: Loss function applied to model output.
        name: Name of the model.
        weights: Path to the model state_dict to load weights.
        device: String passed to ``torch.device`` initialization.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            metric: str = 'f1',
            loss: Callable = torch.nn.CrossEntropyLoss(),
            name: Optional[str] = None,
            weights: Optional[str] = None,
            device: str = 'cuda',
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
            device=device
        )
        self.loss = loss

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward method of the model and returns predictions."""
        x = x.to(self.device)
        return self.model(x)

    def _forward_with_loss(self, x: torch.Tensor, y) -> Dict[str, torch.Tensor]:
        """Implements the train forward method and returns loss."""
        y = y.to(self.device)
        preds = self._forward(x)
        return {'loss': self.loss(preds, y)}

    def _predict_on_batch(self, x: torch.Tensor, proba: bool) -> List:
        """Returns prediction on batch."""
        assert not self.model.training, "model must be in eval mode"
        x = x.to(self.device)
        pred = self.model(x)
        if proba:
            pred = softmax(pred, dim=1).cpu().detach().tolist()
        else:
            pred = pred.argmax(1).cpu().detach().tolist()
        return pred


class ObjectDetectionExperimenter(NNExperimenter):
    """Class for working with object detection models.

    Args:
        model: Trainable model.
        metric: Target metric by which models are compared.
            One of ``'map'``, ``'map_50'``, ``'map_75'``.
        name: Name of the model.
        weights: Path to the model state_dict to load weights.
        device: String passed to ``torch.device`` initialization.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            metric: str = 'map',
            name: Optional[str] = None,
            weights: Optional[str] = None,
            device: str = 'cuda',
    ):
        parameter_value_check(
            parameter='metric',
            value=metric,
            valid_values={'map', 'map_50', 'map_75'},
        )
        super().__init__(
            model=model,
            metric=metric,
            metric_counter=ObjectDetectionMetricCounter,
            name=name,
            weights=weights,
            device=device
        )

    def _forward(self, x: torch.Tensor) -> List:
        """Implements the forward method of the model and returns predictions."""
        assert not self.model.training
        images = list(image.to(self.device) for image in x)
        preds = self.model(images)
        return [{k: v.to('cpu').detach() for k, v in p.items()} for p in preds]

    def _forward_with_loss(self, x: torch.Tensor, y) -> Dict[str, torch.Tensor]:
        """Implements the train forward method and returns loss."""
        assert self.model.training, "model must be in training mode"
        images = [image.to(self.device) for image in x]
        targets = [{k: v.to(self.device) for k, v in target.items()} for target in y]
        return self.model(images, targets)

    def _predict_on_batch(self, x: torch.Tensor, proba: bool) -> List:
        """Returns prediction on batch."""
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


class SegmentationExperimenter(NNExperimenter):
    """Class for working with semantic segmentation models.

    Args:
        model: Trainable model.
        metric: Target metric by which models are compared.
            One of ``'iou'``, ``'dice'``.
        loss: Loss function applied to model output.
        name: Name of the model.
        weights: Path to the model state_dict to load weights.
        device: String passed to ``torch.device`` initialization.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            metric: str = 'iou',
            loss: Callable = torch.nn.CrossEntropyLoss(),
            name: Optional[str] = None,
            weights: Optional[str] = None,
            device: str = 'cuda',
    ):
        parameter_value_check(
            parameter='metric',
            value=metric,
            valid_values={'iou', 'dice'},
        )
        super().__init__(
            model=model,
            metric=metric,
            metric_counter=SegmentationMetricCounter,
            name=name,
            weights=weights,
            device=device
        )
        self.loss = loss

    def _forward(self, x):
        """Implements the forward method of the model and returns predictions."""
        x = x.to(self.device)
        return self.model(x)['out'].to('cpu').detach()

    def _forward_with_loss(self, x, y) -> Dict[str, torch.Tensor]:
        """Implements the train forward method and returns loss."""
        x = x.to(self.device)
        y = y.to(self.device)
        preds = self.model(x)['out']
        return {'loss': self.loss(preds, y)}

    def _predict_on_batch(self, x, proba: bool) -> List:
        """Returns prediction on batch."""
        assert not self.model.training, "model must be in eval mode"
        x = x.to(self.device)
        pred = torch.sigmoid(self.model(x)).cpu().detach()
        if not proba:
            pred = pred.argmax(1)
        return pred
