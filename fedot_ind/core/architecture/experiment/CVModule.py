import os
import shutil
import time
from typing import Dict, List, Optional, Set, Type, Union

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from fedot_ind.core.models.cnn.classification_models import CLF_MODELS
from fedot_ind.core.operation.optimization.ModelStructure import OPTIMIZATIONS


def _parameter_value_check(parameter: str, value: str, valid_values: Set[str]) -> None:
    """Checks if the parameter is in the set of allowed.

    Args:
        parameter: Name of the checked parameter.
        value: Value of the checked parameter.
        valid_values: Set of the valid parameter values.
    """
    if value not in valid_values:
        raise ValueError(
            f"{parameter} must be one of {valid_values}, but got {parameter}='{value}'"
        )


class _GeneralizedExperimenter:
    """Generalized class for working with models.

    Args:
        model: Trainable model.
        optimizable_module_name: Name of the module whose structure will be optimized.
        train_ds: Train dataset.
        val_ds: Validation dataset.
        num_classes: Number of classes in the dataset.
        dataloader_params: Parameter dictionary passed to dataloaders.
        name: Description of the experiment.
        models_path: Path to folder for saving models.
        summary_path: Path to folder for writing experiment summary info.
        summary_per_class: If ``True``, calculates the metrics for each class.
        weights: Path to the model state_dict to load weights (default: ``None``).
        metric: Target metric by which models are compared.
        gpu: If ``True``, uses GPU (default: ``True``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizable_module_name: str,
        train_ds: Dataset,
        val_ds: Dataset,
        num_classes: int,
        dataloader_params: Dict,
        name: str,
        models_path: str,
        summary_path: str,
        summary_per_class: bool,
        metric: str,
        weights: Optional[str] = None,
        gpu: bool = True,
    ) -> None:
        self.model = model
        if weights is not None:
            self.model.load_state_dict(torch.load(weights))
        self.optimizable_module_name = optimizable_module_name
        print(f"Default size: {self.size_of_model():.2f} MB")
        self.num_classes = num_classes

        self.train_dl = DataLoader(dataset=train_ds, shuffle=True, **dataloader_params)
        self.val_dl = DataLoader(dataset=val_ds, shuffle=False, **dataloader_params)

        self.name = name
        self.models_path = models_path
        self.summary_path = summary_path
        self.summary_per_class = summary_per_class
        self.device = torch.device('cuda' if gpu else 'cpu')
        self.best_score = 0
        self.metric = metric
        self.structure_optimization = None

    def save_model(self, name: str = 'trained') -> None:
        """Save all model.

        Args:
            name: File name (default: 'trained').
        """
        dir_path = os.path.join(self.models_path, self.name)
        model_name = f"{name}.model.pt"
        file_path = os.path.join(dir_path, model_name)
        os.makedirs(dir_path, exist_ok=True)
        try:
            torch.save(self.model, file_path)
        except Exception:
            torch.save(self.model, model_name)
            shutil.move(model_name, dir_path)
        print("Model saved.")

    def save_model_state_dict(self, name: str = 'trained') -> None:
        """Save model state_dict.

        Args:
            name: File name (default: 'trained').
        """
        dir_path = os.path.join(self.models_path, self.name)
        model_name = f"{name}.sd.pt"
        file_path = os.path.join(dir_path, model_name)
        os.makedirs(dir_path, exist_ok=True)
        try:
            torch.save(self.model.state_dict(), file_path)
        except Exception:
            torch.save(self.model.state_dict(), model_name)
            shutil.move(model_name, dir_path)
        print("Model state dict saved.")

    def load_model_state_dict(self, name: str = 'trained') -> None:
        """Load model state_dict to ``self.model``

        Args:
            name: File name (default: 'trained').
        """
        file_path = os.path.join(self.models_path, self.name, f"{name}.sd.pt")
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            self.model.to(self.device)
            print("Model state dict loaded.")
        else:
            print(f"File '{file_path}' does not exist.")

    def load_model(self, name: str = 'trained') -> None:
        """Load model to ``self.model``.

        Args:
            name: File name (default: 'trained').
        """
        file_path = os.path.join(self.models_path, self.name, f"{name}.model.pt")
        if os.path.exists(file_path):
            self.model = torch.load(file_path)
            self.model.to(self.device)
            print("Model loaded.")
        else:
            print(f"File '{file_path}' does not exist.")

    def size_of_model(self) -> float:
        """Returns size of model in Mb."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1e6

    def number_of_params(self) -> int:
        """Returns number of model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def _read_image(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        return self.val_dl.dataset.transform(image)

    def train_loop(self) -> Dict[str, float]:
        """Have to implement the training method of the model and return train_scores."""
        return {}

    def val_loop(self) -> Dict[str, float]:
        """Have to implement the validation method of the model and return val_scores."""
        return {}

    def _predict_on_image(self, image: torch.Tensor, proba: bool):
        """Have to implement the prediction method on single image."""
        pass

    def __predict_from_folder(self, image_folder: str, proba: bool) -> Dict:
        predicts = {}
        for image_name in tqdm(os.listdir(image_folder)):
            image_path = os.path.join(image_folder, image_name)
            image = self._read_image(image_path)
            predicts[image_name] = self._predict_on_image(image=image, proba=proba)
        return predicts

    def __predict_from_array(self, image_folder: np.ndarray, proba: bool) -> Dict:
        predicts = {}
        array_iter = enumerate(image_folder)
        for idx, array in tqdm(array_iter):
            image = torch.from_numpy(array).float()
            predicts[str(idx)] = self._predict_on_image(image=image, proba=proba)
        return predicts

    def _predict(self, image_folder: Union[str, np.ndarray], proba: bool) -> Dict:
        """Generalized prediction method of the model."""
        if type(image_folder) == str:
            return self.__predict_from_folder(image_folder, proba)
        else:
            return self.__predict_from_array(image_folder, proba)

    def predict(self, image_folder: Union[str, np.ndarray]) -> Dict:
        """Computes predictions for images in image_folder.

        Args:
            image_folder: Image folder path."""
        return self._predict(image_folder=image_folder, proba=False)

    def predict_proba(self, image_folder: Union[str, np.ndarray]) -> Dict:
        """Computes probability predictions for images in image_folder.

        Args:
            image_folder: Image folder path.
        """
        return self._predict(image_folder=image_folder, proba=True)

    def get_optimizable_module(self) -> torch.nn.Module:
        """Returns the module for optimization applying."""
        return self.model

    def write_scores(
            self,
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

    def fit(self, num_epochs: int) -> None:
        """Run optimization experiment.

        Args:
            num_epochs: Number of epochs.
        """
        writer = SummaryWriter(os.path.join(self.summary_path, self.name, 'train'))
        print(f"{self.name}, using device: {self.device}")
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}")
            train_scores = self.train_loop()
            self.write_scores(writer, 'train', train_scores, epoch)
            self.structure_optimization.optimize_during_training()
            val_scores = self.val_loop()
            self.write_scores(writer, 'val', val_scores, epoch)
            if val_scores[self.metric] > self.best_score:
                self.best_score = val_scores[self.metric]
                print(f'Best score - {self.best_score}')
                self.save_model_state_dict()
        self.structure_optimization.final_optimize()
        writer.close()

    def finetune(self, num_epochs: int, name: str = 'fine-tuning') -> None:
        """Run fine-tuning

        Args:
            num_epochs: Number of epochs.
            name: Description of the experiment (default: ``'fine-tuning'``).
        """
        best_score = 0
        writer = SummaryWriter(os.path.join(self.summary_path, self.name, name))
        print(f"{self.name}/{name}, using device: {self.device}")
        for epoch in range(1, num_epochs + 1):
            print(f"Fine-tuning epoch {epoch}")
            train_scores = self.train_loop()
            self.write_scores(writer, 'fine-tuning_train', train_scores, epoch)
            val_scores = self.val_loop()
            self.write_scores(writer, 'fine-tuning_val', val_scores, epoch)
            if val_scores[self.metric] > best_score:
                best_score = val_scores[self.metric]
                self.save_model_state_dict(name=name)
        self.load_model_state_dict(name=name)
        writer.close()


class ClassificationExperimenter(_GeneralizedExperimenter):
    """Class for working with classification models.

    Args:
        dataset_name: Name of dataset.
        train_dataset: Train dataset.
        val_dataset: Validation dataset.
        num_classes: Number of classes in the dataset.
        dataloader_params: Parameter dictionary passed to dataloaders.
        model: Name of model.
        model_params: Parameter dictionary passed to model initialization.
        models_saving_path: Path to folder for saving models.
        optimizer: Model optimizer, e.g. ``torch.optim.Adam``.
        optimizer_params: Parameter dictionary passed to optimizer initialization.
        target_loss: Loss function applied to model output,
            e.g. ``torch.nn.CrossEntropyLoss``.
        loss_params: Parameter dictionary passed to loss initialization.
        structure_optimization: Structure optimizer, e.g. ``SVDOptimization``.
        structure_optimization_params: Parameter dictionary passed to structure
            optimization initialization.
        metric: Target metric by which models are compared. May be ``'f1'``,
            ``'accuracy'``, ``'precision'``, ``'recall'`` or ``'roc_auc'``
            (default: ``'f1'``).
        summary_path: Path to folder for writing experiment summary info
            (default: ``'runs'``).
        summary_per_class: If ``True``, calculates the metrics for each class
            (default ``False``).
        weights: Path to the model state_dict to load weights (default: ``None``).
        prefix: An explanatory string added to the name of the experiment.
        gpu: If ``True``, uses GPU (default: ``True``).

        Raises:
            ValueError: If ``model``, ``structure_optimization``, ``metric``
                or ``dataset_name`` not in valid values.
    """

    def __init__(
        self,
        dataset_name: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_classes: int,
        dataloader_params: Dict,
        model: str,
        model_params: Dict,
        models_saving_path: str,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict,
        target_loss: Type[torch.nn.Module],
        loss_params: Dict,
        structure_optimization: str,
        structure_optimization_params: Dict,
        metric: str = 'f1',
        summary_path: str = 'runs',
        summary_per_class: bool = False,
        weights: Optional[str] = None,
        prefix: str = '',
        gpu: bool = True,
    ) -> None:

        _parameter_value_check(
            parameter='model', value=model, valid_values=set(CLF_MODELS.keys())
        )
        _parameter_value_check(
            parameter='structure_optimization',
            value=structure_optimization,
            valid_values=set(OPTIMIZATIONS.keys()),
        )
        _parameter_value_check(
            parameter='metric',
            value=metric,
            valid_values={'f1', 'accuracy', 'precision', 'recall', 'roc_auc'},
        )

        super().__init__(
            model=CLF_MODELS[model](num_classes=num_classes, **model_params),
            optimizable_module_name=model,
            train_ds=train_dataset,
            val_ds=val_dataset,
            num_classes=num_classes,
            dataloader_params=dataloader_params,
            name=f"{dataset_name}/{prefix}{model}",
            models_path=models_saving_path,
            summary_path=summary_path,
            summary_per_class=summary_per_class,
            metric=metric,
            gpu=gpu,
        )

        self.structure_optimization = OPTIMIZATIONS[structure_optimization](
            experimenter=self, **structure_optimization_params
        )
        self.target_loss = target_loss(**loss_params)
        self.model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)

    def train_loop(self) -> (Dict[str, float]):
        """Training method of the model.

        Returns:
            Dictionary {metric_name: value}.
        """
        self.model.train()
        train_scores = {'accuracy': 0, 'loss': 0}
        for key in self.structure_optimization.losses:
            train_scores[key] = 0
        for x, y in tqdm(self.train_dl):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            loss = self.target_loss(pred, y)
            train_scores['loss'] += loss.item()
            train_scores['accuracy'] += (
                (pred.argmax(1) == y).type(torch.float).mean().item()
            )
            for key, loss_fn in self.structure_optimization.losses.items():
                opt_loss = loss_fn(self.get_optimizable_module())
                loss += opt_loss
                train_scores[key] += opt_loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for key in train_scores:
            train_scores[key] /= len(self.train_dl)
        return train_scores

    def val_loop(self) -> Dict[Union[str, int], float]:
        """Validation method of the model. Returns val_scores

        Returns:
            Dictionary {metric_name: value}.
        """
        self.model.eval()
        val_loss = 0
        y_true = []
        y_pred = []
        y_score = []
        start = time.time()
        with torch.no_grad():
            for x, y in tqdm(self.val_dl):
                y_true.extend(y)
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                y_pred.extend(pred.cpu().argmax(1))
                y_score.extend(softmax(pred, dim=1).tolist())
                val_loss += self.target_loss(pred, y).item()
        total_time = time.time() - start
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        val_scores = {
            'loss': val_loss / len(self.val_dl),
            'inference_time': total_time / len(self.val_dl),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc_score(y_true, y_score, multi_class='ovo'),
        }
        if self.summary_per_class:
            f1s = f1_score(y_true, y_pred, average=None)
            val_scores.update({f'f1_class{i}': s for i, s in enumerate(f1s)})
        return val_scores

    def _predict_on_image(self, image: torch.Tensor, proba: bool) -> Union[List, int]:
        """Returns prediction for image."""
        self.model.eval()
        with torch.no_grad():

            try:
                pred = self.model(image)
            except Exception:
                image = image.unsqueeze_(0).to(self.device)
                pred = self.model(image)

            if proba:
                pred = softmax(pred, dim=1).cpu().detach().tolist()[0]
            else:
                pred = pred.argmax(1).cpu().detach().item()
        return pred


class FasterRCNNExperimenter(_GeneralizedExperimenter):
    """Class for working with Faster R-CNN model.

        Args:
        dataset_name: Name of dataset.
        train_dataset: Train dataset.
        val_dataset: Validation dataset.
        num_classes: Number of classes in the dataset.
        dataloader_params: Parameter dictionary passed to dataloaders.
        model_params: Parameter dictionary passed to ``fasterrcnn_resnet50_fpn``.
        models_saving_path: Path to folder for saving models.
        optimizer: Model optimizer, e.g. ``torch.optim.SGD``.
        optimizer_params: Parameter dictionary passed to optimizer initialization.
        scheduler_params: Parameter dictionary passed to ``StepLR`` initialization.
        structure_optimization: Structure optimizer, e.g. ``SVDOptimization``.
        structure_optimization_params: Parameter dictionary passed to structure
            optimization initialization.
        metric: Target metric by which models are compared. May be ``'map'``,
            ``'map_50'`` or ``'map_75'`` (default: ``'map_50'``).
        summary_path: Path to folder for writing experiment summary info
            (default: ``'runs'``).
        summary_per_class: If ``True``, calculates the metrics for each class
            (default ``False``).
        gpu: If ``True``, uses GPU (default: ``True``).

        Raises:
            ValueError: If ``model``, ``structure_optimization``, ``metric``
                or ``dataset_name`` not in valid values.
    """

    def __init__(
        self,
        dataset_name: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_classes: int,
        dataloader_params: Dict,
        model_params: Dict,
        models_saving_path: str,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict,
        scheduler_params: Dict,
        structure_optimization: str,
        structure_optimization_params: Dict,
        metric: str = 'map_50',
        summary_path: str = 'runs',
        summary_per_class: bool = False,
        weights: Optional[str] = None,
        prefix: str = '',
        gpu: bool = True,
    ) -> None:

        _parameter_value_check(
            parameter='structure_optimization',
            value=structure_optimization,
            valid_values=set(OPTIMIZATIONS.keys()),
        )
        _parameter_value_check(
            parameter='metric',
            value=metric,
            valid_values={'map', 'map_50', 'map_75'},
        )

        model = fasterrcnn_resnet50_fpn(**model_params)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        super().__init__(
            model=model,
            optimizable_module_name="ResNet50",
            train_ds=train_dataset,
            val_ds=val_dataset,
            num_classes=num_classes,
            dataloader_params=dataloader_params,
            name=f"{dataset_name}/{prefix}FasterR-CNN/ResNet50",
            models_path=models_saving_path,
            summary_path=summary_path,
            summary_per_class=summary_per_class,
            metric=metric,
            weights=weights,
            gpu=gpu,
        )
        self.structure_optimization = OPTIMIZATIONS[structure_optimization](
            experimenter=self, **structure_optimization_params
        )

        self.model.to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer(params, **optimizer_params)
        self.scheduler = StepLR(self.optimizer, **scheduler_params)

    def get_optimizable_module(self) -> torch.nn.Module:
        """Return the module for optimization applying."""
        return self.model.backbone

    def train_loop(self):
        """Training method of the model.

        Returns:
            Dictionary {metric_name: value}.
        """
        self.model.train()
        batches = tqdm(self.train_dl)
        train_scores = {'loss': 0}
        for key in self.structure_optimization.losses:
            train_scores[key] = 0
        i = 0
        for images, targets in batches:
            i += 1
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            losses = sum(loss_dict.values())
            train_scores['loss'] += losses.item()

            for key, loss_fn in self.structure_optimization.losses.items():
                opt_loss = loss_fn(self.get_optimizable_module())
                losses += opt_loss
                train_scores[key] += opt_loss.item()
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            batches.set_postfix({'loss': train_scores['loss'] / i})

        for key in train_scores:
            train_scores[key] /= len(self.train_dl)
        return train_scores

    def val_loop(self):
        """Validation method of the model.

        Returns:
            Dictionary {metric_name: value}.
        """
        metric = MeanAveragePrecision()
        tk = tqdm(self.val_dl)
        for images, targets in tk:
            images = list(image.to(self.device) for image in images)
            preds = self.forward(images)
            metric.update(preds, targets)
        return metric.compute()

    def _predict_on_image(self, image: torch.Tensor, proba: bool) -> Dict:
        """Returns prediction for image."""
        pred = self.forward([image])[0]
        if not proba:
            not_thresh = pred['scores'] > 0.5
            pred['boxes'] = pred['boxes'][not_thresh]
            pred['labels'] = pred['labels'][not_thresh]
            pred.pop('scores')
        for key in pred:
            pred[key] = pred[key].tolist()
        return pred

    def forward(self, images):
        """Predict model outputs from images.

        Args:
            images: List of image tensors.
        """
        self.model.eval()
        with torch.no_grad():
            images = list(image.to(self.device) for image in images)
            outputs = self.model(images)
            outputs = [{k: v.to('cpu').detach() for k, v in t.items()} for t in outputs]
        return outputs
