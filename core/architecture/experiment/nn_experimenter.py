import os
import shutil
from typing import Dict, List, Optional, Callable, Union, Type

import torch
from fedot.core.log import default_log as Logger
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from core.architecture.abstraction.сheckers import parameter_value_check
from core.operation.optimization.structure_optimization import StructureOptimization

DEFAULT_PARAMS = {
    'FasterRCNNExperimenter':{
        'optimizer_params': {'lr': 0.0001},
        'dataloader_params': {'collate_fn': lambda x: tuple(zip(*x))}
    },
    'ClassificationExperimenter': {
        'optimizer_params': {},
        'dataloader_params': {}
    }
}
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


class ClassificationMetricCounter:

    def __init__(self, class_metrics: bool = False) -> None:
        self.y_true = []
        self.y_pred = []
        self.y_score = []
        self.class_metrics = class_metrics

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self.y_true.extend(targets.tolist())
        self.y_score.extend(softmax(predictions, dim=1).tolist())
        self.y_pred.extend(predictions.argmax(1).tolist())

    def compute(self):
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='macro'
        )
        scores = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc_score(self.y_true, self.y_score, multi_class='ovo'),
        }
        if self.class_metrics:
            f1s = f1_score(self.y_true, self.y_pred, average=None)
            scores.update({f'f1_for_class_{i}': s for i, s in enumerate(f1s)})
        return scores


class LossesAverager:

    def __init__(self) -> None:
        self.losses = None
        self.counter = 0

    def update(self, losses: Dict[str, torch.Tensor]) -> None:
        self.counter += 1
        if self.losses is None:
            self.losses = {k: v.item() for k, v in losses.items()}
        else:
            for key, value in losses.items():
                self.losses[key] += value.item()

    def compute(self):
        return {k: v / self.counter for k, v in self.losses.items()}


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
            dataset_name: str,
            train_dataset: Dataset,
            val_dataset: Dataset,
            num_epochs: int,
            optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
            structure_optimization: Optional[StructureOptimization] = None,
            models_path: str = 'models',
            summary_path: str = 'summary',
            class_metrics: bool = False,
            **parameters
    ) -> None:
        """Run experiment.

        Args:
            num_epochs: Number of epochs.
        """
        params = DEFAULT_PARAMS[self.__class__.__name__]
        for k, v in parameters.items():
            params[k].update(v)
        train_dl = DataLoader(train_dataset, shuffle=True, **params['dataloader_params'])
        val_dl = DataLoader(val_dataset, shuffle=False, **params['dataloader_params'])
        description = f"{dataset_name}/{self.name}"

        if structure_optimization is None:
            self.logger.info(f"{description}, using device: {self.device}")
            self.run_epochs(
                writer=SummaryWriter(os.path.join(summary_path, description, 'train')),
                train_dl=train_dl,
                val_dl=val_dl,
                num_epochs=num_epochs,
                optimizer=optimizer(self.model.parameters(), **params['optimizer_params']),
                model_path=os.path.join(models_path, description, 'trained'),
                class_metrics=class_metrics
            )
        else:
            structure_optimization.fit(
                exp=self,
                description=description,
                train_dl=train_dl,
                val_dl=val_dl,
                num_epochs=num_epochs,
                optimizer=optimizer,
                models_path=models_path,
                summary_path=summary_path,
                class_metrics=class_metrics,
                optimizer_params=params['optimizer_params']
            )

    def run_epochs(
            self,
            writer: SummaryWriter,
            train_dl: DataLoader,
            val_dl: DataLoader,
            num_epochs: int,
            optimizer: torch.optim.Optimizer,
            model_path: str,
            class_metrics: bool,
            start_epoch: int = 1,
            model_losses: Optional[Callable] = None
    ) -> None:
        """Run experiment.

        Args:
            num_epochs: Number of epochs.
        """
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.logger.info(f"Epoch {epoch}")
            train_scores = self.train_loop(
                dataloader=train_dl,
                optimizer=optimizer,
                model_losses=model_losses
            )
            write_scores(writer, 'train', train_scores, epoch)
            val_scores = self.val_loop(dataloader=val_dl, class_metrics=class_metrics)
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

    def predict_on_sample(self, sample: torch.Tensor, proba: bool):
        """Have to implement the prediction method on single sample."""
        pass

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

    def predict_on_sample(self, sample: torch.Tensor, proba: bool) -> Union[List, int]:
        """Returns prediction for sample."""
        self.model.eval()
        with torch.no_grad():
            sample = sample.to(self.device)
            pred = self.model(sample)
            if proba:
                pred = softmax(pred, dim=1).cpu().detach().tolist()[0]
            else:
                pred = pred.argmax(1).cpu().detach().item()
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
        assert self.model.training, "model in eval mode"
        images = [image.to(self.device) for image in x]
        targets = [{k: v.to(self.device) for k, v in target.items()} for target in y]
        return self.model(images, targets)

    def predict_on_sample(self, sample: torch.Tensor, proba: bool) -> Dict:
        """Returns prediction for sample."""
        self.model.eval()
        with torch.no_grad():
            sample = sample.to(self.device)
            preds = self.model(sample)
        if not proba:
            not_thresh = preds['scores'] > 0.5
            preds['boxes'] = preds['boxes'][not_thresh]
            preds['labels'] = preds['labels'][not_thresh]
            preds.pop('scores')
        pred = {k: v.tolist() for k, v in preds.items()}
        return pred
