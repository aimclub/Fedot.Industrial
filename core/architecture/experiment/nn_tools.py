import os
import shutil
import time
from typing import Dict, List, Optional, Callable, Union

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


class _GeneralizedNNModel:

    def __init__(
            self,
            model: torch.nn.Module,
            name: Optional[str] = None,
            gpu: bool = True,
    ):
        self.model = model
        self.device = torch.device('cuda' if gpu else 'cpu')
        self.model.to(self.device)
        self.name = name if name is not None else type(model).__name__

    def save_model(
            self,
            file_name: str,
            dir_path: str = 'models',
            state_dict: bool = True,
    ) -> None:
        """Save the model or its state dict.

        Args:
            file_name: File name without extension.
            dir_path: Path to the directory where the model will be saved.
            state_dict: If ``True`` save state_dict with extension ".sd.pt",
                else save all model with extension ".model.pt".
        """
        os.makedirs(dir_path, exist_ok=True)
        file_name = f"{file_name}.{'sd' if state_dict else 'model'}.pt"
        file_path = os.path.join(dir_path, file_name)
        data = self.model.state_dict() if state_dict else self.model
        try:
            torch.save(data, file_path)
        except Exception:
            torch.save(data, file_name)
            shutil.move(file_name, dir_path)
        print(f"Saved to {os.path.abspath(file_path)}.")

    def load_model(
            self,
            file_name: str,
            dir_path: str = 'models',
            state_dict: bool = True,
    ) -> None:
        """Load the model or its state dict.

        Args:
            file_name: File name without extension.
            dir_path: Path to the directory with the model.
            state_dict: If ``True`` load state_dict with extension ".sd.pt",
                else load all model with extension ".model.pt".
        """
        file_name = f"{file_name}.{'sd' if state_dict else 'model'}.pt"
        file_path = os.path.join(dir_path, file_name)
        data = torch.load(file_path)
        if state_dict:
            self.model.load_state_dict(data)
            self.model.to(self.device)
            print("Model state dict loaded.")
        else:
            self.model = data
            print("Model loaded.")

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

    def train_loop(
            self,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            model_losses: Optional[Dict[str, Callable]]
    ) -> Dict[str, float]:
        """Have to implement the training method of the model and return train_scores."""
        return {}

    def val_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            summary_per_class: bool = False,
    ) -> Dict[str, float]:
        """Have to implement the validation method of the model and return val_scores."""
        return {}

    def predict(self, sample: torch.Tensor, proba: bool):
        """Have to implement the prediction method on single sample."""
        pass


class ClassificationModel(_GeneralizedNNModel):

    def __init__(
            self,
            model: torch.nn.Module,
            loss: Callable = torch.nn.CrossEntropyLoss,
            name: Optional[str] = None,
            gpu: bool = True,
    ):
        super().__init__(
            model=model,
            name=name,
            gpu=gpu
        )
        self.loss = loss

    def train_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            model_losses: Optional[Dict[str, Callable]]
    ) -> Dict[str, float]:
        """Training method of the model.

        Returns:
            Dictionary {metric_name: value}.
        """
        self.model.train()
        train_scores = {'accuracy': 0, 'loss': 0}
        for key in model_losses:
            train_scores[key] = 0
        for x, y in tqdm(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            loss = self.loss(pred, y)
            train_scores['loss'] += loss.item()
            train_scores['accuracy'] += (
                (pred.argmax(1) == y).type(torch.float).mean().item()
            )
            for key, loss_fn in model_losses.items():
                opt_loss = loss_fn(self.model)
                loss += opt_loss
                train_scores[key] += opt_loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for key in train_scores:
            train_scores[key] /= len(dataloader)
        return train_scores

    def val_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            summary_per_class: bool = False,
    ) -> Dict[str, float]:
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
            for x, y in tqdm(dataloader):
                y_true.extend(y)
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                y_pred.extend(pred.cpu().argmax(1))
                y_score.extend(softmax(pred, dim=1).tolist())
                val_loss += self.loss(pred, y).item()
        total_time = time.time() - start
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        val_scores = {
            'loss': val_loss / len(dataloader),
            'inference_time': total_time / len(dataloader),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc_score(y_true, y_score, multi_class='ovo'),
        }
        if summary_per_class:
            f1s = f1_score(y_true, y_pred, average=None)
            val_scores.update({f'f1_class{i}': s for i, s in enumerate(f1s)})
        return val_scores

    def predict(self, sample: torch.Tensor, proba: bool) -> Union[List, int]:
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


class FasterRCNNModel(_GeneralizedNNModel):

    def __init__(
            self,
            num_classes: int,
            model_params: Dict = {},
            name: Optional[str] = None,
            gpu: bool = True,
    ):
        model = fasterrcnn_resnet50_fpn(**model_params)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        super().__init__(
            model=model,
            name=name,
            gpu=gpu
        )

    def train_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            model_losses: Optional[Dict[str, Callable]]
    ) -> Dict[str, float]:
        """Training method of the model.

        Returns:
            Dictionary {metric_name: value}.
        """
        self.model.train()
        tk = tqdm(dataloader)
        train_scores = {'loss': 0}
        for key in model_losses:
            train_scores[key] = 0
        i = 0
        for images, targets in tk:
            i += 1
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            losses = sum(loss_dict.values())
            train_scores['loss'] += losses.item()

            for key, loss_fn in model_losses.items():
                opt_loss = loss_fn(self.model)
                losses += opt_loss
                train_scores[key] += opt_loss.item()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            tk.set_postfix({'loss': train_scores['loss'] / i})

        for key in train_scores:
            train_scores[key] /= len(dataloader)
        return train_scores

    def val_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            summary_per_class: bool = False,
    ) -> Dict[str, float]:
        """Validation method of the model. Returns val_scores

        Returns:
            Dictionary {metric_name: value}.
        """
        self.model.eval()
        metric = MeanAveragePrecision()
        tk = tqdm(dataloader)
        with torch.no_grad():
            for images, targets in tk:
                images = list(image.to(self.device) for image in images)
                preds = self.model(images)
                preds = [{k: v.to('cpu').detach() for k, v in t.items()} for t in preds]
                metric.update(preds, targets)
        return metric.compute()

    def predict(self, sample: torch.Tensor, proba: bool) -> Dict:
        """Returns prediction for sample."""
        self.model.eval()
        with torch.no_grad():
            sample = sample.to(self.device)
            pred = self.model(sample)
        if not proba:
            not_thresh = pred['scores'] > 0.5
            pred['boxes'] = pred['boxes'][not_thresh]
            pred['labels'] = pred['labels'][not_thresh]
            pred.pop('scores')
        pred = {k: v.to('cpu').detach().tolist() for k, v in pred.items()}
        return pred