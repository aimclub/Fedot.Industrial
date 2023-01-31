from typing import Dict

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.nn.functional import softmax


def iou_score(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-10) -> torch.Tensor:
    intersection = torch.logical_and(outputs, labels).float().sum((2, 3))
    union = torch.logical_or(outputs, labels).float().sum((2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou


def dice_score(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-10) -> torch.Tensor:
    intersection = torch.logical_and(outputs, labels).float().sum((2, 3))
    sum = (outputs + labels).sum((2, 3))
    dice = (2 * intersection + smooth) / (sum + smooth)
    return dice


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

    def compute(self) -> Dict[str, float]:
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


class SegmentationMetricCounter:

    def __init__(self, class_metrics: bool = False) -> None:
        self.iou = None
        self.dice = None
        self.n = 0
        self.class_metrics = class_metrics

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self.n += predictions.size()[0]
        if self.iou is None:
            self.iou = iou_score(predictions, targets).sum(0)
        else:
            self.iou += iou_score(predictions, targets).sum(0)
        if self.dice is None:
            self.dice = dice_score(predictions, targets).sum(0)
        else:
            self.dice += dice_score(predictions, targets).sum(0)
    def compute(self) -> Dict[str, float]:
        iou = self.iou / self.n
        dice = self.dice/ self.n
        scores = {'iou': iou.mean().item(), 'dice': dice.mean().item()}
        if self.class_metrics:
            scores.update({f'iou_for_class_{i}': s.item() for i, s in enumerate(iou)})
            scores.update({f'dice_for_class_{i}': s.item() for i, s in enumerate(dice)})
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

    def compute(self) -> Dict[str, float]:
        return {k: v / self.counter for k, v in self.losses.items()}