"""This module contains functions and classes for computing metrics
 in computer vision tasks.
 """
from typing import Dict

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.nn.functional import softmax


def iou_score(outputs: torch.Tensor, masks: torch.Tensor, smooth=1e-10) -> torch.Tensor:
    """Computes intersection over union (masks) on batch.

    Args:
        outputs: Output from semantic segmentation model.
        masks: True masks.
        smooth: Additional constant to avoid division by zero.
    """
    outputs = (outputs > 0.5).float()
    intersection = torch.logical_and(outputs, masks).float().sum((2, 3))
    union = torch.logical_or(outputs, masks).float().sum((2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou


def dice_score(outputs: torch.Tensor, masks: torch.Tensor, smooth=1e-10) -> torch.Tensor:
    """Computes dice coefficient (masks) on batch.

    Args:
        outputs: Output from semantic segmentation model.
        masks: True masks.
        smooth: Additional constant to avoid division by zero.
    """
    outputs = (outputs > 0.5).float()
    intersection = torch.logical_and(outputs, masks).float().sum((2, 3))
    total = (outputs + masks).sum((2, 3))
    dice = (2 * intersection + smooth) / (total + smooth)
    return dice


class MetricCounter:
    """Generalized class for calculating metrics"""

    def __init__(self) -> None:
        pass

    def update(self, **kwargs) -> None:
        """Have to implement updating, taking model outputs as input."""
        raise NotImplementedError

    def compute(self) -> Dict[str, float]:
        """Have to implement computing of metrics."""
        raise NotImplementedError


class ClassificationMetricCounter(MetricCounter):
    """Calculates metrics for classification task.

    Args:
        class_metric:  If ``True``, calculates metrics for each class.
    """

    def __init__(self, class_metrics: bool = False) -> None:
        super().__init__()
        self.y_true = []
        self.y_pred = []
        self.y_score = []
        self.class_metrics = class_metrics

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulates predictions and targets"""
        self.y_true.extend(targets.tolist())
        self.y_score.extend(softmax(predictions, dim=1).tolist())
        self.y_pred.extend(predictions.argmax(1).tolist())

    def compute(self) -> Dict[str, float]:
        """Returns metrics."""
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


class SegmentationMetricCounter(MetricCounter):
    """Calculates metrics for semantic segmentation task.

    Args:
        class_metric:  If ``True``, calculates metrics for each class.
    """

    def __init__(self, class_metrics: bool = False) -> None:
        super().__init__()
        self.iou = None
        self.dice = None
        self.n = 0
        self.class_metrics = class_metrics

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulates iou and dice"""
        masks = torch.zeros_like(predictions)
        for i in range(predictions.size()[1]):
            masks[:, i, :, :] = targets == i
        self.n += predictions.size()[0]
        if self.iou is None:
            self.iou = iou_score(predictions, masks).sum(0)
        else:
            self.iou += iou_score(predictions, masks).sum(0)
        if self.dice is None:
            self.dice = dice_score(predictions, masks).sum(0)
        else:
            self.dice += dice_score(predictions, masks).sum(0)
    def compute(self) -> Dict[str, float]:
        """Returns average metrics."""
        iou = self.iou / self.n
        dice = self.dice/ self.n
        scores = {'iou': iou.mean().item(), 'dice': dice.mean().item()}
        if self.class_metrics:
            scores.update({f'iou_for_class_{i}': s.item() for i, s in enumerate(iou)})
            scores.update({f'dice_for_class_{i}': s.item() for i, s in enumerate(dice)})
        return scores


class LossesAverager(MetricCounter):
    """Calculates the average loss."""

    def __init__(self) -> None:
        super().__init__()
        self.losses = None
        self.counter = 0

    def update(self, losses: Dict[str, torch.Tensor]) -> None:
        """Accumulates losses"""
        self.counter += 1
        if self.losses is None:
            self.losses = {k: v.item() for k, v in losses.items()}
        else:
            for key, value in losses.items():
                self.losses[key] += value.item()

    def compute(self) -> Dict[str, float]:
        """Returns average losses."""
        return {k: v / self.counter for k, v in self.losses.items()}
