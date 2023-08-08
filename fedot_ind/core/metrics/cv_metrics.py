"""This module contains functions and classes for computing metrics
 in computer vision tasks.
 """
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.nn.functional import softmax
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def iou_score(
        outputs: torch.Tensor,
        masks: torch.Tensor,
        threshold: float = 0.5,
        smooth: float = 1e-10
) -> torch.Tensor:
    """Computes intersection over union (masks) on batch.

    Args:
        outputs: Output from semantic segmentation model.
        masks: True masks.
        threshold: Binarization threshold for output.
        smooth: Additional constant to avoid division by zero.

    Returns:
        Intersection over union for batch.
    """
    outputs = (outputs > threshold).float()
    intersection = torch.logical_and(outputs, masks).float().sum((2, 3))
    union = torch.logical_or(outputs, masks).float().sum((2, 3))
    iou = (intersection + smooth) / (union + smooth)
    iou[union == 0] = -1
    return iou


def dice_score(
        outputs: torch.Tensor,
        masks: torch.Tensor,
        threshold: float = 0.5,
        smooth: float = 1e-10
) -> torch.Tensor:
    """Computes dice coefficient (masks) on batch.

    Args:
        outputs: Output from semantic segmentation model.
        masks: True masks.
        threshold: Binarization threshold for output.
        smooth: Additional constant to avoid division by zero.

    Returns:
        Dice for batch.
    """
    outputs = (outputs > threshold).float()
    intersection = torch.logical_and(outputs, masks).float().sum((2, 3))
    total = (outputs + masks).sum((2, 3))
    dice = (2 * intersection + smooth) / (total + smooth)
    dice[total == 0] = -1
    return dice


class MetricCounter(ABC):
    """Generalized class for calculating metrics"""

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Have to implement updating, taking model outputs as input."""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Have to implement metrics computing."""
        raise NotImplementedError


class ClassificationMetricCounter(MetricCounter):
    """Calculates metrics for classification task.

    Args:
        class_metrics:  If ``True``, calculates metrics for each class.
    """

    def __init__(self, class_metrics: bool = False) -> None:
        super().__init__()
        self.y_true = []
        self.y_pred = []
        self.y_score = []
        self.class_metrics = class_metrics

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulates predictions and targets."""
        self.y_true.extend(targets.tolist())
        self.y_score.extend(softmax(predictions, dim=1).tolist())
        self.y_pred.extend(predictions.argmax(1).tolist())

    def compute(self) -> Dict[str, float]:
        """Compute accuracy, precision, recall, f1, roc auc metrics.

         Returns:
              Dictionary: `{metric: score}`.
        """
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
        class_metrics:  If ``True``, calculates metrics for each class.
    """

    def __init__(self, class_metrics: bool = False) -> None:
        super().__init__()
        self.iou = []
        self.dice = []
        self.class_metrics = class_metrics

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulates iou and dice."""
        masks = torch.zeros_like(predictions)
        for i in range(predictions.shape[1]):
            masks[:, i, :, :] = torch.squeeze(targets == i)
        self.iou.append(iou_score(predictions, masks))
        self.dice.append(dice_score(predictions, masks))

    def compute(self) -> Dict[str, float]:
        """Compute average metrics.

         Returns:
              Dictionary: `{metric: score}`.
        """
        iou = torch.cat(self.iou).T
        dice = torch.cat(self.dice).T

        scores = {
            'iou': iou[1:][iou[1:] >= 0].mean().item(),
            'dice': dice[1:][dice[1:] >= 0].mean().item()
        }
        if self.class_metrics:
            scores.update({f'iou_for_class_{i}': s[s >= 0].mean().item() for i, s in enumerate(iou)})
            scores.update({f'dice_for_class_{i}': s[s >= 0].mean().item() for i, s in enumerate(dice)})
        return scores


class ObjectDetectionMetricCounter(MetricCounter):
    """Calculates metrics for object detection task.

    Args:
        class_metrics:  If ``True``, calculates metrics for each class.
    """

    def __init__(self, class_metrics: bool = False) -> None:
        super().__init__()
        self.map = MeanAveragePrecision(class_metrics=class_metrics)
        self.class_metrics = class_metrics

    def update(
            self,
            predictions: List[Dict[str, torch.Tensor]],
            targets: List[Dict[str, torch.Tensor]]
    ) -> None:
        """Accumulates predictions and targets."""
        self.map.update(preds=predictions, target=targets)

    def compute(self) -> Dict[str, float]:
        """Compute MAP, MAR metrics.

         Returns:
              Dictionary: `{metric: score}`.
        """

        scores = self.map.compute()
        if self.class_metrics:
            scores.update({f'map_for_class_{i}': s for i, s in enumerate(scores['map_per_class'])})
            scores.update({f'mar_100_for_class_{i}': s for i, s in enumerate(scores['mar_100_per_class'])})
        del scores['map_per_class']
        del scores['mar_100_per_class']
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
        """Compute average losses.

        Returns:
            Dictionary: `{metric: score}`.
        """
        return {k: v / self.counter for k, v in self.losses.items()}
