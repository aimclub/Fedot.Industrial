from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_sample(
        image: torch.Tensor,
        target: Optional[Dict[str, torch.Tensor]] = None,
        prediction: Optional[Dict[str, torch.Tensor]] = None,
        threshold: float = 0.5
) -> plt.Figure:
    """
    Returns the image with bounding boxes.

    Args:
        image: image tensor.
        target: Dictionary of target values with keys ``'boxes'`` and ``'labels'``.
        prediction: Dictionary of predicted values with keys ``'boxes'``, ``'labels'`` and ``'scores'``.
        threshold: Confidence threshold for displaying predicted bounding boxes.

    Returns:
        `matplotlib.pyplot.Figure` of the image with bounding boxes.
    """
    assert prediction is not None or target is not None, "At least one parameter from 'target' and 'prediction' must not be None"
    image = image.permute(1, 2, 0).numpy()
    n = 1 if prediction is None or target is None else 2
    fig = plt.figure(figsize=(10 * n, 10))

    if target is not None:
        ax = plt.subplot(1, n, 1)
        boxes = target['boxes'].numpy().astype(np.int32)
        labels = target['labels'].numpy().astype(str)
        timage = image.copy()
        for box, label in zip(boxes, labels):
            cv2.rectangle(timage, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 3)
            cv2.putText(timage, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
        ax.set_axis_off()
        ax.imshow(timage)

    if prediction is not None:
        ax = plt.subplot(1, n, n)
        not_thresh = prediction['scores'] > threshold
        prediction['boxes'] = prediction['boxes'][not_thresh]
        prediction['labels'] = prediction['labels'][not_thresh]
        prediction['scores'] = prediction['scores'][not_thresh]
        boxes = prediction['boxes'].numpy().astype(np.int32)
        labels = prediction['labels'].numpy()
        scores = prediction['scores'].numpy()
        pimage = image.copy()
        for box, label, score in zip(boxes, labels, scores):
            cv2.rectangle(pimage, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 3)
            cv2.putText(pimage, f'{label} ({score:.2f})', (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
        ax.set_axis_off()
        ax.imshow(pimage)
    return fig
