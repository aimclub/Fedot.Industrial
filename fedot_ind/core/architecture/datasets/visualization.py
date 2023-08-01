from typing import Dict, Optional, Union, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def _2nparray(arrs: List[Union[torch.Tensor, List]]) -> List[np.ndarray]:
    return [arr.numpy() if isinstance(arr, torch.Tensor) else np.array(arr) for arr in arrs]


def draw_sample_with_bboxes(
        image: Union[torch.Tensor, str],
        target: Optional[Dict[str, Union[torch.Tensor, List]]] = None,
        prediction: Optional[Dict[str, Union[torch.Tensor, List]]] = None,
        threshold: float = 0.5
) -> plt.Figure:
    """
    Returns the image with bounding boxes.

    Args:
        image: image tensor or path to image.
        target: Dictionary of target values with keys ``'boxes'`` and ``'labels'``.
        prediction: Dictionary of predicted values with keys ``'boxes'``, ``'labels'`` and ``'scores'``.
        threshold: Confidence threshold for displaying predicted bounding boxes.

    Returns:
        `matplotlib.pyplot.Figure` of the image with bounding boxes.
    """
    assert prediction is not None or target is not None, "At least one parameter from 'target' and 'prediction' must not be None"

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    else:
        image = plt.imread(image)

    n = 1 if prediction is None or target is None else 2
    fig = plt.figure(figsize=(10 * n, 10))

    thickness = 1 + int(image.shape[-2] / 500)
    font_scale = image.shape[-2] / 1000

    if target is not None:
        ax = plt.subplot(1, n, 1)
        boxes, labels = _2nparray([target['boxes'], target['labels']])
        timage = image.copy()
        for box, label in zip(boxes.astype(np.int32), labels.astype(str)):
            cv2.rectangle(timage, (box[0], box[1]), (box[2], box[3]), (220, 255, 255), thickness)
            cv2.putText(timage, label, (box[0], box[1]), 0, font_scale, (255, 255, 255), thickness)
        ax.set_axis_off()
        ax.imshow(timage)

    if prediction is not None:
        ax = plt.subplot(1, n, n)
        boxes, labels, scores = _2nparray([prediction['boxes'], prediction['labels'], prediction['scores']])

        not_thresh = scores > threshold
        boxes = boxes[not_thresh]
        labels = labels[not_thresh]
        scores = scores[not_thresh]

        pimage = image.copy()
        for box, label, score in zip(boxes.astype(np.int32), labels, scores):
            cv2.rectangle(pimage, (box[0], box[1]), (box[2], box[3]), (220, 255, 255), thickness)
            cv2.putText(pimage, f'{label} ({score:.2f})', (box[0], box[1]), 0, font_scale, (255, 255, 255), thickness)
        ax.set_axis_off()
        ax.imshow(pimage)
    return fig
