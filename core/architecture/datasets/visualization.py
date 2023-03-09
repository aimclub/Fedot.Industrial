import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_sample(image, target, pred=None, thresh=0.5):
    boxes = target['boxes'].numpy().astype(np.int32)
    labels = target['labels'].numpy().astype(np.str)
    image = image.permute(1, 2, 0).numpy()
    timage = image.copy()

    n = 1 if pred is None else 2
    fig = plt.figure(figsize=(10 * n, 10))
    ax = plt.subplot(1, n, 1)

    for box, label in zip(boxes, labels):
        cv2.rectangle(timage, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 3)
        cv2.putText(timage, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
    ax.set_axis_off()
    ax.imshow(timage)

    if pred is not None:
        ax2 = plt.subplot(1, n, n)
        not_thresh = pred['scores'] > thresh
        pred['boxes'] = pred['boxes'][not_thresh]
        pred['labels'] = pred['labels'][not_thresh]
        pred['scores'] = pred['scores'][not_thresh]
        boxes = pred['boxes'].numpy().astype(np.int32)
        labels = pred['labels'].numpy()
        scores = pred['scores'].numpy()
        pimage = image.copy()
        for box, label, score in zip(boxes, labels, scores):
            cv2.rectangle(pimage, (box[0], box[1]), (box[2], box[3]), (220, 0, 0), 3)
            cv2.putText(pimage, '{} ({:.2f})'.format(label, score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 255), 2)
        ax2.set_axis_off()
        ax2.imshow(timage)
    return fig