import math

import pytest
import torch

from core.metrics.cv_metrics import ClassificationMetricCounter, LossesAverager
from core.metrics.cv_metrics import iou_score, dice_score, SegmentationMetricCounter


@pytest.fixture()
def get_tensors():
    truth = torch.tensor([
        [
            [
                [0, 1, 1],
                [0, 1, 1],
                [1, 0, 0]
            ],
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 1]
            ]
        ],
        [
            [
                [1, 0, 1],
                [0, 0, 0],
                [1, 0, 1]
            ],
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        ]
    ])
    a = torch.tensor([
        [
            [
                [0.1, 0.9, 0.71],
                [0.4, 0.55, 0.91],
                [0.51, 0.2, 0.77]
            ],
            [
                [0.65, 0.32, 0.2],
                [0.62, 0.11, 0.4],
                [0.1, 0, 0.4]
            ],
            [
                [0.12, 0.47, 0.49],
                [0.25, 0.27, 0.47],
                [0.1, 0.51, 0.2]
            ]
        ],
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        ]
    ])
    b = torch.tensor([
        [
            [
                [0, 0, 1],
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        ],
        [
            [
                [1, 0, 1],
                [0, 0, 0],
                [1, 1, 1]
            ],
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ]
        ]
    ])
    yield truth, a, b


def test_iou(get_tensors):
    truth, a, b = get_tensors
    assert torch.allclose(iou_score(a, truth), torch.tensor([[5 / 6, 1, 0.5], [0, 5 / 9, 1]]))
    assert torch.allclose(iou_score(b, truth), torch.tensor([[0.5, 1 / 3, 0], [0.8, 0.6, 0]]))


def test_dice(get_tensors):
    truth, a, b = get_tensors
    assert torch.allclose(dice_score(a, truth), torch.tensor([[10 / 11, 1, 2 / 3], [0, 5 / 7, 1]]))
    assert torch.allclose(dice_score(b, truth), torch.tensor([[2 / 3, 0.5, 0], [8 / 9, 3 / 4, 0]]))


def test_segmentation_metric_counter(get_tensors):
    answer1 = {
        'iou': 0.510185,
        'dice': 0.5913,
        'iou_for_class_0': 0.533333,
        'iou_for_class_1': 0.622222,
        'iou_for_class_2': 0.375,
        'dice_for_class_0': 0.616162,
        'dice_for_class_1': 0.741071,
        'dice_for_class_2': 0.416666,
    }
    answer2 = {
        'iou': 0.464197,
        'dice': 0.550064,
        'iou_for_class_0': 0.572222,
        'iou_for_class_1': 0.570370,
        'iou_for_class_2': 0.25,
        'dice_for_class_0': 0.670034,
        'dice_for_class_1': 0.702380,
        'dice_for_class_2': 0.277777,
    }
    truth, a, b = get_tensors
    truth = truth.argmax(1)
    counter = SegmentationMetricCounter(class_metrics=True)
    counter.update(a, truth)
    counter.update(b, truth)
    scores = counter.compute()
    assert set(scores.keys()) == set(answer1.keys())
    for k, v in answer1.items():
        assert math.isclose(scores[k], v, rel_tol=1e-5)
    counter.update(b, truth)
    scores = counter.compute()
    assert set(scores.keys()) == set(answer2.keys())
    for k, v in answer2.items():
        assert math.isclose(scores[k], v, rel_tol=1e-5)


def test_classification_metric_counter():
    target1 = torch.tensor([0, 2])
    target2 = torch.tensor([1, 2])
    answer = {
        'accuracy': 0.75,
        'precision': 2.5 / 3,
        'recall': 2.5 / 3,
        'f1': 7 / 9,
        'roc_auc': 0.75
    }
    a = torch.tensor([[0.6, 0.2, 0.2],
                      [0.1, 0.7, 0.2]])
    b = torch.tensor([[0.3, 0.5, 0.2],
                      [0.1, 0.1, 0.8]])
    counter = ClassificationMetricCounter()
    counter.update(a, target1)
    counter.update(b, target2)
    scores = counter.compute()
    assert set(scores.keys()) == set(answer.keys())
    for k, v in answer.items():
        print(k)
        assert math.isclose(scores[k], v, rel_tol=1e-5)


def test_losses_averager():
    averager = LossesAverager()
    averager.update({'loss': torch.tensor([5.5])})
    averager.update({'loss': torch.tensor([2.25])})
    losses = averager.compute()
    assert set(losses.keys()) == {'loss'}
    assert math.isclose(losses['loss'], 3.875)
    averager.update({'loss': torch.tensor([1.75])})
    losses = averager.compute()
    assert set(losses.keys()) == {'loss'}
    assert math.isclose(losses['loss'], 3.1666666666666665)
