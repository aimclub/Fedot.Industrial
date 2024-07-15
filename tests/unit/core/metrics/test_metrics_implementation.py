import pytest

import numpy as np
import pandas as pd

from fedot_ind.core.metrics.anomaly_detection.function import filter_detecting_boundaries, check_errors, \
    confusion_matrix
from fedot_ind.core.metrics.metrics_implementation import ParetoMetrics, calculate_forecasting_metric, kl_divergence, calculate_detection_metric


@pytest.mark.parametrize('basic_multiopt_metric, maximise', (
    (np.array([[1.0, 0.7], [0.9, 0.8], [0.1, 0.3]]), True),
    (np.array([[1.0, 0.7], [0.9, 0.8], [0.1, 0.3]]), False),
))
def test_pareto_metric(basic_multiopt_metric, maximise):
    pareto_front = ParetoMetrics().pareto_metric_list(costs=basic_multiopt_metric,
                                                      maximise=maximise)
    assert pareto_front is not None
    assert pareto_front[2] is not True


@pytest.mark.parametrize('boundaries, expected', (
    ([[], []], []),
    ([[0, 1], [], [0.5, 2]], [[0, 1], [0.5, 2]]),
    ([[], [0, 1], [0.5, 2]], [[0, 1], [0.5, 2]]),
    ([[0, 1], [0.5, 2], []], [[0, 1], [0.5, 2]]),
))
def test_filter_detecting_boundaries(boundaries, expected):
    assert filter_detecting_boundaries(boundaries) == expected


@pytest.mark.parametrize(
    "my_list, expected_output, raises_exception",
    [
        ([pd.Series([1, 2]), pd.Series([3, 4])], 1, False),
        ([[pd.Series([1, 2, 3]), pd.Series([1, 2])], [pd.Series([1, 2])]], 2, False),
        ([pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-02')], 1, False),
        ([[1, 2], pd.Series([3, 4])], None, True),
        ([[pd.Series([1, 2]), pd.Series([1])], [pd.Timestamp(2017, 1, 1, 12)]], None, True),
        ([[[[]], []], []], None, True),
        ([[], [[]]], 3, False),
        ([], 1, False),
    ]
)
def test_check_errors(my_list, expected_output, raises_exception):
    if raises_exception:
        with pytest.raises(Exception):
            check_errors(my_list)
    else:
        assert check_errors(my_list) == expected_output


@pytest.mark.parametrize(
    "true, predicted_labels, expected_output",
    [
        (np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]), (2, 2, 0, 0)),
        (np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0]), (0, 0, 2, 2)),
        (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), (4, 0, 0, 0)),
        (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), (0, 4, 0, 0)),
        (np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), (0, 0, 4, 0)),
        (np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]), (0, 0, 0, 4)),
        (np.array([1, 0, 1, 0]), np.array([0, 1, 1, 0]), (1, 1, 1, 1)),
        (np.array([1, 1, 1, 0, 0]), np.array([1, 0, 1, 0, 1]), (2, 1, 1, 1)),
    ]
)
def test_confusion_matrix(true, predicted_labels, expected_output):
    assert confusion_matrix(true, predicted_labels) == expected_output


def test_calculate_forecasting_metrics():
    metric_names = ['rmse', 'mape', 'smape', 'mae', 'median_absolute_error']
    y_train = np.random.rand(10)
    target = np.random.rand(10)
    forecast = np.random.rand(10)

    metrics = calculate_forecasting_metric(target,
                                           forecast,
                                           metric_names=metric_names,
                                           rounding_order=3,
                                           y_train=y_train)
    assert metrics is not None
    assert isinstance(metrics, pd.DataFrame)
    assert all(key in metrics for key in metric_names)
    assert all(isinstance(metrics[key].values[0], float) for key in metrics)


def test_kl_divergence():
    # proba 10 samples x 5 classes
    proba = np.random.rand(10, 5)
    proba = proba / proba.sum(axis=1)[:, None]
    solution = pd.DataFrame(proba)

    # proba 10 samples x 5 classes
    proba = np.random.rand(10, 5)
    proba = proba / proba.sum(axis=1)[:, None]
    prediction = pd.DataFrame(proba)

    kl_div = kl_divergence(solution, prediction)
    assert isinstance(kl_div, float)
