import numpy as np
import pytest

from fedot_ind.core.metrics.anomaly_detection.function import filter_detecting_boundaries
from fedot_ind.core.metrics.metrics_implementation import ParetoMetrics


def test_pareto_metric():
    basic_multiopt_metric = np.array([[1.0, 0.7],
                                      [0.9, 0.8],
                                      [0.1, 0.3]])
    pareto_front = ParetoMetrics().pareto_metric_list(costs=basic_multiopt_metric)
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
