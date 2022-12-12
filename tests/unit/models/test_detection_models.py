import numpy as np
import pytest
from core.models.detection.subspaces.SSTdetector import SingularSpectrumTransformation


@pytest.fixture()
def basic_periodic_data():
    x0 = 1 * np.ones(1000) + np.random.rand(1000) * 1
    x1 = 3 * np.ones(1000) + np.random.rand(1000) * 2
    x2 = 5 * np.ones(1000) + np.random.rand(1000) * 1.5
    x = np.hstack([x0, x1, x2])
    x += np.random.rand(x.size)
    return x


def test_SST_detector(basic_periodic_data):

    scorer = SingularSpectrumTransformation(time_series=basic_periodic_data,
                                            ts_window_length=100,
                                            lag=10,
                                            trajectory_window_length=30)
    score = scorer.score_offline(dynamic_mode=False)
