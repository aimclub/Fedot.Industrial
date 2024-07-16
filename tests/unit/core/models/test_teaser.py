import pytest
import numpy as np
from fedot_ind.core.models.early_tc import teaser as TEASER


@pytest.fixture(scope='module')
def teaser():
    teaser = TEASER.TEASER({'interval_length': 10, 'prediction_mode': ''})
    return teaser


@pytest.fixture(scope='module')
def xy():
    return np.random.randn((2, 23)), np.random.randint(0, 2, size=(2, 1))


def test_get_applicable_index(teaser):
    teaser._init_model(23)
    idx, offset = teaser._get_last_applicable_idx(100)
    assert offset == 100 - 22, 'Wrong offset estimation when right edge'
    assert idx == len(teaser.prediction_idx) - 1
    idx, offset = teaser._get_last_applicable_idx(12)
    assert offset == 100 - teaser.prediction_idx[idx], 'Wrong offset estimation in the middle'
    assert idx == len(teaser.prediction_idx) - 1


def test_compute_prediction_points(teaser):
    indices = teaser._compute_prediction_points(23)
    assert 2 in indices
    assert 22 in indices
    assert 23 not in indices

# def test_consecutive_count(teaser):
#     pass

# def test_score(teaser):
