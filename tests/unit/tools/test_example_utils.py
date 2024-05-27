from fedot_ind.tools.example_utils import evaluate_metric
import pytest


@pytest.mark.parametrize('name',
                         ['m4_yearly',
                          'm4_weekly',
                          'm4_daily',
                          'm4_monthly',
                          'm4_quarterly'])
def test_get_ts_data(name):
    from fedot_ind.tools.example_utils import get_ts_data
    train_data, test_data, label = get_ts_data(name, 30, 1)
    assert train_data is not None
    assert test_data is not None
    assert label == 1


def test_evaluate_metric():
    target = [0, 1, 0, 1, 0, 1]
    prediction = [0, 1, 0, 1, 0, 1]
    metric = evaluate_metric(target, prediction)
    assert metric == 1.0
