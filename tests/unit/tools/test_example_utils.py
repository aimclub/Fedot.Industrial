from fedot_ind.tools.example_utils import evaluate_metric
from fedot_ind.tools.example_utils import get_ts_data
import pytest


# @pytest.mark.parametrize('name',
#                          ['m4_daily',
#                           'm4_weekly',
#                           'm4_monthly',
#                           'm4_quarterly',
#                           'm4_yearly'])
@pytest.mark.parametrize('group',
                         ['Daily',
                          'Weekly',
                          'Monthly',
                          'Quarterly',
                          'Yearly'])
def test_get_ts_data(group):
    # ds_ids = {'d': 3530,
    #           'w': 124,
    #           'm': 14148,
    #           'q': 12090,
    #           'y': 3917}
    # idx = str.find(name, '_')
    # ds_name = str.capitalize(name[idx + 1]) + str(ds_ids[name[idx + 1]])
    train_data, test_data, label = get_ts_data(dataset=f'M4_{group}',
                                               horizon=30,
                                               m4_id=None)
    assert train_data is not None
    assert test_data is not None


def test_evaluate_metric():
    target = [0, 1, 0, 1, 0, 1]
    prediction = [0, 1, 0, 1, 0, 1]
    metric = evaluate_metric(target, prediction)
    assert metric == 1.0
