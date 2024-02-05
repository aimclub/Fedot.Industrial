import pytest

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.metrics.metrics_implementation import Accuracy, F1, Logloss, MAE, MAPE, MSE, ParetoMetrics, \
    Precision, R2, \
    RMSE, \
    ROCAUC


@pytest.fixture()
def basic_metric_data_clf():
    target_label = np.array([1, 2, 3, 4])
    test_label = np.array([1, 1, 2, 4])
    test_proba = np.array([[0.8, 0.1, 0.05, 0.05],
                           [0.8, 0.1, 0.05, 0.05],
                           [0.1, 0.8, 0.05, 0.05],
                           [0.05, 0.1, 0.05, 0.8]])

    return test_proba, test_label, target_label


@pytest.fixture()
def basic_metric_data_reg():
    test_label = np.array([1.1, 2.2, 3.3, 4.4])
    target_label = np.array([10, 0.1, 2.3, 4.1])

    return test_label, target_label


def test_pareto_metric():
    basic_multiopt_metric = np.array([[1.0, 0.7],
                                      [0.9, 0.8],
                                      [0.1, 0.3]])
    pareto_front = ParetoMetrics().pareto_metric_list(costs=basic_multiopt_metric)
    assert pareto_front is not None
    assert pareto_front[2] is not True


def test_basic_clf_metric(basic_metric_data_clf):
    clf_metric_dict = {'roc_auc': ROCAUC,
                       'f1': F1,
                       'precision': Precision,
                       'accuracy': Accuracy,
                       'logloss': Logloss}

    result_metric = []
    for metric_name in clf_metric_dict.keys():
        chosen_metric = clf_metric_dict[metric_name]
        score = chosen_metric(target=basic_metric_data_clf[2],
                              predicted_labels=basic_metric_data_clf[1],
                              predicted_probs=basic_metric_data_clf[0]).metric()
        score = round(score, 3)
        assert score is not None
        result_metric.append(score)
    assert len(result_metric) > 1


def test_basic_regression_metric(basic_metric_data_reg):
    reg_metric_dict = dict(rmse=RMSE,
                           r2=R2, mae=MAE, mse=MSE, mape=MAPE)
    result_metric = []
    for metric_name in reg_metric_dict.keys():
        chosen_metric = reg_metric_dict[metric_name]
        score = chosen_metric(target=basic_metric_data_reg[1],
                              predicted_labels=basic_metric_data_reg[0]).metric()
        score = round(score, 3)
        assert score is not None
        result_metric.append(score)
    assert len(result_metric) > 1
