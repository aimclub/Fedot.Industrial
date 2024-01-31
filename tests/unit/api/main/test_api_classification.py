import numpy as np
import pytest
from fedot.core.pipelines.pipeline import Pipeline

from fedot_ind.api.main import FedotIndustrial
from tests.unit.api.fixtures import univariate_time_series_np, uni_classification_labels_np, univariate_time_series_df, \
    uni_classification_labels_df, multivariate_time_series_np, multi_classification_labels_df, \
    multi_classification_labels_np, multivariate_time_series_df, regression_target_np, regression_target_df, \
    regression_multi_target_np, regression_multi_target_df


@pytest.fixture
def fedot_industrial_classification():
    return FedotIndustrial(problem='classification',  timeout=0.5)

@pytest.fixture
def fedot_industrial_regression():
    return FedotIndustrial(problem='regression', timeout=0.5)

@pytest.mark.parametrize("series,target", [
    ("univariate_time_series_np", "uni_classification_labels_np"),
    ("univariate_time_series_df", "uni_classification_labels_df"),
    ("multivariate_time_series_np", "uni_classification_labels_np"),
    ("multivariate_time_series_df", "uni_classification_labels_df"),
    ("univariate_time_series_np", "multi_classification_labels_np"),
    ("univariate_time_series_df", "multi_classification_labels_df"),
    ("multivariate_time_series_np", "multi_classification_labels_np"),
    ("multivariate_time_series_df", "multi_classification_labels_df")
])
def test_fit_predict_classification(fedot_industrial_classification, series, target, request):
    train_data = (request.getfixturevalue(series), request.getfixturevalue(target))
    fedot_industrial_classification.fit(train_data)
    predict = fedot_industrial_classification.predict(train_data)


    assert predict.shape[0] == train_data[1].shape[0]

    num_unique = np.unique(train_data[1])
    predict_proba = fedot_industrial_classification.predict_proba(train_data)
    assert predict_proba.shape[0] == train_data[1].shape[0]
    if len(num_unique) > 2:
        assert predict_proba.shape[1] == len(num_unique)
    else:

        assert len(predict_proba.shape) == 1


@pytest.mark.parametrize("series,target", [
    ("univariate_time_series_np", "regression_target_np"),
    ("univariate_time_series_df", "regression_target_df"),
    ("multivariate_time_series_np", "regression_target_np"),
    ("multivariate_time_series_df", "regression_target_df"),
    ("univariate_time_series_np", "regression_multi_target_np"),
    ("univariate_time_series_df", "regression_multi_target_df"),
    ("multivariate_time_series_np", "regression_multi_target_np"),
    ("multivariate_time_series_df", "regression_multi_target_df")
])
def test_fit_predict_regression(fedot_industrial_regression, series, target, request):
    train_data = (request.getfixturevalue(series), request.getfixturevalue(target))
    fedot_industrial_regression.fit(train_data)
    predict = fedot_industrial_regression.predict(train_data)

    assert predict.shape[0] == train_data[1].shape[0]

    if len(train_data[1].shape) > 1:
        assert predict.shape[1] == train_data[1].shape[1]
    else:
        assert len(predict.shape) == 1
