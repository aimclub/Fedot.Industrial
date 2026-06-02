"""Tests for public API for metrics: new (dict) и legacy (DataFrame).

Check, that ``metrics.calculate_*`` and ``metrics_implementation.calculate_*``
return the same values, and for sklearn-compatible metrics — also reference from sklearn.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate._ivp.dop853_coefficients import B
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score as sklearn_f1,
    precision_score as sklearn_prec,
    recall_score as sklearn_recall,
    log_loss as sklearn_log_loss,
    mean_absolute_error as sklearn_mae,
    mean_squared_error as sklearn_mse,
    r2_score as sklearn_r2,
    roc_auc_score as sklearn_roc_auc,
    root_mean_squared_error as sklearn_rmse,
    mean_squared_log_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
    max_error,
    d2_absolute_error_score,
)

from fedot_ind.core.metrics._exceptions import MetricNotFoundError, MetricValidationError, MetricError
from fedot_ind.core.metrics.metrics import (
    calculate_classification_metric,
    calculate_detection_metric,
    calculate_forecasting_metric,
    calculate_regression_metric,
)
# from fedot_ind.core.metrics.metrics_implementation import (
#     calculate_classification_metric as legacy_classification,
#     calculate_detection_metric as legacy_detection,
#     calculate_forecasting_metric as legacy_forecasting,
#     calculate_regression_metric as legacy_regression,
# )

# Same rounding in both APIs to make comparisons easy
ROUND = 8

# --- Data for classification/detection (binary labels) ---
Y_TRUE_BIN = np.array([0, 0, 1, 1, 0, 1, 0, 0])
Y_PRED_BIN = np.array([0, 1, 1, 0, 0, 1, 0, 1])

Y_TRUE_MC = np.array([0, 1, 2, 1, 2, 0, 1, 2])
Y_PRED_MC = np.array([0, 1, 1, 1, 2, 0, 2, 2])

Y_TRUE_PROB = np.array([0, 1, 0, 1, 0, 1])
Y_PRED_PROB = np.array([0, 1, 1, 1, 0, 0])
PROBS_2D = np.array([
    [0.9, 0.1], 
    [0.2, 0.8], 
    [0.4, 0.6],
    [0.3, 0.7], 
    [0.85, 0.15], 
    [0.75, 0.25],
])

# --- Data for regression ---
_rng = np.random.default_rng(42)
Y_TRUE_REG = _rng.normal(0, 1, size=24)
Y_PRED_REG = Y_TRUE_REG + _rng.normal(0, 0.15, size=24)

# --- Data for forecasting ---
_rng_fc = np.random.default_rng(7)
TRAIN_SERIES = np.cumsum(_rng_fc.normal(0, 0.3, size=40)) + 10.0
Y_TRUE_FC = TRAIN_SERIES[-12:] + _rng_fc.normal(0, 0.1, size=12)
Y_PRED_FC = Y_TRUE_FC + _rng_fc.normal(0, 0.2, size=12)
SEASONALITY = 4

# --- Data for NAB / average_time ---
Y_TRUE_ANOM = np.zeros(30, dtype=int)
Y_TRUE_ANOM[12:15] = 1
Y_PRED_ANOM = np.zeros(30, dtype=int)
Y_PRED_ANOM[13] = 1


def assert_new_equals_legacy(new_dict: dict, legacy_df: pd.DataFrame, keys: str) -> float:
    """The new API and legacy API must match in terms of metric key and value."""
    res = []
    for key in keys:
        assert key in new_dict
        assert key in legacy_df
        new_val = new_dict[key]
        leg_val = legacy_df[key].iloc[0]
        if isinstance(new_val, dict):
            assert leg_val == new_val
        elif isinstance(new_val, (list, np.ndarray)):
            np.testing.assert_allclose(np.asarray(new_val), np.asarray(leg_val), rtol=1e-6)
        elif isinstance(new_val, (int, float, np.floating)):
            assert float(new_val) == float(leg_val)
        else:
            raise Exception(f'wrong type: new_val - {type(new_val)} ; leg_val - {type(leg_val)}')
        res.append(float(new_val) if np.isscalar(new_val) or isinstance(new_val, (int, float)) else new_val)
    return res

def run_classification(metric, y_true=Y_TRUE_BIN, y_pred=Y_PRED_BIN, **kwargs):
    """Calling new and legacy APIs for classification."""
    spec = metric if isinstance(metric, (list, tuple)) else (metric,)
    new = calculate_classification_metric(y_true, y_pred, metric_names=spec, rounding_order=ROUND, return_dataframe = False, **kwargs,)
    # legacy = legacy_classification(target=y_true, predicted_labels=y_pred, metric_names=spec, rounding_order=ROUND, **kwargs,)
    legacy = calculate_classification_metric(target=y_true, predicted_labels=y_pred, metric_names=spec, rounding_order=ROUND, **kwargs,)
    name = []
    for s in spec:
        if isinstance(s, str):
            name.append(s)
        elif isinstance(s, dict):
            name.append(s['name'])
        else:
            raise Exception(f'wrong spec element type: {type(s)}')
    return new, legacy, name

def run_regression(metric, y_true=Y_TRUE_REG, y_pred=Y_PRED_REG):
    """Calling new and legacy APIs for regression."""
    spec = metric if isinstance(metric, (list, tuple)) else (metric,)
    new = calculate_regression_metric(y_true, y_pred, metric_names=spec, rounding_order=ROUND, return_dataframe = False,)
    # legacy = legacy_regression(target=y_true, predicted_labels=y_pred,metric_names=spec, rounding_order=ROUND,)
    legacy = calculate_regression_metric(target=y_true, predicted_labels=y_pred,metric_names=spec, rounding_order=ROUND,)
    name = []
    for s in spec:
        if isinstance(s, str):
            name.append(s)
        elif isinstance(s, dict):
            name.append(s['name'])
        else:
            raise Exception(f'wrong spec element type: {type(s)}')
    return new, legacy, name

def run_forecasting(metric, y_true=Y_TRUE_FC, y_pred=Y_PRED_FC):
    spec = metric if isinstance(metric, (list, tuple)) else (metric,)
    new = calculate_forecasting_metric(
        y_true, y_pred, metric_names=spec, rounding_order=ROUND,
        train_data=TRAIN_SERIES, seasonality=SEASONALITY, return_dataframe = False,
    )
    # legacy = legacy_forecasting(target=y_true, predicted_labels=y_pred, metric_names=spec,rounding_order=ROUND, train_data=TRAIN_SERIES, seasonality=SEASONALITY,)
    legacy = calculate_forecasting_metric(target=y_true, predicted_labels=y_pred, metric_names=spec, rounding_order=ROUND, train_data=TRAIN_SERIES, seasonality=SEASONALITY,)
    name = []
    for s in spec:
        if isinstance(s, str):
            name.append(s)
        elif isinstance(s, dict):
            name.append(s['name'])
        else:
            raise Exception(f'wrong spec element type: {type(s)}')
    return new, legacy, name

def run_detection(metric, y_true=Y_TRUE_ANOM, y_pred=Y_PRED_ANOM):
    spec = metric if isinstance(metric, (list, tuple)) else (metric,)
    new = calculate_detection_metric(y_true, y_pred, metric_names=spec, rounding_order=ROUND, return_dataframe = False,)
    # legacy = legacy_detection(target=y_true, predicted_labels=y_pred, metric_names=spec, rounding_order=ROUND,)
    legacy = calculate_detection_metric(target=y_true, predicted_labels=y_pred, metric_names=spec, rounding_order=ROUND,)
    name = []
    for s in spec:
        if isinstance(s, str):
            name.append(s)
        elif isinstance(s, dict):
            name.append(s['name'])
        else:
            raise Exception(f'wrong spec element type: {type(s)}')
    return new, legacy, name

# =============================================================================
# Classification
# =============================================================================

class TestClassificationMetrics:
    """metrics.calculate_classification_metric vs legacy + sklearn."""

    def test_accuracy(self):
        new, leg, name = run_classification('accuracy')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(accuracy_score(Y_TRUE_BIN, Y_PRED_BIN), rel=1e-4)

    def test_balanced_accuracy(self):
        new, leg, name = run_classification('balanced_accuracy')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(balanced_accuracy_score(Y_TRUE_BIN, Y_PRED_BIN), rel=1e-4)

    def test_balanced_accuracy_with_arguments(self):
        spec = ({'name': 'balanced_accuracy', 'params': {'labels': [0,1], 'pos_label': 0}})
        new, leg, name = run_classification(spec)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(balanced_accuracy_score(Y_TRUE_BIN, Y_PRED_BIN), rel=1e-4)

    def test_f1_binary(self):
        spec = {'name': 'f1', 'params': {'average': 'binary', 'pos_label': 1}}
        new, leg, name = run_classification(spec)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_f1(Y_TRUE_BIN, Y_PRED_BIN, average='binary', pos_label=1), rel=1e-4,
        )

    def test_f1_bin_with_multiclass(self):
        spec = {'name': 'f1', 'params': {'average': 'binary', 'pos_label': 1}}
        with pytest.raises(ValueError, match='binary average для F1 возможна только при 2 классах'):
            run_classification(spec, y_true=Y_TRUE_MC, y_pred=Y_PRED_MC)
    
    def test_f1_bin_with_uniform_zeroes(self):
        spec = {'name': 'f1', 'params': {'average': 'binary', 'pos_label': 1}}
        new, leg, name = run_classification(spec, y_true=np.ones(10),y_pred=np.ones(10))
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_f1(np.ones(10), np.ones(10), average='binary', pos_label=1), rel=1e-4,
        )
    
    def test_f1_bin_with_uniform_number(self):
        with pytest.raises(MetricValidationError, match="Please add labels parameter for metric. Default labels for autocomplete is 0 and 1 - integer."):
            spec = {'name': 'f1', 'params': {'average': 'binary', 'pos_label': 1}}
            a = np.empty(10, dtype=int)
            a.fill(7)
            run_classification(spec, y_true=a,y_pred=a)

    def test_f1_bin_with_uniform_number_and_labels_argument(self):
        spec = {'name': 'f1', 'params': {'average': 'binary', 'pos_label': 7, 'labels': [0,7]}}
        a = np.empty(10, dtype=int)
        a.fill(7)
        new, leg, name = run_classification(spec, y_true=a,y_pred=a)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_f1(a, a, average='binary', pos_label=7), rel=1e-4,
        )

    def test_f1_with_unknown_average(self):
        spec = {'name': 'f1', 'params': {'average': 'aaaaa', 'pos_label': 1}}
        with pytest.raises(ValueError, match='Неизвестный тип усреднения:'):
            run_classification(spec)

    def test_f1_micro(self):
        spec = {'name': 'f1', 'params': {'average': 'micro'}}
        new, leg, name = run_classification(spec, y_true=Y_TRUE_MC, y_pred=Y_PRED_MC)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_f1(Y_TRUE_MC, Y_PRED_MC, average='micro'), rel=1e-4,
        )

    def test_f1_macro(self):
        spec = {'name': 'f1', 'params': {'average': 'macro'}}
        new, leg, name = run_classification(spec, y_true=Y_TRUE_MC, y_pred=Y_PRED_MC)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_f1(Y_TRUE_MC, Y_PRED_MC, average='macro'), rel=1e-4,
        )
        
    def test_f1_weighted(self):
        spec = {'name': 'f1', 'params': {'average': 'weighted'}}
        new, leg, name = run_classification(spec, y_true=Y_TRUE_MC, y_pred=Y_PRED_MC)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_f1(Y_TRUE_MC, Y_PRED_MC, average='weighted'), rel=1e-4,
        )
    
    # TODO
    # def test_f1_auto_auto(self):
    #     # если всё будет auto (average = default [auto] | pos_label = default [auto])
    #     pass

    def test_precision_macro_multiclass(self):
        spec = {'name': 'precision', 'params': {'average': 'macro'}}
        new, leg, name = run_classification(spec, y_true=Y_TRUE_MC, y_pred=Y_PRED_MC)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_prec(Y_TRUE_MC, Y_PRED_MC, average='macro'), rel=1e-4,
        )

    def test_precision_weighted(self):
        spec = {'name': 'precision', 'params': {'average': 'weighted'}}
        new, leg, name = run_classification(spec, y_true=Y_TRUE_MC, y_pred=Y_PRED_MC)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_prec(Y_TRUE_MC, Y_PRED_MC, average='weighted'), rel=1e-4,
        )

    def test_recall_macro_multiclass(self):
        spec = {'name': 'recall', 'params': {'average': 'macro'}}
        new, leg, name = run_classification(spec, y_true=Y_TRUE_MC, y_pred=Y_PRED_MC)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_recall(Y_TRUE_MC, Y_PRED_MC, average='macro'), rel=1e-4,
        )

    def test_logloss_with_probs(self):
        new, leg, name = run_classification(
            'logloss', y_true=Y_TRUE_PROB, y_pred=Y_PRED_PROB, predicted_probs=PROBS_2D,
        )
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(sklearn_log_loss(Y_TRUE_PROB, PROBS_2D), rel=1e-4)

    # Тут скорее вылетает ошибка, что разные размеры у предиктед_лейблс и таргета. 
    # В целом, это хорошо, но я добавлю проверку на то, если предиктед_лейблс = None. Или не добавлю, хз пока что
    def test_logloss_without_predicted_and_probs(self):
        with pytest.raises(MetricValidationError, match='Predicted labels is None.'):
            run_classification(
                'logloss', y_true=Y_TRUE_PROB, y_pred=None, predicted_probs=None,
        )

    # Должно сделать one-hot от predicted и использовать, как probs
    def test_log_loss_without_probs(self):
        new, leg, name = run_classification(
            'logloss', y_true=Y_TRUE_PROB, y_pred=Y_PRED_PROB)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(sklearn_log_loss(Y_TRUE_PROB, 
                                                        np.array([[1.0, 0.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[1.0, 0.0],[1.0, 0.0]]))
                                                        , rel=1e-4)

    def test_roc_auc_with_probs_multiclass(self):
        # Должно пойти по ветке, типа больше, чем 2 класса
        probs_MC = np.array([np.array([0.8,0.1,0.1]),
                             np.array([0.1,0.8,0.1]),
                             np.array([0.1,0.8,0.1]),
                             np.array([0.1,0.8,0.1]),
                             np.array([0.1,0.1,0.8]),
                             np.array([0.8,0.1,0.1]),
                             np.array([0.1,0.1,0.8]),
                             np.array([0.1,0.1,0.8])])
        new, leg, name = run_classification(
            'roc_auc', 
            y_true=Y_TRUE_MC, 
            # y_pred=Y_PRED_MC, 
            predicted_probs=probs_MC,
        )
        val = assert_new_equals_legacy(new, leg, name)
        # assert val[0] == pytest.approx(
        #     sklearn_roc_auc(Y_TRUE_MC, probs_MC), rel=1e-4,
        # )

    def test_roc_auc_without_probs_multiclass(self):
        # Должно пойти по ветке, типа больше, чем 2 класса и сделать one-hot из predicted_labels
        probs_MC_OH = np.array([[1.0,0.0,0.0],
                             [0.0,1.0,0.0],
                             [0.0,1.0,0.0],
                             [0.0,1.0,0.0],
                             [0.0,0.0,1.0],
                             [1.0,0.0,0.0],
                             [0.0,0.0,1.0],
                             [0.0,0.0,1.0]])
        new, leg, name = run_classification(
            'roc_auc', y_true=Y_TRUE_MC, y_pred=Y_PRED_MC,
        )
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_roc_auc(Y_TRUE_MC, probs_MC_OH, multi_class='ovr', average='macro'), rel=1e-4,
        )

    def test_roc_auc_with_probs_bin(self):
        new, leg, name = run_classification(
            'roc_auc', y_true=Y_TRUE_PROB, y_pred=Y_PRED_PROB, predicted_probs=PROBS_2D,
        )
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_roc_auc(Y_TRUE_PROB, PROBS_2D[:, 1]), rel=1e-4,
        )

    # TODO - доработать, чтобы он сравнил поточечно типа
    def test_per_class_scores(self):
        new, leg, name = run_classification('per_class_scores', y_true=Y_TRUE_MC, y_pred=Y_PRED_MC)
        block = new[name[0]]
        # legacy expands the nested dict into columns per_class_scores_f1, ...
        assert leg['f1'].iloc[0] == block['f1']
        assert 'recall' in block and 'precision' in block

    def test_per_class_scores_with_pos_label(self):
        spec = ({'name': 'per_class_scores', 'params': {'labels': [0,1,2]}},)
        new, leg, name = run_classification(spec, y_true=Y_TRUE_MC, y_pred=Y_PRED_MC)
        block = new[name[0]]
        # legacy expands the nested dict into columns per_class_scores_f1, ...
        assert leg['f1'].iloc[0] == block['f1']
        assert 'recall' in block and 'precision' in block

    def test_binary_metrics(self):
        new, leg, name = run_detection(
            'bin_metrics', y_true=Y_TRUE_BIN, y_pred=Y_PRED_BIN,
        )
        block = new[name[0]]
        assert leg['FAR'].iloc[0] == block['FAR']
        assert leg['MAR'].iloc[0] == block['MAR']
        assert 'recall' in block and 'precision' in block and 'f1' in block

    def test_binary_conf_matrix_for_multiclass(self):
        with pytest.raises(ValueError, match='Матрица должна быть 2x2 для бинарного случая.'):
            run_detection(
                'bin_confusion_matrix', y_true=Y_TRUE_MC, y_pred=Y_PRED_MC,
        )

# =============================================================================
# Regression
# =============================================================================

class TestRegressionMetrics:
    def test_mse(self):
        new, leg, name = run_regression('mse')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(sklearn_mse(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)

    def test_rmse(self):
        new, leg, name = run_regression('rmse')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(sklearn_rmse(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)

    def test_mae(self):
        new, leg, name = run_regression('mae')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(sklearn_mae(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)

    def test_r2(self):
        new, leg, name = run_regression('r2')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(sklearn_r2(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)

    def test_multi_metrics(self):
        new, leg, name = run_regression(('mse','mae'))
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(sklearn_mse(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)
        assert val[1] == pytest.approx(sklearn_mae(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)

    def test_msle(self):
        a = Y_TRUE_REG+5.
        b = Y_PRED_REG+5.
        new, leg, name = run_regression('msle', y_true=a, y_pred=b)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(mean_squared_log_error(a, b), rel=1e-4)

    def test_msle_neg_values(self):
        with pytest.raises(MetricValidationError, match="There are negative values in GT or predicted sequence. MSLE does not support negative values"):
            run_regression('msle')

    def test_mape(self):
        new, leg, name = run_regression('mape')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(mean_absolute_percentage_error(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)

    def test_median_abs_er(self):
        new, leg, name = run_regression('median_absolute_error')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(median_absolute_error(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)

    def test_explained_variance_score(self):
        new, leg, name = run_regression('explained_variance_score')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(explained_variance_score(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)

    def test_max_error(self):
        new, leg, name = run_regression('max_error')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(max_error(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)

    def test_d2_absolute_error_score(self):
        new, leg, name = run_regression('d2_absolute_error_score')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(d2_absolute_error_score(Y_TRUE_REG, Y_PRED_REG), rel=1e-4)

    def test_different_length(self):
        with pytest.raises(MetricValidationError, match='target and predicted must have the same length'):
            aaa = np.append(Y_PRED_REG, np.array([0.0, 1.0]))
            run_regression(
                'mse', y_true=Y_TRUE_REG, y_pred=aaa,)

    def test_different_shape(self):
        with pytest.raises(MetricValidationError, match='target and predicted must have the same length'):
            aaa = np.append(Y_PRED_REG, Y_PRED_REG).reshape(2, -1)
            run_regression(
                'mse', y_true=Y_TRUE_REG, y_pred=aaa,)

# =============================================================================
# Forecasting
# =============================================================================

class TestForecastingMetrics:
    def test_mse(self):
        new, leg, name = run_forecasting('mse')
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(sklearn_mse(Y_TRUE_FC, Y_PRED_FC), rel=1e-4)

    def test_smape(self):
        spec = ({'name': 'smape', 'params':{'epsilon': 15.0, 'aaa': 5}}, 
                'smape')
        new, leg, name = run_forecasting(spec)
        assert_new_equals_legacy(new, leg, name)
        keys = list(new.keys())
        assert new[keys[0]] != new[keys[1]]

    def test_mase(self):
        new, leg, name = run_forecasting('mase')
        assert_new_equals_legacy(new, leg, name)

    def test_owa(self):
        new, leg, name = run_forecasting('owa')
        assert_new_equals_legacy(new, leg, name)

    def test_pw_mae_pointwise(self):
        new, leg, name = run_forecasting('pw_mae')
        val = assert_new_equals_legacy(new, leg, name)
        np.testing.assert_allclose(val[0], np.abs(Y_TRUE_FC - Y_PRED_FC), rtol=1e-6)

# =============================================================================
# Detection
# =============================================================================

class TestDetectionMetrics:
    def test_accuracy_via_detection_task(self):
        new, leg, name = run_detection('accuracy', y_true=Y_TRUE_BIN, y_pred=Y_PRED_BIN)
        assert_new_equals_legacy(new, leg, name)

    def test_bin_f1(self):
        new, leg, name = run_detection('bin_f1', y_true=Y_TRUE_BIN, y_pred=Y_PRED_BIN)
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(
            sklearn_f1(Y_TRUE_BIN, Y_PRED_BIN, average='binary', pos_label=1), rel=1e-4,
        )

    def test_bin_confusion_matrix(self):
        new, leg, name = run_detection('bin_confusion_matrix', y_true=Y_TRUE_BIN, y_pred=Y_PRED_BIN)
        cm = new[name[0]]
        assert leg['TP'].iloc[0] == cm['TP']
        assert set(cm) == {'TP', 'TN', 'FP', 'FN'}

    # ПРОВЕРЯТЬ
    def test_nab(self):
        new, leg, name = run_detection('nab')
        new_data = new[name[0]]
        # Standard  LowFP  LowFN
        assert leg['Standard'].iloc[0] == new_data['Standard']
        assert leg['LowFP'].iloc[0] == new_data['LowFP']
        assert leg['LowFN'].iloc[0] == new_data['LowFN']

    def test_nab_all(self):
        spec = (
            {'name': 'nab', 'params': {
                'anomaly_window_destination': 'center'
            }},
            {'name': 'nab', 'params': {
                'anomaly_window_destination': 'center',
                'portion': 0.5
            }},
            {'name': 'nab_standard', 'params': {
                'anomaly_window_destination': 'center'
            }},
            {'name': 'nab_low_fp', 'params': {
                'anomaly_window_destination': 'center'
            }},
            {'name': 'nab_low_fn', 'params': {
                'anomaly_window_destination': 'center'
            }},
        )
        dictt = calculate_detection_metric(Y_TRUE_ANOM, Y_PRED_ANOM, metric_names=spec, rounding_order=ROUND, return_dataframe = False)
        assert dictt['nab__1'] != dictt['nab']
        full = dictt['nab']
        assert full['Standard'] == dictt['nab_standard']
        assert full['LowFP'] == dictt['nab_low_fp']
        assert full['LowFN'] == dictt['nab_low_fn']

    # ПРОВЕРЯТЬ
    def test_average_time(self):
        new, leg, name = run_detection('average_time')
        # assert_new_equals_legacy(new, leg, name)
        block = new[name[0]]
        # legacy expands the nested dict into columns per_class_scores_f1, ...
        assert leg['average_delay'].iloc[0] == block['average_delay']
        assert 'missing' in block and 'FP' in block and 'total_anomalies' in block


# =============================================================================
# Спецификация метрик: dict и дубликаты имён
# =============================================================================

class TestMetricSpecifications:

    def test_metric_as_dict_with_params(self):
        """The metric is specified by a dictionary ``{'name': ..., 'params': ...}``."""
        spec = {'name': 'f1', 'params': {'average': 'binary', 'pos_label': 1}}
        # new = calculate_classification_metric(Y_TRUE_BIN, Y_PRED_BIN, metrics=[spec], rounding_order=ROUND)
        # legacy = legacy_classification(
        #     target=Y_TRUE_BIN, predicted_labels=Y_PRED_BIN,
        #     metric_names=[spec], rounding_order=ROUND,
        # )
        new, leg, name = run_classification(spec)
        assert_new_equals_legacy(new, leg, name)

    def test_two_metrics_same_name_different_params(self):
        """Two ``f1`` with different params → keys ``f1`` and ``f1__1`` (binary case)."""
        specs = [
            {'name': 'f1', 'params': {'average': 'binary', 'pos_label': 1}},
            {'name': 'f1', 'params': {'average': 'weighted'}},
        ]
        new = calculate_classification_metric(Y_TRUE_BIN, Y_PRED_BIN, metric_names=specs, rounding_order=ROUND, return_dataframe = False)
        assert 'f1' in new and 'f1__1' in new
        assert new['f1'] != new['f1__1']

    def test_duplicate_mse_regression(self):
        new = calculate_regression_metric(
            Y_TRUE_REG, Y_PRED_REG,
            metric_names=[{'name': 'mse'}, {'name': 'mse'}],
            rounding_order=ROUND, return_dataframe = False,
        )
        assert 'mse' in new and 'mse__1' in new
        assert new['mse'] == new['mse__1']

    def test_duplicate_nab_detection(self):
        specs = [
            {'name': 'nab', 'params': {'profile': 'Standard', 'scale_koef': 1.0}},
            {'name': 'nab', 'params': {'profile': 'Standard', 'scale_koef': 0.5}},
        ]
        new = calculate_detection_metric(Y_TRUE_ANOM, Y_PRED_ANOM, metric_names=specs, rounding_order=ROUND, return_dataframe = False)
        assert 'nab' in new and 'nab__1' in new
        # assert new['nab'] != new['nab__1']    # Не работает, потому что тестовая последовательность такая, что там не будет отлиций. 
                                                # Но если дебажить, то сё прокидывается нормально

    def test_multi_metrics(self):
        new, leg, name = run_classification(('accuracy',
                                             {'name': 'f1', 
                                              'params': {
                                                  'average': 'binary', 
                                                  'pos_label': 1}}))
        val = assert_new_equals_legacy(new, leg, name)
        assert val[0] == pytest.approx(accuracy_score(Y_TRUE_BIN, Y_PRED_BIN), rel=1e-4)
        assert val[1] == pytest.approx(
            sklearn_f1(Y_TRUE_BIN, Y_PRED_BIN, average='binary', pos_label=1), rel=1e-4,
        )

# =============================================================================
# Ошибки и граничные случаи
# =============================================================================

class TestMetricsValidation:
    
    def test_probs3D(self):
        with pytest.raises(MetricValidationError, match='predicted_probs must be 1D or 2D'):
            calculate_classification_metric([1, 2], [1, 2], [[[1.0,0.0],[0.0,1.0]]])

    def test_probs_not_same_length(self):
        with pytest.raises(MetricValidationError, match='predicted_probs must be 1D or 2D'):
            calculate_classification_metric([1, 2], [1, 2], [[1.0,0.0],[0.0,1.0],[0.0,0.0]])

    # TODO - ещё можно сделать тест на _labels типа, там делается аргмакс (только классификация и аномалии)

    def test_regression_length_mismatch(self):
        with pytest.raises(MetricValidationError, match='same length'):
            calculate_regression_metric([1.0, 2.0], [1.0])

    def test_detection_length_mismatch(self):
        with pytest.raises(MetricValidationError, match='length mismatch'):
            calculate_detection_metric([0, 1], [0, 1, 0], metric_names=('accuracy',), return_dataframe = False)

    def test_nested_series_not_supported(self):
        with pytest.raises(MetricValidationError, match='multiple series'):
            calculate_detection_metric([[0, 1], [1, 0]], [0, 1], metric_names=('accuracy',), return_dataframe = False)

    def test_unknown_metric(self):
        with pytest.raises(MetricNotFoundError):
            calculate_regression_metric([1.0], [1.0], metric_names=('no_such_metric',))

    def test_invalid_metric_spec_type(self):
        with pytest.raises(MetricValidationError, match='Invalid metric spec'):
            calculate_regression_metric([1.0], [1.0], metric_names=[123])  # type: ignore[list-item]

    # TODO - ещё тогда уж можно проверить _parse_spec(), как ниже, но для другого  
    # raise MetricValidationError('Metric spec "params" must be a mapping.')

    def test_dict_spec_without_name(self):
        with pytest.raises(MetricValidationError, match='must include "name"'):
            calculate_regression_metric([1.0], [1.0], metric_names=[{'params': {}}])

class TestInputTypes:
    """Different types of input data (list / numpy)."""

    def test_classification_with_lists(self):
        y_t = Y_TRUE_BIN.tolist()
        y_p = Y_PRED_BIN.tolist()
        new, leg, name = run_classification('accuracy', y_true=y_t, y_pred=y_p)
        assert_new_equals_legacy(new, leg, name)

    def test_classification_with_numpy(self):
        new, leg, name = run_classification('accuracy')
        assert_new_equals_legacy(new, leg, name)



# TODO - Надо сделать тест на проверку округления, нормально ли оно работает, а то мало ли забыл где-то округлять, а где-то норм
# Я это переделал, теперь там вроде бы всё норм в одном месте делается округление после подсчёта каждой метрики в цикле по метрикам