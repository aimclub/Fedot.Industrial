import numpy as np
import pytest

from fedot_ind.core.models.automl import kernel_automl as kernel_automl_module


def test_unified_okhs_automl_forecasting_tracks_candidate_scores(monkeypatch):
    observed_qs = []

    class FakeForecaster:
        def __init__(self, q, method, q_policy):
            self.q = float(q)
            self.method = method
            self.q_policy = q_policy

        def fit(self, series):
            observed_qs.append(self.q)
            self.series = np.asarray(series)
            return self

        def predict(self, time_series=None):
            return np.array([self.q])

    monkeypatch.setattr(kernel_automl_module, 'OKHSForecaster', FakeForecaster)

    automl = kernel_automl_module.UnifiedOKHSAutoML(
        task_type='forecasting',
        q_range=(0.3, 0.9),
        q_grid_size=5,
    )
    automl.fit(np.full(12, 0.6))

    assert automl.selected_q_ == pytest.approx(0.6)
    assert automl.best_score_ == pytest.approx(0.0)
    assert len(automl.candidate_scores_) == 5
    assert [round(item['q'], 2) for item in automl.candidate_scores_] == [0.3, 0.45, 0.6, 0.75, 0.9]
    assert automl.fit_summary_['selected_q'] == pytest.approx(0.6)
    assert observed_qs[-1] == pytest.approx(0.6)


def test_unified_okhs_automl_classification_passes_verbose_and_grid(monkeypatch):
    kernel_qs = []
    classifier_calls = {}

    class FakeKernel:
        def __init__(self, q, kernel_type):
            kernel_qs.append((float(q), kernel_type))
            self.q = float(q)

    class FakeClassifier:
        def __init__(self, kernels, penalty, verbose):
            classifier_calls['kernels'] = kernels
            classifier_calls['penalty'] = penalty
            classifier_calls['verbose'] = verbose
            self.kernel_importance_ = np.array([1.0] * len(kernels))

        def fit(self, X, y):
            classifier_calls['fit_shape'] = np.asarray(X).shape
            classifier_calls['fit_targets'] = tuple(y)
            return self

    monkeypatch.setattr(kernel_automl_module, 'OccupationKernel', FakeKernel)
    monkeypatch.setattr(kernel_automl_module, 'RKBSCompositeClassifier', FakeClassifier)

    automl = kernel_automl_module.UnifiedOKHSAutoML(
        task_type='classification',
        q_range=(0.2, 0.8),
        q_grid_size=4,
        verbose=False,
    )
    automl.fit(np.ones((6, 3)), np.array([0, 1, 0, 1, 0, 1]))

    assert [round(q, 2) for q, kernel_type in kernel_qs] == [0.2, 0.4, 0.6, 0.8]
    assert all(kernel_type == 'rbf' for _, kernel_type in kernel_qs)
    assert classifier_calls['penalty'] == 'l1'
    assert classifier_calls['verbose'] is False
    assert automl.fit_summary_['kernel_count'] == 4
    assert automl.fit_summary_['kernel_q_values'] == pytest.approx([0.2, 0.4, 0.6, 0.8])


def test_evaluate_forecaster_returns_negative_infinity_and_records_error():
    class FailingForecaster:
        def fit(self, series):
            raise RuntimeError('synthetic fit failure')

    automl = kernel_automl_module.UnifiedOKHSAutoML(task_type='forecasting')
    score = automl._evaluate_forecaster(FailingForecaster(), np.arange(10, dtype=float))

    assert score == -np.inf
    assert isinstance(automl.last_evaluation_error_, RuntimeError)
    assert 'synthetic fit failure' in str(automl.last_evaluation_error_)


def test_build_q_grid_validates_range():
    automl = kernel_automl_module.UnifiedOKHSAutoML(q_range=(0.9, 0.2))

    with pytest.raises(ValueError, match='q_range'):
        automl._build_q_grid()
