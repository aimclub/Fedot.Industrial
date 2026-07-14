from __future__ import annotations

import numpy as np
import pandas as pd

from fedot_ind.api.services import metrics as metrics_module
from fedot_ind.api.services.metrics import MetricEvaluationService


class FakeConditionCheck:
    def __init__(self, have_encoder=False):
        self.have_encoder = have_encoder

    def solver_have_target_encoder(self, encoder):
        return self.have_encoder and encoder is not None


def test_metric_service_evaluates_single_prediction_without_encoder(monkeypatch):
    calls = []

    def fake_metric(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"metric": [1.0]})

    monkeypatch.setitem(metrics_module.FEDOT_GET_METRICS, "classification", fake_metric)
    target = np.array([[0], [1]])
    labels = np.array([0, 1])

    result = MetricEvaluationService().evaluate(
        target=target,
        predicted_labels=labels,
        predicted_probs=None,
        problem="classification",
        metric_names=("accuracy",),
        rounding_order=3,
        train_data=None,
        seasonality=1,
        condition_check=FakeConditionCheck(),
        target_encoder=None,
    )

    assert result.iloc[0, 0] == 1.0
    np.testing.assert_array_equal(calls[0]["target"], np.array([0, 1]))
    np.testing.assert_array_equal(calls[0]["labels"], target)
    assert calls[0]["metric_names"] == ("accuracy",)


def test_metric_service_transforms_target_and_labels_with_encoder(monkeypatch):
    calls = []

    class FakeEncoder:
        def transform(self, values):
            return np.asarray(values) + 10

    def fake_metric(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setitem(metrics_module.FEDOT_GET_METRICS, "classification", fake_metric)

    result = MetricEvaluationService().evaluate(
        target=np.array([[0], [1]]),
        predicted_labels=np.array([1, 0]),
        predicted_probs=None,
        problem="classification",
        metric_names=("f1",),
        rounding_order=3,
        train_data=None,
        seasonality=1,
        condition_check=FakeConditionCheck(have_encoder=True),
        target_encoder=FakeEncoder(),
    )

    assert result == {"ok": True}
    np.testing.assert_array_equal(calls[0]["target"], np.array([10, 11]))
    np.testing.assert_array_equal(calls[0]["labels"], np.array([[11], [10]]))


def test_metric_service_evaluates_prediction_dict_per_model(monkeypatch):
    calls = []

    def fake_metric(**kwargs):
        calls.append(kwargs)
        return kwargs["labels"].sum()

    monkeypatch.setitem(metrics_module.FEDOT_GET_METRICS, "classification", fake_metric)

    result = MetricEvaluationService().evaluate(
        target=np.array([0, 1]),
        predicted_labels={"a": np.array([0, 1]), "b": np.array([1, 1])},
        predicted_probs=np.array([[0.8, 0.2], [0.1, 0.9]]),
        problem="classification",
        metric_names=("accuracy",),
        rounding_order=3,
        train_data=None,
        seasonality=1,
        condition_check=FakeConditionCheck(),
        target_encoder=None,
    )

    assert result == {"a": 1, "b": 2}
    assert [call["labels"].tolist() for call in calls] == [[0, 1], [1, 1]]
