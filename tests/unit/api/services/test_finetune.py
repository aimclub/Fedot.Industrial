from __future__ import annotations

from types import SimpleNamespace

import pytest

from fedot_ind.api.services import finetune as finetune_module
from fedot_ind.api.services.finetune import FinetunePayload, FinetuneService


class FakeBuilder:
    def __init__(self, model):
        self.model = model
        self.build_calls = 0

    def build(self):
        self.build_calls += 1
        return self.model


class FakeModel:
    def __init__(self):
        self.fit_calls = []

    def fit(self, data):
        self.fit_calls.append(data)
        return self


def test_prepare_tuning_params_maps_metric_and_tuner(monkeypatch):
    monkeypatch.setitem(finetune_module.FEDOT_TUNING_METRICS, "classification", "f1")
    monkeypatch.setitem(finetune_module.FEDOT_TUNER_STRATEGY, "sequential", "seq_tuner")
    params = {}

    result = FinetuneService().prepare_tuning_params(params, "classification")

    assert result is params
    assert result == {"metric": "f1", "tuner": "seq_tuner"}


def test_prepare_payload_processes_raw_input_and_materializes_builder(monkeypatch):
    monkeypatch.setitem(finetune_module.FEDOT_TUNING_METRICS, "classification", "f1")
    monkeypatch.setitem(finetune_module.FEDOT_TUNER_STRATEGY, "sequential", "seq_tuner")
    model = FakeModel()
    builder = FakeBuilder(model)
    calls = []

    def process_input(data):
        calls.append(("process", data))
        return "processed"

    def init_backend(data):
        calls.append(("backend", data))
        return "backend_ready"

    payload = FinetuneService().prepare_payload(
        train_data="raw",
        tuning_params={},
        model_to_tune=builder,
        task="classification",
        is_fedot_datatype=False,
        process_input=process_input,
        init_backend=init_backend,
    )

    assert payload.train_data == "backend_ready"
    assert payload.model_to_tune is model
    assert payload.tuning_params == {"metric": "f1", "tuner": "seq_tuner"}
    assert calls == [("process", "raw"), ("backend", "processed")]
    assert builder.build_calls == 1


def test_prepare_payload_preserves_fedot_input_without_processing(monkeypatch):
    monkeypatch.setitem(finetune_module.FEDOT_TUNING_METRICS, "classification", "f1")
    monkeypatch.setitem(finetune_module.FEDOT_TUNER_STRATEGY, "sequential", "seq_tuner")
    process_calls = []

    payload = FinetuneService().prepare_payload(
        train_data="fedot_data",
        tuning_params={},
        model_to_tune=FakeModel(),
        task="classification",
        is_fedot_datatype=True,
        process_input=lambda data: process_calls.append(data),
        init_backend=lambda data: f"backend:{data}",
    )

    assert payload.train_data == "backend:fedot_data"
    assert process_calls == []


def test_run_return_only_fitted_fits_model_without_tuner():
    model = FakeModel()
    payload = FinetunePayload(train_data="train", model_to_tune=model, tuning_params={})

    result = FinetuneService().run(
        api=SimpleNamespace(),
        payload=payload,
        return_only_fitted=True,
    )

    assert result is model
    assert model.fit_calls == ["train"]


def test_run_builds_tuner(monkeypatch):
    calls = []

    def fake_build_tuner(api, **kwargs):
        calls.append((api, kwargs))
        return "tuned"

    monkeypatch.setattr(finetune_module, "build_tuner", fake_build_tuner)
    api = SimpleNamespace(name="api")
    payload = FinetunePayload(train_data="train", model_to_tune="model", tuning_params={"metric": "f1"})

    result = FinetuneService().run(api=api, payload=payload, return_only_fitted=False)

    assert result == "tuned"
    assert calls == [
        (
            api,
            {
                "train_data": "train",
                "model_to_tune": "model",
                "tuning_params": {"metric": "f1"},
            },
        )
    ]


def test_prepare_payload_rejects_missing_model_to_tune():
    with pytest.raises(ValueError, match="model_to_tune"):
        FinetuneService().prepare_payload(
            train_data="raw",
            tuning_params={},
            model_to_tune=None,
            task="classification",
            is_fedot_datatype=True,
            process_input=lambda data: data,
            init_backend=lambda data: data,
        )
