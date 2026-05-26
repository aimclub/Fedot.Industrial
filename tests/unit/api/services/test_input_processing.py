from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fedot_ind.api.services import input_processing as input_processing_module
from fedot_ind.api.services.input_processing import IndustrialInputProcessor


def test_input_processor_copies_input_before_data_check(monkeypatch):
    original = SimpleNamespace(features=np.array([[[1.0]]]), target=np.array([1]))
    observed = {}

    class FakeDataCheck:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.input_data = kwargs["input_data"]

        def check_input_data(self):
            return self.input_data

        def get_target_encoder(self):
            return "encoder"

    monkeypatch.setattr(input_processing_module, "DataCheck", FakeDataCheck)

    bundle = IndustrialInputProcessor().process(
        original,
        task="classification",
        task_params={"x": 1},
        fit_stage=True,
        industrial_task_params={"strategy": "test"},
        default_fedot_context=False,
    )

    assert observed["input_data"] is not original
    assert np.array_equal(observed["input_data"].features, original.features)
    assert observed["task"] == "classification"
    assert observed["task_params"] == {"x": 1}
    assert observed["fit_stage"] is True
    assert observed["industrial_task_params"] == {"strategy": "test"}
    assert bundle.data is observed["input_data"]
    assert bundle.target_encoder == "encoder"


def test_input_processor_squeezes_features_for_default_fedot_context(monkeypatch):
    original = SimpleNamespace(features=np.ones((2, 1, 3)), target=np.array([0, 1]))

    class FakeDataCheck:
        def __init__(self, **kwargs):
            self.input_data = kwargs["input_data"]

        def check_input_data(self):
            return self.input_data

        def get_target_encoder(self):
            return None

    monkeypatch.setattr(input_processing_module, "DataCheck", FakeDataCheck)

    bundle = IndustrialInputProcessor().process(
        original,
        task="classification",
        task_params={},
        default_fedot_context=True,
    )

    assert bundle.data.features.shape == (2, 3)
    assert original.features.shape == (2, 1, 3)


def test_input_processor_preserves_features_for_industrial_context(monkeypatch):
    original = SimpleNamespace(features=np.ones((2, 1, 3)), target=np.array([0, 1]))

    class FakeDataCheck:
        def __init__(self, **kwargs):
            self.input_data = kwargs["input_data"]

        def check_input_data(self):
            return self.input_data

        def get_target_encoder(self):
            return None

    monkeypatch.setattr(input_processing_module, "DataCheck", FakeDataCheck)

    bundle = IndustrialInputProcessor().process(
        original,
        task="classification",
        task_params={},
        default_fedot_context=False,
    )

    assert bundle.data.features.shape == (2, 1, 3)
