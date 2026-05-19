from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

kernel_module = pytest.importorskip("fedot_ind.core.ensemble.kernel_ensemble")
industrial_strategy_module = pytest.importorskip("fedot_ind.api.utils.industrial_strategy")
OperationParameters = pytest.importorskip("fedot.core.operations.operation_parameters").OperationParameters


def test_kernel_strategy_accepts_legacy_trailing_space_key():
    ensembler = kernel_module.KernelEnsembler(OperationParameters(**{"kernel_strategy ": "one_step_pwmk"}))

    assert ensembler.kernel_strategy == "one_step_pwmk"


def test_kernel_strategy_canonical_key_wins_over_legacy_typo():
    ensembler = kernel_module.KernelEnsembler(
        OperationParameters(
            **{
                "kernel_strategy": "two_step_rmkl",
                "kernel_strategy ": "one_step_cka",
            }
        )
    )

    assert ensembler.kernel_strategy == "two_step_rmkl"


def test_create_kernel_data_maps_multiclass_targets_with_isin_membership():
    ensembler = kernel_module.KernelEnsembler.__new__(kernel_module.KernelEnsembler)
    ensembler.learning_strategy = "selected_classes"
    ensembler.mapper_dict = {"gen": {"class_a": 0, "class_b": 1}}
    input_data = SimpleNamespace(target=np.array([["class_a"], ["class_c"], ["class_b"], ["class_d"]], dtype=object))

    remapped = ensembler._create_kernel_data(
        input_data,
        classes_described_by_generator={"gen": ["class_a", "class_b"]},
        gen="gen",
    )

    assert remapped.target.shape == input_data.target.shape
    assert remapped.target.reshape(-1).tolist() == [0, 2, 1, 2]
    assert input_data.target.reshape(-1).tolist() == ["class_a", "class_c", "class_b", "class_d"]


def test_generate_grammian_resets_accumulated_train_features(monkeypatch):
    class FakeGenerator:
        def __init__(self, offset: float):
            self.predict = np.array(
                [
                    [[offset, 0.0]],
                    [[offset + 1.0, 1.0]],
                    [[offset + 2.0, 2.0]],
                ]
            )

        def fit(self, input_data):
            del input_data
            return self

    class FakeBuilder:
        def __init__(self, offset: float):
            self.offset = offset

        def build(self):
            return FakeGenerator(self.offset)

    monkeypatch.setattr(
        kernel_module,
        "KERNEL_BASELINE_FEATURE_GENERATORS",
        {"first": FakeBuilder(1.0), "second": FakeBuilder(10.0)},
    )
    ensembler = kernel_module.KernelEnsembler.__new__(kernel_module.KernelEnsembler)
    ensembler.feature_extractor = ["first", "second"]
    ensembler.distance_metric = "euclidean"
    ensembler.feature_matrix_train = ["stale"]

    first_call = ensembler.generate_grammian(SimpleNamespace())
    second_call = ensembler.generate_grammian(SimpleNamespace())

    assert len(first_call) == 2
    assert len(second_call) == 2
    assert len(ensembler.feature_matrix_train) == 2
    assert all(kernel.shape == (3, 3) for kernel in second_call)


def test_kernel_prediction_remapping_is_deterministic_for_missed_classes():
    strategy = industrial_strategy_module.IndustrialStrategy.__new__(industrial_strategy_module.IndustrialStrategy)
    strategy.kernel_ensembler = SimpleNamespace(
        all_classes=np.array(["class_a", "class_b", "class_c"], dtype=object),
        classes_described_by_generator={
            "gen_ab": ["class_a", "class_b"],
            "gen_bc": ["class_b", "class_c"],
        },
        classes_misses_by_generator={
            "gen_ab": ["class_c"],
            "gen_bc": ["class_a"],
        },
        mapper_dict={
            "gen_ab": {"class_a": 0, "class_b": 1},
            "gen_bc": {"class_b": 0, "class_c": 1},
        },
    )
    predictions = {
        "gen_ab": np.array([[0.70, 0.20, 0.10], [0.10, 0.40, 0.50]]),
        "gen_bc": np.array([[0.60, 0.30, 0.10], [0.20, 0.50, 0.30]]),
    }

    remapped = strategy._check_predictions(predictions)

    expected = np.array(
        [
            [[0.70, 0.20, 0.10], [0.10, 0.60, 0.30]],
            [[0.10, 0.40, 0.50], [0.30, 0.20, 0.50]],
        ]
    )
    np.testing.assert_allclose(remapped, expected)
