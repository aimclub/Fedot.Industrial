from types import SimpleNamespace

import numpy as np
import pytest

industrial_strategy_module = pytest.importorskip("fedot_ind.api.utils.industrial_strategy")


def test_kernel_strategy_uses_legacy_path_when_warm_start_flag_is_disabled(monkeypatch):
    calls = []
    strategy = industrial_strategy_module.IndustrialStrategy(
        industrial_strategy_params={},
        industrial_strategy="kernel_automl",
        api_config={"problem": "classification"},
    )
    monkeypatch.setattr(
        strategy,
        "_legacy_kernel_strategy",
        lambda input_data: calls.append(input_data),
    )

    input_data = SimpleNamespace()
    strategy._kernel_strategy(input_data)

    assert calls == [input_data]


def test_kernel_warm_start_passes_initial_population_to_fedot(monkeypatch):
    import fedot_ind.core.kernel_learning as kernel_learning_module
    import fedot_ind.core.kernel_learning.integration as integration_module

    created_fedot = []

    class FakeEstimator:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            self.fit_shape_ = np.asarray(X).shape
            self.fit_target_ = np.asarray(y)
            self.kernel_importance_ = SimpleNamespace(selected_generators=("wavelet_basis",))
            return self

    class FakeBuilder:
        def __init__(self, **params):
            self.params = params
            self.diagnostics_ = {"head_model": params["head_model"]}

        def build_pipelines(self, importance):
            self.importance_ = importance
            return ["wavelet_rf", "union_rf"]

        def restrict_available_operations(self, available_operations):
            self.available_operations_ = available_operations
            return ["wavelet_basis", "quantile_extractor_torch", "rf"]

    class FakeFedot:
        def __init__(self, **config):
            self.config = config
            created_fedot.append(self)

        def fit(self, input_data):
            self.fit_input_ = input_data

    monkeypatch.setattr(kernel_learning_module, "KernelEnsembleClassifier", FakeEstimator)
    monkeypatch.setattr(integration_module, "KernelInitialPopulationBuilder", FakeBuilder)
    monkeypatch.setattr(industrial_strategy_module, "Fedot", FakeFedot)
    strategy = industrial_strategy_module.IndustrialStrategy(
        industrial_strategy_params={
            "use_kernel_warm_start": True,
            "restrict_search_space": True,
            "importance_threshold": 0.2,
            "max_union_size": 2,
            "kernel_learning_params": {"generator_names": ("wavelet_basis",)},
        },
        industrial_strategy="kernel_automl",
        api_config={
            "problem": "classification",
            "available_operations": ["wavelet_basis", "rf", "topological_extractor"],
        },
    )

    input_data = SimpleNamespace(features=np.zeros((3, 4)), target=np.array([0, 1, 0]))
    strategy._kernel_strategy(input_data)

    assert created_fedot[0].config["initial_assumption"] == ["wavelet_rf", "union_rf"]
    assert created_fedot[0].config["available_operations"] == ["wavelet_basis", "quantile_extractor_torch", "rf"]
    assert strategy.kernel_learning_estimator_.params["importance_threshold"] == 0.2
    assert strategy.kernel_initial_population_ == ["wavelet_rf", "union_rf"]


def test_kernel_warm_start_search_space_narrowing_can_be_disabled(monkeypatch):
    import fedot_ind.core.kernel_learning as kernel_learning_module
    import fedot_ind.core.kernel_learning.integration as integration_module

    created_fedot = []

    class FakeEstimator:
        def __init__(self, **params):
            pass

        def fit(self, X, y):
            self.kernel_importance_ = SimpleNamespace(selected_generators=("wavelet_basis",))
            return self

    class FakeBuilder:
        diagnostics_ = {}

        def __init__(self, **params):
            pass

        def build_pipelines(self, importance):
            return ["wavelet_rf"]

        def restrict_available_operations(self, available_operations):
            raise AssertionError("restrict_available_operations should not be called")

    class FakeFedot:
        def __init__(self, **config):
            self.config = config
            created_fedot.append(self)

        def fit(self, input_data):
            pass

    monkeypatch.setattr(kernel_learning_module, "KernelEnsembleClassifier", FakeEstimator)
    monkeypatch.setattr(integration_module, "KernelInitialPopulationBuilder", FakeBuilder)
    monkeypatch.setattr(industrial_strategy_module, "Fedot", FakeFedot)
    strategy = industrial_strategy_module.IndustrialStrategy(
        industrial_strategy_params={"use_kernel_warm_start": True},
        industrial_strategy="kernel_automl",
        api_config={"problem": "classification", "available_operations": ["rf", "topological_extractor"]},
    )

    strategy._kernel_strategy(SimpleNamespace(features=np.zeros((3, 4)), target=np.array([0, 1, 0])))

    assert created_fedot[0].config["available_operations"] == ["rf", "topological_extractor"]
