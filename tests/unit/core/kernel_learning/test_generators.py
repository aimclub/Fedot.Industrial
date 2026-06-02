import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from fedot_ind.core.kernel_learning import (
    BudgetedRepositoryFeatureGeneratorAdapter,
    GeneratorBudgetPolicy,
    OperationSpec,
    RepositoryFeatureGeneratorAdapter,
    ShapeletFeatureGenerator,
    SummaryFeatureGenerator,
    build_generator_registry,
    create_feature_generator,
    resolve_torch_device,
)
from fedot_ind.core.kernel_learning.generators import adapters


def test_statistical_summary_is_repo_native_adapter_not_manual_summary():
    generator = SummaryFeatureGenerator()

    assert isinstance(generator, RepositoryFeatureGeneratorAdapter)
    assert not hasattr(generator, "_build_features")
    assert generator.operation_specs[0].name == "quantile_extractor_torch"


def test_statistical_summary_handles_single_timestamp_batches():
    pytest.importorskip("fedot")
    pytest.importorskip("torch")
    generator = SummaryFeatureGenerator()
    X = np.array([[0.0], [0.2], [1.0], [1.2]])
    y = np.array([0, 0, 1, 1])

    train_features = generator.fit_transform(X, y).features
    test_features = generator.transform(np.array([[0.1], [1.1]])).features

    assert train_features.shape[0] == 4
    assert test_features.shape[0] == 2
    assert train_features.shape[1] == test_features.shape[1]
    assert np.all(np.isfinite(train_features))
    assert np.all(np.isfinite(test_features))


def test_default_registry_exposes_repo_native_generators():
    registry = build_generator_registry()

    for name in (
            "quantile_extractor",
            "wavelet_extractor",
            "fourier_extractor",
            "eigen_extractor",
            "recurrence_extractor",
            "topological_extractor",
            "tabular_extractor",
            "shapelet_extractor",
            "embedding_extractor",
            "foundation_embedding",
    ):
        assert name in registry

    assert create_feature_generator("wavelet_extractor").operation_specs[0].module_path.endswith("basis.wavelet")
    assert create_feature_generator("fourier_extractor").operation_specs[0].module_path.endswith("basis.fourier")
    assert create_feature_generator("eigen_extractor").operation_specs[0].module_path.endswith("basis.eigen_basis")


def test_repository_feature_generator_adapter_is_deterministic_and_target_free(monkeypatch):
    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y_left = np.array([0, 1])
    y_right = np.array([1, 0])

    class FakeOperation:
        def __init__(self, params):
            self.scale = params.get("scale", 1.0)

        def transform(self, input_data, use_cache=False):
            del use_cache
            features = np.asarray(input_data.features, dtype=float).reshape(input_data.features.shape[0], -1)
            return SimpleNamespace(predict=features * self.scale)

    fake_module = types.ModuleType("fake_kernel_learning_ops")
    fake_module.FakeOperation = FakeOperation
    monkeypatch.setitem(sys.modules, "fake_kernel_learning_ops", fake_module)
    monkeypatch.setattr(
        adapters,
        "to_fedot_input_data",
        lambda X, y=None, task_type="classification", use_torch=False, torch_device="auto": SimpleNamespace(
            features=np.asarray(X),
            target=None if y is None else np.asarray(y).reshape(-1, 1),
            idx=np.arange(np.asarray(X).shape[0]),
            task=task_type,
            supplementary_data=None,
        ),
    )

    generator = RepositoryFeatureGeneratorAdapter(
        name="fake_repo_generator",
        operation_specs=(
            OperationSpec(
                name="fake_op",
                module_path="fake_kernel_learning_ops",
                class_name="FakeOperation",
                params={"scale": 2.0},
            ),
        ),
    )
    left = generator.fit_transform(X, y_left).features
    right = RepositoryFeatureGeneratorAdapter(
        name="fake_repo_generator",
        operation_specs=generator.operation_specs,
    ).fit_transform(X, y_right).features

    assert left.shape == (2, 4)
    assert np.allclose(left, right)
    assert np.all(np.isfinite(left))


def test_shapelet_generator_is_deterministic_and_target_free():
    X = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 3.0, 2.0, 2.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
        ]
    )

    left = ShapeletFeatureGenerator(n_shapelets=3, window_size=2).fit_transform(X, np.array([0, 1, 0])).features
    right = ShapeletFeatureGenerator(n_shapelets=3, window_size=2).fit_transform(X, np.array([1, 0, 1])).features

    assert left.shape == (3, 3)
    assert np.allclose(left, right)
    assert np.all(np.isfinite(left))


def test_embedding_generator_is_deterministic_under_seed():
    X = np.arange(12, dtype=float).reshape(3, 4)

    left = create_feature_generator("embedding_extractor").fit_transform(X).features
    right = create_feature_generator("embedding_extractor").fit_transform(X).features

    assert left.shape == (3, 16)
    assert np.allclose(left, right)


def test_budgeted_topology_adapter_falls_back_without_importing_heavy_operation():
    generator = BudgetedRepositoryFeatureGeneratorAdapter(
        name="topological_extractor",
        operation_specs=(
            OperationSpec(
                name="topological_extractor",
                module_path="missing_topology_module",
                class_name="MissingTopology",
            ),
        ),
        budget_policy=GeneratorBudgetPolicy(max_cells=1, fallback_generator="identity"),
    )
    X = np.zeros((2, 3))

    bundle = generator.fit_transform(X)

    assert bundle.name == "topological_extractor"
    assert bundle.features.shape == (2, 3)
    assert bundle.diagnostics["source"] == "budgeted_fallback"
    assert bundle.diagnostics["budget"]["skip_reason"] == "budget_exceeded"


def test_resolve_torch_device_auto_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert str(resolve_torch_device("auto")) == "cpu"


def test_resolve_torch_device_auto_prefers_cuda_when_available(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert str(resolve_torch_device("auto")) == "cuda"
