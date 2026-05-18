import sys
import types
from types import SimpleNamespace

import numpy as np

from fedot_ind.core.kernel_learning import (
    OperationSpec,
    RepositoryFeatureGeneratorAdapter,
    SummaryFeatureGenerator,
    build_generator_registry,
    create_feature_generator,
)
from fedot_ind.core.kernel_learning.generators import adapters


def test_statistical_summary_is_repo_native_adapter_not_manual_summary():
    generator = SummaryFeatureGenerator()

    assert isinstance(generator, RepositoryFeatureGeneratorAdapter)
    assert not hasattr(generator, "_build_features")
    assert generator.operation_specs[0].name == "quantile_extractor_torch"


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
        lambda X, y=None, task_type="classification", use_torch=False: SimpleNamespace(
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
