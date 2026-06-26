import importlib
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from fedot_ind.core.kernel_learning import (
    BudgetedRepositoryFeatureGeneratorAdapter,
    GeneratorBudgetPolicy,
    KernelFeatureGeneratorMixin,
    OperationSpec,
    RepositoryFeatureGeneratorAdapter,
    ShapeletFeatureGenerator,
    SummaryFeatureGenerator,
    build_generator_registry,
    create_feature_generator,
    resolve_torch_device,
)
from fedot_ind.core.kernel_learning.contracts import FeatureBundle
from fedot_ind.core.kernel_learning.generators import adapters
from fedot_ind.core.kernel_learning.generators import repository as repository_module
from fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding import RiemannExtractor


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


def test_generator_adapters_facade_stays_thin():
    source = Path(adapters.__file__).read_text(encoding="utf-8")

    for module_name in ("base", "lightweight", "registry", "repository", "specs"):
        assert importlib.import_module(f"fedot_ind.core.kernel_learning.generators.{module_name}")

    assert adapters.RepositoryFeatureGeneratorAdapter is RepositoryFeatureGeneratorAdapter
    assert adapters.SummaryFeatureGenerator is SummaryFeatureGenerator
    assert "class RepositoryFeatureGeneratorAdapter" not in source
    assert "def build_generator_registry" not in source


def test_default_registry_exposes_repo_native_generators():
    registry = build_generator_registry()

    for name in (
            "quantile_extractor",
            "wavelet_extractor",
            "fourier_extractor",
            "eigen_extractor",
            "recurrence_extractor",
            "topological_extractor",
            "riemann_extractor",
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
        repository_module,
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


def test_feature_generators_implement_kernel_contract_with_optional_cross_kernel():
    train = np.array([[0.0], [1.0], [2.0]])
    test = np.array([[1.5], [3.0]])
    generator = create_feature_generator("identity")

    train_bundle = generator.kernel(train)
    cross_bundle = generator.kernel(train, test)

    assert train_bundle.name == "identity"
    assert train_bundle.train_kernel.shape == (3, 3)
    assert train_bundle.test_kernel is None
    assert cross_bundle.train_kernel.shape == (3, 3)
    assert cross_bundle.test_kernel.shape == (2, 3)
    assert cross_bundle.train_features.shape == (3, 1)
    assert cross_bundle.test_features.shape == (2, 1)


class _RecordingKernelGenerator(KernelFeatureGeneratorMixin):
    name = "recording"

    def __init__(self):
        self.received_task_type = None

    def fit(self, X, y=None, *, task_type="classification"):
        self.received_task_type = task_type
        return self

    def transform(self, X):
        features = np.asarray(X, dtype=float).reshape(len(X), -1)
        return FeatureBundle(name=self.name, features=features)

    def fit_transform(self, X, y=None, *, task_type="classification"):
        self.fit(X, y, task_type=task_type)
        return self.transform(X)


def test_kernel_contract_forwards_explicit_task_type_to_feature_builder():
    generator = _RecordingKernelGenerator()

    bundle = generator.kernel(np.array([[0.0], [1.0]]), task_type="regression")

    assert generator.received_task_type == "regression"
    assert bundle.train_kernel.shape == (2, 2)


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


def test_tabular_generator_is_budget_controlled_by_default():
    generator = create_feature_generator("tabular_extractor")

    assert isinstance(generator, BudgetedRepositoryFeatureGeneratorAdapter)
    assert generator.budget_policy.max_samples == 25
    assert generator.budget_policy.max_cells == 10_000
    assert generator.budget_policy.fallback_generator == "statistical_summary"


def test_budgeted_adapter_can_use_statistical_summary_fallback():
    pytest.importorskip("fedot")
    pytest.importorskip("torch")
    generator = BudgetedRepositoryFeatureGeneratorAdapter(
        name="tabular_extractor",
        operation_specs=(
            OperationSpec(
                name="tabular_extractor",
                module_path="missing_tabular_module",
                class_name="MissingTabular",
            ),
        ),
        budget_policy=GeneratorBudgetPolicy(max_cells=1, fallback_generator="statistical_summary"),
    )
    X = np.arange(12, dtype=float).reshape(3, 4)

    bundle = generator.fit_transform(X)

    assert bundle.name == "tabular_extractor"
    assert bundle.features.shape[0] == 3
    assert bundle.diagnostics["source"] == "budgeted_fallback"
    assert bundle.diagnostics["requested_generator"] == "tabular_extractor"
    assert bundle.diagnostics["budget"]["fallback_generator"] == "statistical_summary"


def test_resolve_torch_device_auto_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert str(resolve_torch_device("auto")) == "cpu"


def test_resolve_torch_device_auto_prefers_cuda_when_available(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert str(resolve_torch_device("auto")) == "cuda"


def test_riemann_extractor_is_budgeted_repo_adapter():
    generator = create_feature_generator("riemann_extractor")

    assert isinstance(generator, BudgetedRepositoryFeatureGeneratorAdapter)
    assert generator.operation_specs[0].name == "riemann_extractor"


def test_riemann_extractor_adapter_passes_extraction_strategy_param():
    generator = BudgetedRepositoryFeatureGeneratorAdapter(
        name="riemann_extractor",
        operation_specs=(
            OperationSpec(
                name="riemann_extractor",
                module_path="fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding",
                class_name="RiemannExtractor",
                params={"extraction_strategy": "tangent"},
            ),
        ),
        budget_policy=GeneratorBudgetPolicy(max_cells=100, fallback_generator="identity"),
    )
    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )

    generator.fit(X)

    assert generator.operations_[0].extraction_strategy == "tangent"


def test_riemann_extractor_handles_short_series_without_nan_or_inf():
    generator = create_feature_generator("riemann_extractor")
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ]
    )

    features = generator.fit_transform(X).features

    assert features.shape[0] == X.shape[0]
    assert np.all(np.isfinite(features))


def test_riemann_extractor_sanitizes_nan_and_inf_inputs():
    generator = create_feature_generator("riemann_extractor")
    X = np.array(
        [
            [0.0, np.nan, 1.0],
            [np.inf, -1.0, 0.0],
            [1.0, 0.0, 0.5],
        ]
    )

    features = generator.fit_transform(X).features

    assert features.shape[0] == X.shape[0]
    assert np.all(np.isfinite(features))


def test_riemann_extractor_fit_transform_and_transform_are_target_free():
    pytest.importorskip("fedot")
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y_left = np.array([0, 1])
    y_right = np.array([1, 0])

    generator_left = create_feature_generator("riemann_extractor")
    generator_right = create_feature_generator("riemann_extractor")

    left = generator_left.fit_transform(X, y_left).features
    right = generator_right.fit_transform(X, y_right).features

    assert np.allclose(left, right)
    assert np.all(np.isfinite(left))
    assert np.all(np.isfinite(right))
    assert left.shape == right.shape


def test_topological_extractor_fit_transform_and_transform_are_target_free():
    
    pytest.importorskip("fedot")
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y_left = np.array([0, 1])
    y_right = np.array([1, 0])

    generator_left = create_feature_generator("topological_extractor")
    generator_right = create_feature_generator("topological_extractor")

    left = generator_left.fit_transform(X, y_left).features
    right = generator_right.fit_transform(X, y_right).features

    assert np.allclose(left, right)
    assert np.all(np.isfinite(left))
    assert np.all(np.isfinite(right))
    assert left.shape == right.shape


def test_riemann_extractor_output_is_finite_and_has_expected_shape():
    pytest.importorskip("fedot")    
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    generator = create_feature_generator("riemann_extractor")
    features = generator.fit_transform(X).features

    assert np.all(np.isfinite(features))
    assert features.shape == (2, 4)


def test_topological_extractor_output_is_finite_and_has_expected_shape():
    pytest.importorskip("fedot")    
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    generator = create_feature_generator("topological_extractor")
    features = generator.fit_transform(X).features

    assert np.all(np.isfinite(features))
    assert features.shape == (2, 4)


def test_empty_input_in_riemann_extractor_raises_value_error():
    generator = create_feature_generator("riemann_extractor")

    with pytest.raises(ValueError):
        generator.fit_transform(np.array(0), np.array([0]))


def test_empty_input_in_topological_extractor_raises_value_error():

    generator = create_feature_generator("topological_extractor")

    with pytest.raises(ValueError):
        generator.fit_transform(np.array(0), np.array([0]))


def test_budgeted_riemann_adapter_falls_back_on_budget_exceeded():
    generator = BudgetedRepositoryFeatureGeneratorAdapter(
        name="riemann_extractor",
        operation_specs=(
            OperationSpec(
                name="riemann_extractor",
                module_path="missing_riemann_module",
                class_name="MissingRiemann",
            ),
        ),
        budget_policy=GeneratorBudgetPolicy(max_cells=1, fallback_generator="statistical_summary"),
    )
    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 1])

    bundle = generator.fit_transform(X, y)

    assert bundle.name == "riemann_extractor"
    assert bundle.features.shape[0] == 2
    assert bundle.diagnostics["source"] == "budgeted_fallback"


def test_budgeted_topological_adapter_falls_back_on_budget_exceeded():
    generator = BudgetedRepositoryFeatureGeneratorAdapter(
        name="topological_extractor",
        operation_specs=(
            OperationSpec(
                name="topological_extractor",
                module_path="fedot_ind.core.operation.transformation.representation.topological.topological_extractor",
                class_name="TopologicalExtractor",
            ),
        ),
        budget_policy=GeneratorBudgetPolicy(max_cells=1, fallback_generator="statistical_summary"),
    )
    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 1])

    bundle = generator.fit_transform(X, y)

    assert bundle.name == "topological_extractor"
    assert bundle.features.shape[0] == 2
    assert bundle.diagnostics["source"] == "budgeted_fallback"


def test_riemann_extractor_same_for_classification_and_regression_and_ts_forecasting():
    pytest.importorskip("fedot")
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 1])

    gen_clf = create_feature_generator("riemann_extractor")
    out_clf = gen_clf.fit_transform(X, y, task_type="classification").features

    gen_reg = create_feature_generator("riemann_extractor")
    out_reg = gen_reg.fit_transform(X, y, task_type="regression").features

    gen_ts = create_feature_generator("riemann_extractor")
    out_ts = gen_ts.fit_transform(X, y, task_type="ts_forecasting").features

    assert out_clf.shape == out_reg.shape == out_ts.shape
    assert np.all(np.isfinite(out_clf))
    assert np.all(np.isfinite(out_reg))
    assert np.all(np.isfinite(out_ts))
    assert np.allclose(out_clf, out_reg)
    assert np.allclose(out_clf, out_ts)


def test_topological_extractor_same_for_classification_and_regression_and_ts_forecasting():
    pytest.importorskip("fedot")
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 1])

    gen_clf = create_feature_generator("topological_extractor")
    out_clf = gen_clf.fit_transform(X, y, task_type="classification").features

    gen_reg = create_feature_generator("topological_extractor")
    out_reg = gen_reg.fit_transform(X, y, task_type="regression").features

    gen_ts = create_feature_generator("topological_extractor")
    out_ts = gen_ts.fit_transform(X, y, task_type="ts_forecasting").features

    assert out_clf.shape == out_reg.shape == out_ts.shape
    assert np.all(np.isfinite(out_clf))
    assert np.all(np.isfinite(out_reg))
    assert np.all(np.isfinite(out_ts))
    assert np.allclose(out_clf, out_reg)


def test_budgeted_riemann_adapter_diagnostics_include_operation_params():
    generator = BudgetedRepositoryFeatureGeneratorAdapter(
        name="riemann_extractor",
        operation_specs=(
            OperationSpec(
                name="riemann_extractor",
                module_path="fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding",
                class_name="RiemannExtractor",
                params={"SPD_metric": "logeuclid", "tangent_metric": "euclid"},
            ),
        ),
        budget_policy=GeneratorBudgetPolicy(max_cells=100, fallback_generator="statistical_summary"),
    )
    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 1])

    bundle = generator.fit_transform(X, y)

    assert bundle.name == "riemann_extractor"
    assert bundle.features.shape[0] == 2
    assert bundle.diagnostics["SPD_metric"] == "logeuclid"
    assert bundle.diagnostics["tangent_metric"] == "euclid"
    assert bundle.diagnostics['estimator'] == 'scm'


@pytest.mark.parametrize("invalid_params, expected_error_match", [
    (
        {"extraction_strategy": "magic_method"}, 
        "Unsupported extraction strategy: 'magic_method'"
    ),
    (
        {"estimator": "pearson"}, 
        "Unsupported estimator: 'pearson'"
    ),
    (
        {"SPD_metric": "manhattan"}, 
        "Unsupported SPD_metric: 'manhattan'"
    ),
    (
        {"tangent_metric": "cosine"}, 
        "Unsupported tangent_metric: 'cosine'"
    ),
])
def test_riemann_extractor_incorrect_params_raise_value_error(invalid_params, expected_error_match):

    with pytest.raises(ValueError, match=expected_error_match):
        RiemannExtractor(invalid_params)



















