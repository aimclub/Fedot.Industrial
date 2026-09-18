import json
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
from fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding import RiemannExtractor
from fedot_ind.core.tuning.search_space import industrial_search_space
from fedot_ind.tools.serialisation.path_lib import PATH_TO_DEFAULT_PARAMS


def _raw_riemann_view() -> dict:
    """Return the canonical raw-covariance view used by adapter tests."""
    return {
        "name": "raw",
        "builder": "covariance",
        "weight": 1.0,
        "shrinkage": 0.1,
        "params": {"estimator": "scm", "estimator_params": {}},
    }


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


def test_repository_adapter_uses_native_fit_transform_for_stateful_operation(monkeypatch):
    """Opt-in operations must own fitting while inference keeps using transform."""
    class StatefulOperation:
        def __init__(self, params):
            """Store configured device and initialise lifecycle counters."""
            self.torch_device = params.get("torch_device")
            self.fit_transform_calls = 0
            self.transform_calls = 0
            self.fit_target = None

        def fit_transform(self, input_data):
            """Record fitting and return deterministic train features."""
            self.fit_transform_calls += 1
            self.fit_target = np.asarray(input_data.target).reshape(-1)
            return SimpleNamespace(predict=np.asarray(input_data.features) + 1.0)

        def transform(self, input_data):
            """Record inference and return distinguishable features."""
            self.transform_calls += 1
            return SimpleNamespace(predict=np.asarray(input_data.features) + 2.0)

    fake_module = types.ModuleType("fake_stateful_kernel_learning_ops")
    fake_module.StatefulOperation = StatefulOperation
    monkeypatch.setitem(sys.modules, "fake_stateful_kernel_learning_ops", fake_module)

    generator = RepositoryFeatureGeneratorAdapter(
        name="stateful_repo_generator",
        operation_specs=(
            OperationSpec(
                name="stateful_op",
                module_path="fake_stateful_kernel_learning_ops",
                class_name="StatefulOperation",
                params={"torch_device": "cuda"},
                use_torch=True,
                fit_transform_on_fit=True,
            ),
        ),
        torch_device="cpu",
    )
    X = np.arange(8, dtype=float).reshape(2, 4)
    y = np.array([0, 1])

    train = generator.fit_transform(X, y).features
    inference = generator.transform(X).features
    operation = generator.operations_[0]

    assert operation.fit_transform_calls == 1
    assert operation.transform_calls == 1
    np.testing.assert_array_equal(operation.fit_target, y)
    assert operation.torch_device == "cpu"
    assert generator.resolved_torch_device_ == "cpu"
    np.testing.assert_allclose(train, X + 1.0)
    np.testing.assert_allclose(inference, X + 2.0)


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


def test_budgeted_adapter_reuses_train_features_after_fallback_refit(monkeypatch):
    """A successful refit must clear fallback state and avoid train inference."""
    class StatefulOperation:
        def __init__(self, params):
            """Initialise lifecycle counters for the budgeted operation."""
            del params
            self.fit_transform_calls = 0
            self.transform_calls = 0

        def fit_transform(self, input_data):
            """Record fitting and return deterministic train features."""
            self.fit_transform_calls += 1
            return SimpleNamespace(predict=np.asarray(input_data.features) + 1.0)

        def transform(self, input_data):
            """Record inference and return distinguishable features."""
            self.transform_calls += 1
            return SimpleNamespace(predict=np.asarray(input_data.features) + 2.0)

    fake_module = types.ModuleType("fake_budgeted_stateful_ops")
    fake_module.StatefulOperation = StatefulOperation
    monkeypatch.setitem(sys.modules, "fake_budgeted_stateful_ops", fake_module)

    generator = BudgetedRepositoryFeatureGeneratorAdapter(
        name="budgeted_stateful_generator",
        operation_specs=(
            OperationSpec(
                name="stateful_op",
                module_path="fake_budgeted_stateful_ops",
                class_name="StatefulOperation",
                fit_transform_on_fit=True,
            ),
        ),
        budget_policy=GeneratorBudgetPolicy(max_cells=1, fallback_generator="identity"),
    )
    X = np.arange(8, dtype=float).reshape(2, 4)

    fallback = generator.fit_transform(X)
    generator.budget_policy = GeneratorBudgetPolicy(
        max_cells=100,
        fallback_generator="identity",
    )
    primary = generator.fit_transform(X)
    operation = generator.operations_[0]

    assert fallback.diagnostics["source"] == "budgeted_fallback"
    assert primary.diagnostics["source"] == "fedot_industrial_operation"
    assert generator.fallback_generator_ is None
    assert operation.fit_transform_calls == 1
    assert operation.transform_calls == 0
    np.testing.assert_allclose(primary.features, X + 1.0)


def test_resolve_torch_device_auto_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert str(resolve_torch_device("auto")) == "cpu"


def test_resolve_torch_device_auto_prefers_cuda_when_available(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert str(resolve_torch_device("auto")) == "cuda"


def test_riemann_extractor_is_budgeted_repo_adapter():
    """The registry must expose the stateful Torch multi-view operation."""
    generator = create_feature_generator("riemann_extractor")
    spec = generator.operation_specs[0]

    assert isinstance(generator, BudgetedRepositoryFeatureGeneratorAdapter)
    assert spec.name == "riemann_extractor"
    assert spec.use_torch is True
    assert spec.fit_transform_on_fit is True
    assert spec.params["views"] == [_raw_riemann_view()]
    assert spec.params["feature_mode"] == "both"
    assert spec.params["mdm_centroid_scope"] == "class"
    assert spec.params["torch_dtype"] == "float64"


def test_riemann_extractor_adapter_passes_multi_view_params():
    """The adapter must instantiate and fit RiemannExtractor from the new schema."""
    generator = BudgetedRepositoryFeatureGeneratorAdapter(
        name="riemann_extractor",
        operation_specs=(
            OperationSpec(
                name="riemann_extractor",
                module_path="fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding",
                class_name="RiemannExtractor",
                params={
                    "views": [_raw_riemann_view()],
                    "feature_mode": "tangent",
                    "torch_device": "cpu",
                    "torch_dtype": "float64",
                },
                use_torch=True,
                fit_transform_on_fit=True,
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

    operation = generator.operations_[0]
    assert generator.fallback_generator_ is None
    assert operation.feature_mode == "tangent"
    assert operation.views == (_raw_riemann_view(),)
    assert str(operation.dtype) == "torch.float64"


def test_riemann_default_configs_use_multi_view_contract():
    """Registry, FEDOT defaults, and tuning space must share the new schema."""
    spec = create_feature_generator("riemann_extractor").operation_specs[0]
    with open(PATH_TO_DEFAULT_PARAMS, encoding="utf-8") as params_file:
        default_params = json.load(params_file)["riemann_extractor"]

    assert default_params == spec.params
    assert set(industrial_search_space["riemann_extractor"]) == {
        "tangent_metric",
        "mdm_metric",
    }


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
    """Default class MDM must run through the primary registry operation."""
    pytest.importorskip("fedot")    
    pytest.importorskip("torch")

    X = np.random.default_rng(42).normal(size=(4, 2, 16))
    y = np.array([0, 0, 1, 1])
    generator = create_feature_generator("riemann_extractor")
    bundle = generator.fit_transform(X, y)

    assert bundle.diagnostics["source"] == "fedot_industrial_operation"
    assert np.all(np.isfinite(bundle.features))
    assert bundle.features.shape == (4, 5)


def test_default_riemann_class_mdm_without_target_uses_budgeted_fallback():
    """The budget wrapper must retain its fallback contract for missing target."""
    X = np.random.default_rng(42).normal(size=(4, 2, 16))

    bundle = create_feature_generator("riemann_extractor").fit_transform(X)

    assert bundle.diagnostics["source"] == "budgeted_fallback"
    assert bundle.diagnostics["budget"]["skip_reason"] == "operation_unavailable:ValueError"


def test_topological_extractor_output_is_finite_and_has_expected_shape():
    """Default topology must return H0/H1 statistics from the primary operation."""
    pytest.importorskip("fedot")    
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    generator = create_feature_generator("topological_extractor")
    bundle = generator.fit_transform(X)

    assert bundle.diagnostics["source"] == "fedot_industrial_operation"
    assert np.all(np.isfinite(bundle.features))
    assert bundle.features.shape == (2, 20)


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
    """Diagnostics must expose the effective multi-view Riemann configuration."""
    generator = BudgetedRepositoryFeatureGeneratorAdapter(
        name="riemann_extractor",
        operation_specs=(
            OperationSpec(
                name="riemann_extractor",
                module_path="fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding",
                class_name="RiemannExtractor",
                params={
                    "views": [_raw_riemann_view()],
                    "feature_mode": "both",
                    "tangent_metric": "euclid",
                    "mdm_metric": "logeuclid",
                    "mdm_centroid_scope": "class",
                    "torch_device": "cpu",
                    "torch_dtype": "float64",
                },
                use_torch=True,
                fit_transform_on_fit=True,
            ),
        ),
        budget_policy=GeneratorBudgetPolicy(max_cells=1_000, fallback_generator="statistical_summary"),
    )
    X = np.random.default_rng(42).normal(size=(4, 2, 16))
    y = np.array([0, 0, 1, 1])

    bundle = generator.fit_transform(X, y)

    assert bundle.name == "riemann_extractor"
    assert bundle.features.shape[0] == 4
    assert bundle.diagnostics["source"] == "fedot_industrial_operation"
    assert bundle.diagnostics["views"] == (_raw_riemann_view(),)
    assert bundle.diagnostics["mdm_metric"] == "logeuclid"
    assert bundle.diagnostics["tangent_metric"] == "euclid"
    assert bundle.diagnostics["feature_mode"] == "both"





















