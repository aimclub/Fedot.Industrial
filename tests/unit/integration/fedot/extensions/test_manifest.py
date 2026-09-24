from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

try:
    from fedot.core.repository.operation_types_repository import OperationTypesRepository
    from fedot.core.repository.tasks import TaskTypesEnum
    from fedot.extensions import clear_extension_registry, register_extension, validate_extension_manifest

    from fedot_ind.integration.fedot.extensions.bootstrap import (
        industrial_extension_scope,
        register_industrial_extension,
        resolve_industrial_operation,
    )
    from fedot_ind.integration.fedot.extensions.contracts import (
        IndustrialExtensionErrorCode,
        IndustrialExtensionStatus,
        IndustrialOperationDeclaration,
        IndustrialOperationKind,
        IndustrialRuntimeInterface,
    )
    from fedot_ind.integration.fedot.extensions.factories import make_deferred_factory
    from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest
except ModuleNotFoundError:
    pytest.skip("The active FEDOT profile has no extension contract.", allow_module_level=True)


@pytest.fixture(autouse=True)
def isolated_extension_registry():
    clear_extension_registry()
    yield
    clear_extension_registry()


def test_manifest_is_valid_and_has_unique_operation_names():
    manifest = build_industrial_extension_manifest()
    names = [spec.name for spec in manifest.models + manifest.transforms]

    assert validate_extension_manifest(manifest).is_right()
    assert len(names) == len(set(names))
    assert manifest.models
    assert manifest.transforms


def test_manifest_does_not_shadow_builtin_fedot_operations():
    manifest = build_industrial_extension_manifest()
    extension_names = {spec.name for spec in manifest.models + manifest.transforms}
    builtin_names = {operation.id for operation in OperationTypesRepository("all").operations}

    assert extension_names.isdisjoint(builtin_names)


def test_registration_is_dry_run_safe_and_idempotent():
    planned = register_industrial_extension(dry_run=True)
    registered = register_industrial_extension()
    repeated = register_industrial_extension()

    assert planned.status is IndustrialExtensionStatus.PLANNED
    assert registered.status is IndustrialExtensionStatus.REGISTERED
    assert repeated.status is IndustrialExtensionStatus.ALREADY_REGISTERED
    assert planned.plan == registered.plan == repeated.plan


def test_registration_reports_manifest_conflict():
    manifest = build_industrial_extension_manifest()
    assert register_extension(replace(manifest, version="different")).is_right()

    result = register_industrial_extension()

    assert result.status is IndustrialExtensionStatus.REJECTED
    assert result.error_code == IndustrialExtensionErrorCode.REGISTRATION_CONFLICT.value


def test_dry_run_reports_operation_name_conflict():
    manifest = build_industrial_extension_manifest()
    competitor = replace(
        manifest,
        name="competing_extension",
        models=(manifest.models[0],),
        transforms=(),
    )
    assert register_extension(competitor).is_right()

    result = register_industrial_extension(dry_run=True)

    assert result.status is IndustrialExtensionStatus.REJECTED
    assert result.error_code == IndustrialExtensionErrorCode.REGISTRATION_REJECTED.value
    assert result.error_context["fedot_error_code"] == "operation_name_conflict"


def test_registration_detects_factory_target_change_without_version_change():
    manifest = build_industrial_extension_manifest()

    def replacement_factory(params=None):
        return object()

    changed_model = replace(manifest.models[0], factory=replacement_factory)
    changed_manifest = replace(manifest, models=(changed_model,) + manifest.models[1:])
    assert register_extension(changed_manifest).is_right()

    result = register_industrial_extension()

    assert result.status is IndustrialExtensionStatus.REJECTED
    assert result.error_code == IndustrialExtensionErrorCode.REGISTRATION_CONFLICT.value


def test_extension_scope_restores_registry_and_exposes_operations():
    with industrial_extension_scope() as result:
        names = OperationTypesRepository("model").suitable_operation(
            task_type=TaskTypesEnum.classification,
            tags=["industrial"],
        )
        assert result.status is IndustrialExtensionStatus.REGISTERED
        assert "pdl_clf" in names

    names = OperationTypesRepository("model").suitable_operation(
        task_type=TaskTypesEnum.classification,
        tags=["industrial"],
    )
    assert "pdl_clf" not in names


def test_unknown_operation_returns_typed_error():
    result = resolve_industrial_operation("missing")

    assert result.is_left()
    assert result.monoid[0].code is IndustrialExtensionErrorCode.OPERATION_NOT_FOUND


def test_deferred_factory_does_not_import_runtime_target():
    declaration = IndustrialOperationDeclaration(
        name="deferred",
        kind=IndustrialOperationKind.MODEL,
        factory="missing_runtime_target.module:Model",
        tasks=("classification",),
        data_types=("tabular",),
        output_data_type="tabular",
        tags=("industrial",),
        problems=("classification",),
        runtime_interface=IndustrialRuntimeInterface.ARRAY,
    )

    instance = make_deferred_factory(declaration)({"alpha": 1})

    with pytest.raises(Exception) as error:
        _ = instance.implementation
    assert getattr(error.value, "code", None) is IndustrialExtensionErrorCode.RUNTIME_TARGET_UNAVAILABLE


def test_deferred_factory_uses_keyword_parameters_for_sklearn_style_target():
    declaration = IndustrialOperationDeclaration(
        name="kernel",
        kind=IndustrialOperationKind.MODEL,
        factory="fedot_ind.core.kernel_learning.estimators.classifier:KernelEnsembleClassifier",
        tasks=("classification",),
        data_types=("tabular",),
        output_data_type="tabular",
        tags=("industrial",),
        problems=("classification",),
        runtime_interface=IndustrialRuntimeInterface.ARRAY,
    )

    instance = make_deferred_factory(declaration)(
        {"generator_names": ["identity"], "kernel": "linear"}
    ).implementation

    assert instance.generator_names == ["identity"]
    assert instance.kernel == "linear"


class _ArrayRuntime:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.fit_payload = None

    def fit(self, features, target):
        self.fit_payload = (features, target)
        return self

    def predict(self, features):
        return np.zeros(len(features), dtype=int)

    def predict_proba(self, features):
        return np.tile([0.75, 0.25], (len(features), 1))


class _InputDataRuntime:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.fit_payload = None

    def fit(self, data):
        self.fit_payload = data
        return self

    def predict(self, data, output_mode="labels"):
        values = (np.tile([0.4, 0.6], (len(data.idx), 1))
                  if output_mode == "probs" else np.ones(len(data.idx), dtype=int))
        return SimpleNamespace(predict=values)


class _ForecastRuntime:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.forecast_horizon = None

    def fit(self, data):
        self.forecast_horizon = data.task.task_params.forecast_length
        return self

    def predict(self, data):
        del data
        return SimpleNamespace(predict=np.arange(self.forecast_horizon, dtype=float))


class _InputDataTransformRuntime:
    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, data):
        return self

    def transform(self, data):
        from fedot.core.data.input_data.data import OutputData

        return OutputData(
            idx=data.idx,
            features=data.features,
            predict=np.asarray(data.features) + 10,
            task=data.task,
            target=data.target,
            data_type=data.data_type,
        )


class _FourDimensionalTransformRuntime:
    def fit(self, data):
        return self

    def transform(self, data):
        sample_count = len(data.idx)
        return np.arange(sample_count * 24, dtype=float).reshape(sample_count, 2, 3, 4)


class _ThreeDimensionalTransformRuntime:
    def fit(self, data):
        return self

    def transform(self, data):
        sample_count = len(data.idx)
        return np.arange(sample_count * 6, dtype=float).reshape(sample_count, 2, 3)


@pytest.mark.parametrize(
    ("runtime_interface", "target_name"),
    [
        (IndustrialRuntimeInterface.ARRAY, "ArrayRuntime"),
        (IndustrialRuntimeInterface.INPUT_DATA, "InputDataRuntime"),
    ],
)
def test_deferred_factory_adapts_declared_runtime_interface(
        monkeypatch, runtime_interface, target_name):
    from fedot.core.data.input_data.data import InputData
    from fedot_ind.integration.fedot.extensions import factories

    module = SimpleNamespace(
        ArrayRuntime=_ArrayRuntime,
        InputDataRuntime=_InputDataRuntime,
    )
    monkeypatch.setattr(factories, "import_module", lambda _: module)
    declaration = IndustrialOperationDeclaration(
        name="adapter_test",
        kind=IndustrialOperationKind.MODEL,
        factory=f"example.runtime:{target_name}",
        tasks=("classification",),
        data_types=("tabular",),
        output_data_type="tabular",
        tags=("industrial",),
        problems=("classification",),
        runtime_interface=runtime_interface,
    )
    features = np.arange(12, dtype=float).reshape(6, 2)
    target = np.array([0, 0, 0, 1, 1, 1])
    runtime = make_deferred_factory(declaration)({"depth": 2})

    runtime.fit(features, target)
    probabilities = runtime.predict_proba(features)

    assert probabilities.shape == (6, 2)
    assert runtime.implementation.params == {"depth": 2}
    if runtime_interface is IndustrialRuntimeInterface.ARRAY:
        np.testing.assert_array_equal(runtime.implementation.fit_payload[0], features)
        np.testing.assert_array_equal(runtime.implementation.fit_payload[1], target)
    else:
        assert isinstance(runtime.implementation.fit_payload, InputData)
        np.testing.assert_array_equal(runtime.implementation.fit_payload.target, target)
        assert runtime.implementation.fit_payload.task.task_type is TaskTypesEnum.classification


def test_deferred_forecaster_infers_horizon_and_preserves_tensor_sample_count(monkeypatch):
    from fedot_ind.integration.fedot.extensions import factories

    monkeypatch.setattr(
        factories,
        "import_module",
        lambda _: SimpleNamespace(ForecastRuntime=_ForecastRuntime),
    )
    declaration = IndustrialOperationDeclaration(
        name="forecast_adapter_test",
        kind=IndustrialOperationKind.MODEL,
        factory="example.runtime:ForecastRuntime",
        tasks=("ts_forecasting",),
        data_types=("ts",),
        output_data_type="tabular",
        tags=("industrial",),
        problems=("ts_forecasting",),
        runtime_interface=IndustrialRuntimeInterface.INPUT_DATA,
    )
    runtime = make_deferred_factory(declaration)({})
    features = np.arange(90, dtype=float).reshape(1, -1)
    target = np.arange(5, dtype=float).reshape(1, -1)

    runtime.fit(features, target)
    prediction = runtime.predict(features)

    assert runtime.implementation.forecast_horizon == 5
    assert prediction.shape == (1, 5)


def test_deferred_transform_returns_calculated_output_instead_of_source_features(monkeypatch):
    from fedot_ind.integration.fedot.extensions import factories

    monkeypatch.setattr(
        factories,
        "import_module",
        lambda _: SimpleNamespace(InputDataTransformRuntime=_InputDataTransformRuntime),
    )
    declaration = IndustrialOperationDeclaration(
        name="transform_adapter_test",
        kind=IndustrialOperationKind.TRANSFORM,
        factory="example.runtime:InputDataTransformRuntime",
        tasks=("classification",),
        data_types=("tabular",),
        output_data_type="tabular",
        tags=("industrial",),
        problems=("classification",),
        runtime_interface=IndustrialRuntimeInterface.INPUT_DATA,
    )
    features = np.arange(12, dtype=float).reshape(6, 2)
    runtime = make_deferred_factory(declaration)({})

    runtime.fit(features, np.array([0, 0, 0, 1, 1, 1]))
    transformed = runtime.transform(features)

    np.testing.assert_array_equal(transformed, features + 10)


def test_tabular_transform_flattens_feature_axes_beyond_fedot_runtime_limit(monkeypatch):
    from fedot_ind.integration.fedot.extensions import factories

    monkeypatch.setattr(
        factories,
        "import_module",
        lambda _: SimpleNamespace(FourDimensionalTransformRuntime=_FourDimensionalTransformRuntime),
    )
    declaration = IndustrialOperationDeclaration(
        name="four_dimensional_transform_test",
        kind=IndustrialOperationKind.TRANSFORM,
        factory="example.runtime:FourDimensionalTransformRuntime",
        tasks=("regression",),
        data_types=("image",),
        output_data_type="tabular",
        tags=("industrial",),
        problems=("regression",),
        runtime_interface=IndustrialRuntimeInterface.INPUT_DATA,
    )
    features = np.arange(60, dtype=float).reshape(5, 3, 4)
    runtime = make_deferred_factory(declaration)({})

    runtime.fit(features, np.arange(5, dtype=float))
    transformed = runtime.transform(features)

    assert transformed.shape == (5, 24)


def test_tabular_transform_flattens_three_dimensional_feature_output(monkeypatch):
    from fedot_ind.integration.fedot.extensions import factories

    monkeypatch.setattr(
        factories,
        "import_module",
        lambda _: SimpleNamespace(ThreeDimensionalTransformRuntime=_ThreeDimensionalTransformRuntime),
    )
    declaration = IndustrialOperationDeclaration(
        name="three_dimensional_tabular_transform_test",
        kind=IndustrialOperationKind.TRANSFORM,
        factory="example.runtime:ThreeDimensionalTransformRuntime",
        tasks=("classification",),
        data_types=("image",),
        output_data_type="tabular",
        tags=("industrial",),
        problems=("classification",),
        runtime_interface=IndustrialRuntimeInterface.INPUT_DATA,
    )
    features = np.arange(60, dtype=float).reshape(5, 3, 4)
    runtime = make_deferred_factory(declaration)({})

    runtime.fit(features, np.arange(5, dtype=int))
    transformed = runtime.transform(features)

    assert transformed.shape == (5, 6)


def test_image_transform_preserves_last_axis_within_fedot_runtime_limit(monkeypatch):
    from fedot_ind.integration.fedot.extensions import factories

    monkeypatch.setattr(
        factories,
        "import_module",
        lambda _: SimpleNamespace(FourDimensionalTransformRuntime=_FourDimensionalTransformRuntime),
    )
    declaration = IndustrialOperationDeclaration(
        name="four_dimensional_image_transform_test",
        kind=IndustrialOperationKind.TRANSFORM,
        factory="example.runtime:FourDimensionalTransformRuntime",
        tasks=("classification",),
        data_types=("image",),
        output_data_type="image",
        tags=("industrial",),
        problems=("classification",),
        runtime_interface=IndustrialRuntimeInterface.INPUT_DATA,
    )
    features = np.arange(60, dtype=float).reshape(5, 3, 4)
    runtime = make_deferred_factory(declaration)({})

    runtime.fit(features, np.arange(5, dtype=int))
    transformed = runtime.transform(features)

    assert transformed.shape == (5, 6, 4)


def test_multi_task_model_requires_explicit_task_type(monkeypatch):
    from fedot_ind.integration.fedot.extensions import factories

    monkeypatch.setattr(
        factories,
        "import_module",
        lambda _: SimpleNamespace(InputDataRuntime=_InputDataRuntime),
    )
    declaration = IndustrialOperationDeclaration(
        name="multi_task_adapter_test",
        kind=IndustrialOperationKind.MODEL,
        factory="example.runtime:InputDataRuntime",
        tasks=("classification", "regression"),
        data_types=("tabular",),
        output_data_type="tabular",
        tags=("industrial",),
        problems=("classification", "regression"),
        runtime_interface=IndustrialRuntimeInterface.INPUT_DATA,
    )
    features = np.arange(12, dtype=float).reshape(6, 2)
    integer_regression_target = np.arange(6)

    missing_task = make_deferred_factory(declaration)({})
    with pytest.raises(Exception) as error:
        missing_task.fit(features, integer_regression_target)
    assert error.value.code is IndustrialExtensionErrorCode.RUNTIME_TASK_REQUIRED

    regression = make_deferred_factory(declaration)({"task_type": "regression"})
    regression.fit(features, integer_regression_target)
    assert regression.implementation.fit_payload.task.task_type is TaskTypesEnum.regression


def test_multi_task_model_schema_requires_explicit_task_type():
    manifest = build_industrial_extension_manifest()
    specs = {spec.name: spec for spec in manifest.models}

    assert "task_type" in specs["resnet_model"].hyperparams_schema.required


def test_task_sensitive_transform_schema_requires_explicit_task_type():
    manifest = build_industrial_extension_manifest()
    specs = {spec.name: spec for spec in manifest.transforms}

    assert "task_type" in specs["channel_filtration"].hyperparams_schema.required


@pytest.mark.parametrize(
    ("operation_name", "expected_parameters"),
    [
        ("pdl_clf", {"model", "criterion", "max_depth", "max_pairs", "pairing_policy"}),
        ("pdl_reg", {"model", "alpha", "max_depth", "max_pairs", "pairing_policy"}),
    ],
)
def test_pdl_schema_accepts_all_downstream_and_pairwise_parameters(
        operation_name,
        expected_parameters,
):
    manifest = build_industrial_extension_manifest()
    specs = {spec.name: spec for spec in manifest.models}

    assert expected_parameters.issubset(specs[operation_name].hyperparams_schema.optional)
