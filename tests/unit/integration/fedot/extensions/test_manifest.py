from dataclasses import replace

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
        assert "industrial_stat_clf" in names

    names = OperationTypesRepository("model").suitable_operation(
        task_type=TaskTypesEnum.classification,
        tags=["industrial"],
    )
    assert "industrial_stat_clf" not in names


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
    )

    instance = make_deferred_factory(declaration)({"alpha": 1})

    with pytest.raises(Exception) as error:
        _ = instance.implementation
    assert getattr(error.value, "code", None) is IndustrialExtensionErrorCode.RUNTIME_TARGET_UNAVAILABLE


@pytest.mark.parametrize(
    ("operation_name", "expected_parameters"),
    [
        ("pdl_clf", {"model", "criterion", "max_features"}),
        ("pdl_reg", {"model", "max_features", "min_samples_split"}),
    ],
)
def test_pdl_schema_accepts_inherited_search_parameters(operation_name, expected_parameters):
    manifest = build_industrial_extension_manifest()
    specs = {spec.name: spec for spec in manifest.models}

    assert expected_parameters.issubset(specs[operation_name].hyperparams_schema.optional)
