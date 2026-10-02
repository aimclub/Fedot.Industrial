"""Load and validate the data-only Industrial extension catalog."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
from typing import Any, Mapping, NoReturn

from fedot_ind.integration.fedot.extensions.contracts import (
    IndustrialExtensionCatalog,
    IndustrialExtensionContractError,
    IndustrialExtensionErrorCode,
    IndustrialExtensionPlan,
    IndustrialOperationDeclaration,
    IndustrialOperationKind,
    IndustrialRuntimeInterface,
    IndustrialConstructorPolicy,
    IndustrialProbabilityPolicy,
    IndustrialTransformPolicy,
)


CATALOG_SCHEMA_VERSION = 1
CATALOG_RESOURCE = "catalog.json"
RESERVED_BEHAVIOR_TAGS = frozenset({"correct_params"})
SUPPORTED_TASKS = frozenset({"classification", "regression", "ts_forecasting", "clustering"})
SUPPORTED_DATA_TYPES = frozenset({"tabular", "ts", "multi_ts", "text", "image"})
SUPPORTED_BACKENDS = frozenset({"numpy", "torch"})
SUPPORTED_DEVICES = frozenset({"cpu", "cuda"})
SUPPORTED_POSITIONS = frozenset({"any", "primary", "secondary", "root"})
SUPPORTED_PROBLEMS = frozenset({"classification", "regression", "ts_forecasting", "anomaly_detection"})
SUPPORTED_STRUCTURAL_ROLES = frozenset({"resampling", "decomposition", "class_decomposition"})


@lru_cache(maxsize=1)
def load_industrial_extension_catalog() -> IndustrialExtensionCatalog:
    """Load the packaged catalog once and return its immutable representation."""
    resource = files(__package__).joinpath(CATALOG_RESOURCE)
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.CATALOG_NOT_FOUND,
            "Industrial extension catalog is not packaged.",
            context={"resource": str(resource)},
            cause=error,
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.INVALID_CATALOG,
            "Industrial extension catalog cannot be read.",
            context={"resource": str(resource), "cause_type": type(error).__name__},
            cause=error,
        ) from error
    return parse_industrial_extension_catalog(payload)


def load_industrial_extension_catalog_from(path: str | Path) -> IndustrialExtensionCatalog:
    """Load an explicit catalog path for validation tools and tests."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.CATALOG_NOT_FOUND,
            "Industrial extension catalog path does not exist.",
            context={"path": str(source)},
            cause=error,
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.INVALID_CATALOG,
            "Industrial extension catalog path cannot be read.",
            context={"path": str(source), "cause_type": type(error).__name__},
            cause=error,
        ) from error
    return parse_industrial_extension_catalog(payload)


def build_industrial_extension_plan() -> IndustrialExtensionPlan:
    """Prepare a registration plan without importing FEDOT."""
    catalog = load_industrial_extension_catalog()
    return IndustrialExtensionPlan(
        manifest_name=catalog.name,
        manifest_version=catalog.version,
        catalog_digest=catalog.digest,
        model_names=tuple(operation.name for operation in catalog.operations
                          if operation.kind is IndustrialOperationKind.MODEL),
        transform_names=tuple(operation.name for operation in catalog.operations
                              if operation.kind is IndustrialOperationKind.TRANSFORM),
    )


def parse_industrial_extension_catalog(payload: Any) -> IndustrialExtensionCatalog:
    """Validate an external JSON value without importing FEDOT or model modules."""
    root = _mapping(payload, "catalog")
    _require_keys(root, {"schema_version", "manifest", "operations"}, "catalog")
    _reject_unknown_keys(root, {"schema_version", "manifest", "operations"}, "catalog")
    schema_version = root["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        _invalid("Catalog schema_version must be an integer.", field="schema_version")
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.UNSUPPORTED_SCHEMA,
            "Industrial extension catalog schema is not supported.",
            context={"expected": CATALOG_SCHEMA_VERSION, "actual": schema_version},
        )

    manifest = _mapping(root["manifest"], "manifest")
    _require_keys(manifest, {"name", "version", "description"}, "manifest")
    _reject_unknown_keys(manifest, {"name", "version", "description"}, "manifest")
    name = _nonempty_string(manifest["name"], "manifest.name")
    version = _nonempty_string(manifest["version"], "manifest.version")
    description = _nonempty_string(manifest["description"], "manifest.description")

    raw_operations = root["operations"]
    if not isinstance(raw_operations, list) or not raw_operations:
        _invalid("Catalog operations must be a non-empty list.", field="operations")
    operations = tuple(_parse_operation(raw, index) for index, raw in enumerate(raw_operations))
    duplicates = sorted({operation.name for operation in operations if sum(
        candidate.name == operation.name for candidate in operations) > 1})
    if duplicates:
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.DUPLICATE_OPERATION,
            "Industrial extension catalog contains duplicate operation names.",
            context={"operations": duplicates},
        )

    normalized = {
        "schema_version": schema_version,
        "manifest": {"name": name, "version": version, "description": description},
        "operations": [operation.to_dict() for operation in operations],
    }
    digest = sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return IndustrialExtensionCatalog(
        schema_version=schema_version,
        name=name,
        version=version,
        description=description,
        operations=operations,
        digest=digest,
    )


def _parse_operation(raw: Any, index: int) -> IndustrialOperationDeclaration:
    value = _mapping(raw, f"operations[{index}]")
    allowed = {
        "name", "kind", "factory", "tasks", "data_types", "output_data_type", "tags", "problems",
        "defaults_key", "supports_multimodal", "requires_target", "requires_fit", "backend", "description",
        "runtime_interface", "requires_task_type",
        "devices", "serializable", "allowed_positions", "min_parents", "max_parents",
        "allows_identical_parents",
        "constructor_policy", "probability_policy", "transform_policy",
        "structural_role",
    }
    required = {"name", "kind", "factory", "tasks", "data_types", "output_data_type", "tags", "problems"}
    _require_keys(value, required, f"operations[{index}]")
    unknown = sorted(set(value) - allowed)
    if unknown:
        _invalid("Operation declaration contains unknown fields.", index=index, fields=unknown)

    name = _nonempty_string(value["name"], f"operations[{index}].name")
    if name != name.strip() or "/" in name:
        _invalid("Operation name contains unsupported characters.", index=index, operation=name)
    try:
        kind = IndustrialOperationKind(value["kind"])
    except (TypeError, ValueError) as error:
        _invalid("Operation kind is unsupported.", index=index, kind=value.get("kind"), cause=error)
    factory = _nonempty_string(value["factory"], f"operations[{index}].factory")
    if factory.count(":") != 1 or any(not part.strip() for part in factory.split(":")):
        _invalid("Factory must use the 'module:attribute' form.", index=index, factory=factory)

    tasks = _string_tuple(
        value["tasks"], f"operations[{index}].tasks", supported=SUPPORTED_TASKS, index=index)
    data_types = _string_tuple(
        value["data_types"], f"operations[{index}].data_types", supported=SUPPORTED_DATA_TYPES, index=index)
    output_data_type = _nonempty_string(
        value["output_data_type"], f"operations[{index}].output_data_type")
    if output_data_type not in SUPPORTED_DATA_TYPES:
        _invalid("Operation output data type is unsupported.", index=index, data_type=output_data_type)
    tags = _string_tuple(value["tags"], f"operations[{index}].tags", allow_empty=True, index=index)
    reserved_tags = sorted(RESERVED_BEHAVIOR_TAGS.intersection(tags))
    if reserved_tags:
        _invalid("Operation uses reserved FEDOT behavior tags.", index=index, tags=reserved_tags)
    problems = _string_tuple(
        value["problems"], f"operations[{index}].problems", supported=SUPPORTED_PROBLEMS, index=index)

    defaults_key = value.get("defaults_key")
    if defaults_key is not None:
        defaults_key = _nonempty_string(defaults_key, f"operations[{index}].defaults_key")
    backend = value.get("backend", "numpy")
    if backend not in SUPPORTED_BACKENDS:
        _invalid("Operation backend is unsupported.", index=index, backend=backend)
    supports_multimodal = _boolean(value.get("supports_multimodal", False), "supports_multimodal", index)
    requires_target = _boolean(value.get("requires_target", kind is IndustrialOperationKind.MODEL),
                               "requires_target", index)
    requires_fit = _boolean(value.get("requires_fit", True), "requires_fit", index)
    requires_task_type = _boolean(
        value.get("requires_task_type", False),
        "requires_task_type",
        index,
    )
    description_value = value.get("description", "")
    if not isinstance(description_value, str):
        _invalid("Operation description must be a string.", index=index)
    try:
        runtime_interface = IndustrialRuntimeInterface(
            value.get("runtime_interface", IndustrialRuntimeInterface.INPUT_DATA.value)
        )
    except (TypeError, ValueError) as error:
        _invalid(
            "Operation runtime interface is unsupported.",
            index=index,
            runtime_interface=value.get("runtime_interface"),
            cause=error,
        )

    devices = _string_tuple(
        value.get("devices", ["cpu", "cuda"] if backend == "torch" else ["cpu"]),
        f"operations[{index}].devices",
        supported=SUPPORTED_DEVICES,
        index=index,
    )
    serializable = _boolean(value.get("serializable", True), "serializable", index)
    allowed_positions = _string_tuple(
        value.get("allowed_positions", ["any"]),
        f"operations[{index}].allowed_positions",
        supported=SUPPORTED_POSITIONS,
        index=index,
    )
    if "any" in allowed_positions and len(allowed_positions) > 1:
        _invalid(
            "The 'any' position cannot be combined with explicit positions.",
            index=index,
            field="allowed_positions",
        )
    min_parents = _nonnegative_integer(value.get("min_parents", 0), "min_parents", index)
    max_parents = _optional_nonnegative_integer(value.get("max_parents"), "max_parents", index)
    if max_parents is not None and max_parents < min_parents:
        _invalid(
            "Operation max_parents must not be smaller than min_parents.",
            index=index,
            min_parents=min_parents,
            max_parents=max_parents,
        )
    allows_identical_parents = _boolean(
        value.get("allows_identical_parents", True),
        "allows_identical_parents",
        index,
    )
    structural_role = value.get("structural_role")
    if structural_role is not None and (
            not isinstance(structural_role, str)
            or structural_role not in SUPPORTED_STRUCTURAL_ROLES):
        _invalid("Operation structural role is unsupported.",
                 index=index, field="structural_role", value=structural_role)
    policies = {}
    for field, policy_type in (
        ("constructor_policy", IndustrialConstructorPolicy),
        ("probability_policy", IndustrialProbabilityPolicy),
        ("transform_policy", IndustrialTransformPolicy),
    ):
        default = getattr(IndustrialOperationDeclaration, field)
        try:
            policies[field] = policy_type(value.get(field, default))
        except (TypeError, ValueError) as error:
            _invalid("Operation invocation policy is unsupported.",
                     index=index, field=field, value=value.get(field), cause=error)

    return IndustrialOperationDeclaration(
        name=name,
        kind=kind,
        factory=factory,
        tasks=tasks,
        data_types=data_types,
        output_data_type=output_data_type,
        tags=tuple(dict.fromkeys(("industrial", "non-default") + tags)),
        problems=problems,
        defaults_key=defaults_key,
        supports_multimodal=supports_multimodal,
        requires_target=requires_target,
        requires_fit=requires_fit,
        requires_task_type=requires_task_type,
        backend=backend,
        description=description_value,
        runtime_interface=runtime_interface,
        devices=devices,
        serializable=serializable,
        allowed_positions=allowed_positions,
        min_parents=min_parents,
        max_parents=max_parents,
        allows_identical_parents=allows_identical_parents,
        structural_role=structural_role,
        **policies,
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _invalid("Catalog object must be a string-keyed mapping.", field=field)
    return value


def _require_keys(value: Mapping[str, Any], required: set[str], field: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        _invalid("Catalog object is missing required fields.", field=field, missing=missing)


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        _invalid("Catalog object contains unknown fields.", field=field, fields=unknown)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid("Catalog value must be a non-empty string.", field=field)
    return value


def _string_tuple(
        value: Any,
        field: str,
        *,
        supported: frozenset[str] | None = None,
        allow_empty: bool = False,
        index: int | None = None,
) -> tuple[str, ...]:
    context: dict[str, Any] = {"field": field}
    if index is not None:
        context["index"] = index
    if not isinstance(value, list) or (not value and not allow_empty):
        _invalid("Catalog value must be a list of strings.", **context)
    if not all(isinstance(item, str) and item.strip() for item in value):
        _invalid("Catalog list members must be non-empty strings.", **context)
    if len(value) != len(set(value)):
        _invalid("Catalog list contains duplicate values.", **context)
    result = tuple(dict.fromkeys(value))
    unsupported = sorted(set(result) - supported) if supported is not None else []
    if unsupported:
        _invalid("Catalog list contains unsupported values.", values=unsupported, **context)
    return result


def _boolean(value: Any, field: str, index: int) -> bool:
    if not isinstance(value, bool):
        _invalid("Catalog flag must be boolean.", field=field, index=index)
    return value


def _nonnegative_integer(value: Any, field: str, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid("Catalog value must be a non-negative integer.", field=field, index=index)
    return value


def _optional_nonnegative_integer(value: Any, field: str, index: int) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, field, index)


def _invalid(message: str, **context: Any) -> NoReturn:
    cause = context.pop("cause", None)
    raise IndustrialExtensionContractError(
        IndustrialExtensionErrorCode.INVALID_OPERATION
        if "index" in context else IndustrialExtensionErrorCode.INVALID_CATALOG,
        message,
        context=context,
        cause=cause,
    )
