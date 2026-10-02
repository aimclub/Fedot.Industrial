import ast
from copy import deepcopy
from pathlib import Path

import pytest

from fedot_ind.integration.fedot.extensions.catalog import (
    load_industrial_extension_catalog,
    load_industrial_extension_catalog_from,
    parse_industrial_extension_catalog,
)
from fedot_ind.integration.fedot.extensions.contracts import (
    IndustrialExtensionContractError,
    IndustrialExtensionErrorCode,
)


def _payload():
    return {
        "schema_version": 1,
        "manifest": {"name": "example", "version": "1", "description": "Example."},
        "operations": [{
            "name": "example_model",
            "kind": "model",
            "factory": "example.module:Model",
            "tasks": ["classification"],
            "data_types": ["tabular"],
            "output_data_type": "tabular",
            "tags": ["example"],
            "problems": ["classification"],
        }],
    }


def test_catalog_parsing_is_deterministic_and_adds_boundary_tags():
    first = parse_industrial_extension_catalog(_payload())
    second = parse_industrial_extension_catalog(deepcopy(_payload()))

    assert first == second
    assert first.digest == second.digest
    assert first.operations[0].tags == ("industrial", "non-default", "example")
    assert first.operations[0].devices == ("cpu",)
    assert first.operations[0].serializable is True
    assert first.operations[0].allowed_positions == ("any",)
    assert first.operations[0].min_parents == 0
    assert first.operations[0].max_parents is None


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda value: value["operations"].append(deepcopy(value["operations"][0])),
         IndustrialExtensionErrorCode.DUPLICATE_OPERATION),
        (lambda value: value["operations"][0]["tags"].append("correct_params"),
         IndustrialExtensionErrorCode.INVALID_OPERATION),
        (lambda value: value.update(schema_version=2),
         IndustrialExtensionErrorCode.UNSUPPORTED_SCHEMA),
    ],
)
def test_catalog_rejects_ambiguous_contracts(mutate, expected_code):
    payload = _payload()
    mutate(payload)

    with pytest.raises(IndustrialExtensionContractError) as error:
        parse_industrial_extension_catalog(payload)

    assert error.value.code is expected_code


def test_catalog_rejects_unknown_fields():
    payload = _payload()
    payload["operations"][0]["implicit_behavior"] = True

    with pytest.raises(IndustrialExtensionContractError) as error:
        parse_industrial_extension_catalog(payload)

    assert error.value.code is IndustrialExtensionErrorCode.INVALID_OPERATION


def test_catalog_rejects_unknown_runtime_interface():
    payload = _payload()
    payload["operations"][0]["runtime_interface"] = "dataframe"

    with pytest.raises(IndustrialExtensionContractError) as error:
        parse_industrial_extension_catalog(payload)

    assert error.value.code is IndustrialExtensionErrorCode.INVALID_OPERATION
    assert error.value.context["runtime_interface"] == "dataframe"


@pytest.mark.parametrize("field", ["constructor_policy", "probability_policy", "transform_policy"])
def test_catalog_rejects_unknown_invocation_policy(field):
    payload = _payload()
    payload["operations"][0][field] = "guess_from_signature"

    with pytest.raises(IndustrialExtensionContractError) as error:
        parse_industrial_extension_catalog(payload)

    assert error.value.code is IndustrialExtensionErrorCode.INVALID_OPERATION
    assert error.value.context["field"] == field


def test_catalog_validates_structural_role_without_importing_graph_runtime():
    payload = _payload()
    payload["operations"][0]["structural_role"] = "resampling"
    assert parse_industrial_extension_catalog(payload).operations[0].structural_role == "resampling"

    payload["operations"][0]["structural_role"] = ["resampling"]
    with pytest.raises(IndustrialExtensionContractError) as error:
        parse_industrial_extension_catalog(payload)
    assert error.value.code is IndustrialExtensionErrorCode.INVALID_OPERATION


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("devices", ["tpu"]),
        ("serializable", "yes"),
        ("allowed_positions", ["any", "root"]),
        ("min_parents", -1),
        ("max_parents", False),
    ],
)
def test_catalog_rejects_invalid_graph_capabilities(field, value):
    payload = _payload()
    payload["operations"][0][field] = value

    with pytest.raises(IndustrialExtensionContractError) as error:
        parse_industrial_extension_catalog(payload)

    assert error.value.code is IndustrialExtensionErrorCode.INVALID_OPERATION


def test_catalog_rejects_inverted_parent_limits():
    payload = _payload()
    payload["operations"][0].update(min_parents=2, max_parents=1)

    with pytest.raises(IndustrialExtensionContractError) as error:
        parse_industrial_extension_catalog(payload)

    assert error.value.code is IndustrialExtensionErrorCode.INVALID_OPERATION


def test_catalog_rejects_unknown_root_fields_and_duplicate_values():
    unknown = _payload()
    unknown["implicit_defaults"] = {}
    duplicate = _payload()
    duplicate["operations"][0]["tasks"].append("classification")

    with pytest.raises(IndustrialExtensionContractError) as unknown_error:
        parse_industrial_extension_catalog(unknown)
    with pytest.raises(IndustrialExtensionContractError) as duplicate_error:
        parse_industrial_extension_catalog(duplicate)

    assert unknown_error.value.code is IndustrialExtensionErrorCode.INVALID_CATALOG
    assert duplicate_error.value.code is IndustrialExtensionErrorCode.INVALID_OPERATION


def test_packaged_runtime_targets_exist_without_importing_them():
    root = Path(__file__).resolve().parents[5]

    for operation in load_industrial_extension_catalog().operations:
        module_name, attribute_name = operation.factory.split(":", maxsplit=1)
        module_path = root.joinpath(*module_name.split(".")).with_suffix(".py")
        assert module_path.is_file(), operation.factory
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        declarations = {node.name for node in ast.walk(tree)
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
        assert attribute_name in declarations, operation.factory


def test_basis_transforms_accept_legacy_table_series_as_tabular_data():
    operations = {
        operation.name: operation
        for operation in load_industrial_extension_catalog().operations
    }

    for operation_name in ("eigen_basis", "wavelet_basis", "fourier_basis"):
        assert "tabular" in operations[operation_name].data_types


def test_explicit_defaults_and_mapping_order_do_not_change_catalog_identity():
    payload = _payload()
    original = parse_industrial_extension_catalog(payload)
    payload["operations"][0].update(
        devices=["cpu"], serializable=True, allowed_positions=["any"],
        min_parents=0, max_parents=None,
    )
    reordered = dict(reversed(tuple(payload.items())))

    assert parse_industrial_extension_catalog(reordered) == original


@pytest.mark.parametrize("change", [
    {"factory": "example.module:Replacement"},
    {"serializable": False},
    {"devices": ["cpu", "cuda"]},
    {"runtime_interface": "array"},
])
def test_behavior_changes_invalidate_catalog_digest_without_mutating_prior_catalog(change):
    payload = _payload()
    original = parse_industrial_extension_catalog(payload)
    payload["operations"][0].update(change)

    updated = parse_industrial_extension_catalog(payload)

    assert updated.digest != original.digest
    assert original == parse_industrial_extension_catalog(_payload())


@pytest.mark.parametrize("contents", ["{", "[]", "null"])
def test_explicit_catalog_file_rejects_invalid_json_or_root(tmp_path, contents):
    path = tmp_path / "catalog.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(IndustrialExtensionContractError) as error:
        load_industrial_extension_catalog_from(path)

    assert error.value.code is IndustrialExtensionErrorCode.INVALID_CATALOG
