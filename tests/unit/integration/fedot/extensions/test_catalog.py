import ast
from copy import deepcopy
from pathlib import Path

import pytest

from fedot_ind.integration.fedot.extensions.catalog import (
    load_industrial_extension_catalog,
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
