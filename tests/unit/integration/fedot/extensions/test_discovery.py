from types import SimpleNamespace

import pytest

try:
    from fedot_ind.integration.fedot.extensions import discovery
except ModuleNotFoundError:
    pytest.skip("The active FEDOT profile has no extension contract.", allow_module_level=True)


def test_tabular_data_type_supports_legacy_and_tensor_names(monkeypatch):
    legacy_table = object()
    monkeypatch.setattr(discovery, "DataTypesEnum", SimpleNamespace(table=legacy_table))
    assert discovery._tabular_data_type() is legacy_table

    tensor_table = object()
    monkeypatch.setattr(discovery, "DataTypesEnum", SimpleNamespace(tabular=tensor_table))
    assert discovery._tabular_data_type() is tensor_table


@pytest.mark.parametrize(
    ("problem", "excluded"),
    [
        ("classification", {"resnet_model", "one_class_svm"}),
        ("regression", {"lora_model", "topological_extractor"}),
        ("classification_tabular", {"resnet_model", "one_class_svm"}),
        ("regression_tabular", {"lora_model", "topological_extractor"}),
    ],
)
def test_catalog_discovery_preserves_operation_exclusions(problem, excluded):
    assert excluded.isdisjoint(discovery.catalog_operation_names(problem))
