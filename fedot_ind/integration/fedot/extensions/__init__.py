"""Lazy public index for the FEDOT extension integration."""

from importlib import import_module


_EXPORTS = {
    "FEDOT_EXTENSION_MANIFEST": "manifest",
    "IndustrialExtensionContractError": "contracts",
    "IndustrialExtensionErrorCode": "contracts",
    "IndustrialExtensionResult": "contracts",
    "IndustrialExtensionStatus": "contracts",
    "IndustrialOperationDeclaration": "contracts",
    "IndustrialOperationKind": "contracts",
    "build_industrial_extension_manifest": "manifest",
    "build_industrial_extension_plan": "catalog",
    "catalog_operation_names": "discovery",
    "default_industrial_available_operations": "discovery",
    "default_industrial_availiable_operation": "discovery",
    "industrial_extension_scope": "bootstrap",
    "load_industrial_extension_catalog": "catalog",
    "register_industrial_extension": "bootstrap",
    "resolve_industrial_operation": "bootstrap",
    "supports_fedot_extension_contract": "bootstrap",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
