import importlib
from concurrent.futures import ThreadPoolExecutor

import pytest

try:
    from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
    from fedot.core.repository.operation_types_repository import OperationTypesRepository
    from fedot.extensions import clear_extension_registry, get_registered_extensions
except ModuleNotFoundError:
    pytest.skip("The active FEDOT profile has no extension contract.", allow_module_level=True)


def test_initializer_import_does_not_mutate_fedot():
    clear_extension_registry()
    repository = dict(OperationTypesRepository.__repository_dict__)
    search_method = PipelineSearchSpace.get_parameters_dict
    module = importlib.import_module("fedot_ind.core.repository.initializer_industrial_models")

    importlib.reload(module)

    assert get_registered_extensions() == ()
    assert dict(OperationTypesRepository.__repository_dict__) == repository
    assert PipelineSearchSpace.get_parameters_dict is search_method


def test_tensor_facade_restores_scoped_registration_without_monkeypatches():
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

    clear_extension_registry()
    repository = dict(OperationTypesRepository.__repository_dict__)
    search_method = PipelineSearchSpace.get_parameters_dict
    facade = IndustrialModels(profile="tensor")

    facade.setup_repository()
    assert len(get_registered_extensions()) == 1
    facade.setup_repository()
    assert len(get_registered_extensions()) == 1
    facade.setup_default_repository()

    assert get_registered_extensions() == ()
    assert dict(OperationTypesRepository.__repository_dict__) == repository
    assert PipelineSearchSpace.get_parameters_dict is search_method


def test_tensor_context_manager_restores_registry():
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

    clear_extension_registry()
    with IndustrialModels(profile="tensor"):
        assert len(get_registered_extensions()) == 1
    assert get_registered_extensions() == ()


def test_temporary_facade_keeps_legacy_global_activation_semantics():
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

    clear_extension_registry()
    IndustrialModels(profile="tensor").setup_repository()

    assert len(get_registered_extensions()) == 1
    IndustrialModels(profile="tensor").setup_default_repository()
    assert get_registered_extensions() == ()


def test_nested_context_does_not_close_outer_activation():
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

    clear_extension_registry()
    with IndustrialModels(profile="tensor"):
        with IndustrialModels(profile="tensor"):
            assert len(get_registered_extensions()) == 1
        assert len(get_registered_extensions()) == 1
    assert get_registered_extensions() == ()


def test_same_facade_supports_nested_contexts():
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

    clear_extension_registry()
    facade = IndustrialModels(profile="tensor")
    with facade:
        with facade:
            assert len(get_registered_extensions()) == 1
        assert len(get_registered_extensions()) == 1
    assert get_registered_extensions() == ()


def test_manual_activation_survives_nested_context_until_explicit_restore():
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

    clear_extension_registry()
    facade = IndustrialModels(profile="tensor")
    facade.setup_repository()
    with facade:
        assert len(get_registered_extensions()) == 1
    assert len(get_registered_extensions()) == 1
    facade.setup_default_repository()
    assert get_registered_extensions() == ()


def test_extension_activation_is_isolated_between_threads():
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

    def registry_size_inside_and_after_scope():
        clear_extension_registry()
        with IndustrialModels(profile="tensor"):
            inside = len(get_registered_extensions())
        return inside, len(get_registered_extensions())

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: registry_size_inside_and_after_scope(), range(2)))

    assert results == ((1, 0), (1, 0))


def test_facade_uses_shared_integration_profile_variable(monkeypatch):
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

    monkeypatch.setenv("FEDOT_INTEGRATION_PROFILE", "legacy")

    assert IndustrialModels().profile == "legacy"
