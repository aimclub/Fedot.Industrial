import importlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.extensions import clear_extension_registry, get_registered_extensions

from fedot_ind.integration.fedot.extensions import (
    IndustrialExtensionContractError,
    IndustrialExtensionErrorCode,
    IndustrialExtensionSession,
    IndustrialExtensionSessionState,
    industrial_extension_scope,
)


def test_extension_import_does_not_mutate_fedot():
    clear_extension_registry()
    repository = dict(OperationTypesRepository.__repository_dict__)
    search_method = PipelineSearchSpace.get_parameters_dict
    module = importlib.import_module("fedot_ind.integration.fedot.extensions.bootstrap")

    importlib.reload(module)

    assert get_registered_extensions() == ()
    assert dict(OperationTypesRepository.__repository_dict__) == repository
    assert PipelineSearchSpace.get_parameters_dict is search_method


def test_session_owns_registration_and_restores_fedot_state():
    clear_extension_registry()
    repository = dict(OperationTypesRepository.__repository_dict__)
    search_method = PipelineSearchSpace.get_parameters_dict
    session = IndustrialExtensionSession()

    first = session.activate()
    second = session.activate()

    assert first is second
    assert session.state is IndustrialExtensionSessionState.ACTIVE
    assert len(get_registered_extensions()) == 1
    session.close()
    assert session.state is IndustrialExtensionSessionState.CLOSED
    assert get_registered_extensions() == ()
    assert dict(OperationTypesRepository.__repository_dict__) == repository
    assert PipelineSearchSpace.get_parameters_dict is search_method


def test_closed_session_cannot_be_reactivated():
    session = IndustrialExtensionSession()
    session.close()

    with pytest.raises(IndustrialExtensionContractError) as error:
        session.activate()

    assert error.value.code is IndustrialExtensionErrorCode.INVALID_SESSION_STATE


def test_nested_scope_does_not_close_session_registration():
    clear_extension_registry()
    session = IndustrialExtensionSession()
    session.activate()

    with industrial_extension_scope():
        assert len(get_registered_extensions()) == 1

    assert len(get_registered_extensions()) == 1
    session.close()
    assert get_registered_extensions() == ()


def test_registration_remains_until_last_session_closes():
    clear_extension_registry()
    first = IndustrialExtensionSession()
    second = IndustrialExtensionSession()

    first.activate()
    second.activate()
    first.close()

    assert len(get_registered_extensions()) == 1
    assert second.state is IndustrialExtensionSessionState.ACTIVE
    second.close()
    assert get_registered_extensions() == ()


def test_extension_scope_is_isolated_between_threads():
    def registry_size_inside_and_after_scope():
        clear_extension_registry()
        with industrial_extension_scope():
            inside = len(get_registered_extensions())
        return inside, len(get_registered_extensions())

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: registry_size_inside_and_after_scope(), range(2)))

    assert results == ((1, 0), (1, 0))
