"""Temporal operations must match the installed FEDOT, not an assumed patch."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fedot_ind.integration.fedot.extensions import factories, temporal
from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
from fedot_ind.integration.fedot.extensions.contracts import (
    IndustrialExtensionContractError, IndustrialExtensionErrorCode,
)
from fedot_ind.integration.fedot.extensions.discovery import catalog_operation_names
from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest


def test_absent_temporal_contract_disables_discovery_manifest_and_factory(monkeypatch):
    declaration = load_industrial_extension_catalog().operation('hankelisation')
    monkeypatch.setattr(temporal.runtime_contracts, 'TemporalTransformInput', None, raising=False)
    monkeypatch.setattr(temporal.runtime_contracts, 'TemporalTransformOutput', None, raising=False)
    monkeypatch.setattr(factories, '_instantiate_target', lambda *args: pytest.fail('No model construction'))
    build_industrial_extension_manifest.cache_clear()
    try:
        assert 'hankelisation' not in catalog_operation_names('ts_forecasting')
        assert 'hankelisation' not in {spec.name for spec in build_industrial_extension_manifest().transforms}
        with pytest.raises(IndustrialExtensionContractError) as error:
            factories.make_deferred_factory(declaration)({'forecast_length': 8})
        assert error.value.code is IndustrialExtensionErrorCode.FEDOT_CONTRACT_UNAVAILABLE
    finally:
        build_industrial_extension_manifest.cache_clear()


def test_temporal_array_boundary_never_constructs_input_data(monkeypatch):
    if not temporal.supports_temporal_contract():
        pytest.skip('The installed FEDOT does not provide temporal transforms')
    monkeypatch.setattr(factories, '_build_input_data', lambda *args: pytest.fail('No legacy InputData bridge'))
    declaration = load_industrial_extension_catalog().operation('hankelisation')
    model = factories.make_deferred_factory(declaration)({'forecast_length': 8, 'window_size': 10})
    history = np.arange(80, dtype=float)
    snapshot = history.copy()
    model.fit(history)
    input_type, _ = temporal.require_temporal_contract()
    windows = model.transform_temporal(input_type(history, 8, True, np.arange(80)))
    np.testing.assert_array_equal(windows.features[0], history[:10])
    np.testing.assert_array_equal(windows.target[0], history[10:18])
    latest = model.transform_temporal(input_type(history, 8, False, np.arange(80)))
    np.testing.assert_array_equal(latest.features, history[-10:][None, :])
    assert latest.target is None
    np.testing.assert_array_equal(history, snapshot)


def test_installed_fedot_without_manual_checkout_has_consistent_discovery():
    root = Path(__file__).resolve().parents[5]
    script = f'''
import sys
sys.path.insert(0, {str(root)!r})
from fedot_ind.integration.fedot.extensions.temporal import supports_temporal_contract
from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest
from fedot_ind.integration.fedot.extensions.discovery import catalog_operation_names
available = supports_temporal_contract()
assert ('hankelisation' in catalog_operation_names('ts_forecasting')) == available
assert ('hankelisation' in {{spec.name for spec in build_industrial_extension_manifest().transforms}}) == available
'''
    result = subprocess.run([sys.executable, '-I', '-c', script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
