from fedot_ind.core.tuning.search_space import industrial_search_space
from fedot_ind.core.repository.forecasting_registry import (
    CANONICAL_STAGE_FORECASTING_MODELS,
    canonical_forecasting_model_name,
    forecasting_aliases_for,
)
from fedot_ind.core.repository.IndustrialOperationParameters import get_default_params
import pytest

pytest.importorskip('fedot.core.operations.operation_parameters')


def test_canonical_forecasting_model_name_normalizes_short_aliases():
    assert canonical_forecasting_model_name('mssa') == 'mssa_forecaster'
    assert canonical_forecasting_model_name('havok') == 'havok_forecaster'
    assert canonical_forecasting_model_name('lagged_ridge_forecaster') == 'lagged_ridge_forecaster'


def test_default_params_lookup_uses_canonical_forecasting_names():
    assert get_default_params('mssa') == get_default_params('mssa_forecaster')
    assert get_default_params('havok') == get_default_params('havok_forecaster')


def test_search_space_contains_alias_entries_for_short_forecasting_names():
    assert industrial_search_space['mssa'] == industrial_search_space['mssa_forecaster']
    assert industrial_search_space['havok'] == industrial_search_space['havok_forecaster']


def test_stage_forecasting_models_publish_alias_sets():
    assert 'mssa_forecaster' in CANONICAL_STAGE_FORECASTING_MODELS
    assert forecasting_aliases_for('mssa_forecaster') == ('mssa_forecaster', 'mssa')
