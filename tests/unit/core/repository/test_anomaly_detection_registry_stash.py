import pytest

pytest.importorskip('fedot.core.operations.operation_parameters')

from fedot_ind.core.repository.IndustrialOperationParameters import get_default_params
from fedot_ind.core.repository.detection_registry import (
    CANONICAL_STAGE_DETECTION_MODELS,
    canonical_detection_model_name,
    detection_aliases_for
)
from fedot_ind.core.tuning.search_space import industrial_search_space


def test_canonical_anomaly_detection_model_name_normalizes_short_aliases():
    assert canonical_detection_model_name('iforest_detector') == 'feature_iforest_detector'
    assert canonical_detection_model_name('feature_iforest') == 'feature_iforest_detector'
    assert canonical_detection_model_name('stat_detector') == 'feature_oneclass_detector'
    assert canonical_detection_model_name('feature_oneclass') == 'feature_oneclass_detector'
    assert canonical_detection_model_name('conv_ae_detector') == 'conv_autoencoder_detector'
    assert canonical_detection_model_name('conv_autoencoder') == 'conv_autoencoder_detector'
    assert canonical_detection_model_name('tcn_ae_detector') == 'tcn_autoencoder_detector'
    assert canonical_detection_model_name('tcn_autoencoder') == 'tcn_autoencoder_detector'
    assert canonical_detection_model_name('lstm_ae_detector') == 'legacy_lstm_autoencoder_detector'
    assert canonical_detection_model_name('arima_detector') == 'legacy_arima_detector'
    assert canonical_detection_model_name('sst') == 'legacy_sst_detector'
    assert canonical_detection_model_name('unscented_kalman_filter') == 'legacy_kalman_detector'
    assert canonical_detection_model_name('functional_pca') == 'legacy_functional_pca_detector'
    assert canonical_detection_model_name('feature_iforest_detector') == 'feature_iforest_detector'


def test_search_space_contains_alias_entries_for_short_anomaly_detection_names():
    assert industrial_search_space['iforest_detector'] == industrial_search_space['feature_iforest_detector']
    assert industrial_search_space['feature_iforest'] == industrial_search_space['feature_iforest_detector']
    assert industrial_search_space['stat_detector'] == industrial_search_space['feature_oneclass_detector']
    assert industrial_search_space['feature_oneclass'] == industrial_search_space['feature_oneclass_detector']
    assert industrial_search_space['conv_ae_detector'] == industrial_search_space['conv_autoencoder_detector']
    assert industrial_search_space['conv_autoencoder'] == industrial_search_space['conv_autoencoder_detector']
    assert industrial_search_space['tcn_ae_detector'] == industrial_search_space['tcn_autoencoder_detector']
    assert industrial_search_space['tcn_autoencoder'] == industrial_search_space['tcn_autoencoder_detector']
    assert industrial_search_space['lstm_ae_detector'] == industrial_search_space['legacy_lstm_autoencoder_detector']
    assert industrial_search_space['arima_detector'] == industrial_search_space['legacy_arima_detector']
    assert industrial_search_space['sst'] == industrial_search_space['legacy_sst_detector']
    assert industrial_search_space['unscented_kalman_filter'] == industrial_search_space['legacy_kalman_detector']
    assert industrial_search_space['functional_pca'] == industrial_search_space['legacy_functional_pca_detector']


def test_default_params_lookup_uses_canonical_anomaly_detection_names():

    alias_params = get_default_params('iforest_detector')
    canonical_params = get_default_params('feature_iforest_detector')
    
    assert alias_params is not None
    assert alias_params == canonical_params
    assert get_default_params('iforest_detector') == get_default_params('feature_iforest_detector')
    assert get_default_params('feature_iforest') == get_default_params('feature_iforest_detector')
    assert get_default_params('stat_detector') == get_default_params('feature_oneclass_detector')
    assert get_default_params('feature_oneclass') == get_default_params('feature_oneclass_detector')
    assert get_default_params('conv_ae_detector') == get_default_params('conv_autoencoder_detector')
    assert get_default_params('conv_autoencoder') == get_default_params('conv_autoencoder_detector')
    assert get_default_params('tcn_ae_detector') == get_default_params('tcn_autoencoder_detector')
    assert get_default_params('tcn_autoencoder') == get_default_params('tcn_autoencoder_detector')
    assert get_default_params('lstm_ae_detector') == get_default_params('legacy_lstm_autoencoder_detector')
    assert get_default_params('arima_detector') == get_default_params('legacy_arima_detector')
    assert get_default_params('sst') == get_default_params('legacy_sst_detector')
    assert get_default_params('unscented_kalman_filter') == get_default_params('legacy_kalman_detector')
    assert get_default_params('functional_pca') == get_default_params('legacy_functional_pca_detector')


def test_stage_anomaly_detection_models_publish_alias_sets():
    assert 'feature_iforest_detector' in CANONICAL_STAGE_DETECTION_MODELS
    assert detection_aliases_for('feature_iforest_detector') == ('feature_iforest', 'feature_iforest_detector', 'iforest_detector')