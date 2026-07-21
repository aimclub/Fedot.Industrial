from fedot_ind.core.repository.detection_registry import (
    CANONICAL_STAGE_DETECTION_MODELS,
    LEGACY_DETECTION_MODELS,
    canonical_detection_model_name,
    detection_aliases_for,
    detection_family_for,
)
from fedot_ind.core.repository.IndustrialOperationParameters import get_default_params
import pytest

pytest.importorskip('fedot.core.operations.operation_parameters')


def test_canonical_detection_model_name_normalizes_aliases_and_whitespace():
    assert canonical_detection_model_name(' iforest_detector ') == 'feature_iforest_detector'
    assert canonical_detection_model_name('STAT_DETECTOR') == 'feature_oneclass_detector'
    assert canonical_detection_model_name('legacy_lstm_autoencoder_detector') == 'legacy_lstm_autoencoder_detector'


def test_detection_aliases_for_include_canonical_name_and_known_aliases():
    aliases = detection_aliases_for('conv_autoencoder_detector')

    assert 'conv_autoencoder_detector' in aliases
    assert 'conv_ae_detector' in aliases
    assert 'conv_autoencoder' in aliases


def test_detection_family_for_groups_first_class_and_legacy_models():
    assert detection_family_for('feature_iforest_detector') == 'feature_baseline'
    assert detection_family_for('conv_autoencoder_detector') == 'neural_reconstruction'
    assert detection_family_for('legacy_arima_detector') == 'legacy_detection'


def test_detection_registry_partitions_first_class_and_legacy_sets():
    assert set(CANONICAL_STAGE_DETECTION_MODELS).isdisjoint(set(LEGACY_DETECTION_MODELS))
    assert 'feature_iforest_detector' in CANONICAL_STAGE_DETECTION_MODELS
    assert 'legacy_lstm_autoencoder_detector' in LEGACY_DETECTION_MODELS


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


# def test_search_space_contains_alias_entries_for_short_anomaly_detection_names():
#     assert industrial_search_space['iforest_detector'] == industrial_search_space['feature_iforest_detector']
#     assert industrial_search_space['feature_iforest'] == industrial_search_space['feature_iforest_detector']
#     assert industrial_search_space['stat_detector'] == industrial_search_space['feature_oneclass_detector']
#     assert industrial_search_space['feature_oneclass'] == industrial_search_space['feature_oneclass_detector']
#     assert industrial_search_space['conv_ae_detector'] == industrial_search_space['conv_autoencoder_detector']
#     assert industrial_search_space['conv_autoencoder'] == industrial_search_space['conv_autoencoder_detector']
#     assert industrial_search_space['tcn_ae_detector'] == industrial_search_space['tcn_autoencoder_detector']
#     assert industrial_search_space['tcn_autoencoder'] == industrial_search_space['tcn_autoencoder_detector']
#     assert industrial_search_space['lstm_ae_detector'] == industrial_search_space['legacy_lstm_autoencoder_detector']
#     assert industrial_search_space['arima_detector'] == industrial_search_space['legacy_arima_detector']
#     assert industrial_search_space['sst'] == industrial_search_space['legacy_sst_detector']
#     assert industrial_search_space['unscented_kalman_filter'] == industrial_search_space['legacy_kalman_detector']
#     assert industrial_search_space['functional_pca'] == industrial_search_space['legacy_functional_pca_detector']


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
    assert detection_aliases_for('feature_iforest_detector') == (
        'feature_iforest', 'feature_iforest_detector', 'iforest_detector')
