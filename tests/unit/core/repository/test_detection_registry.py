from fedot_ind.core.repository.detection_registry import (
    CANONICAL_STAGE_DETECTION_MODELS,
    LEGACY_DETECTION_MODELS,
    canonical_detection_model_name,
    detection_aliases_for,
    detection_family_for,
)


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
