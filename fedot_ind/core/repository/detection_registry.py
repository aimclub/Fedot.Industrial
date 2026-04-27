from __future__ import annotations

DETECTION_MODEL_ALIASES: dict[str, str] = {
    'iforest_detector': 'feature_iforest_detector',
    'feature_iforest': 'feature_iforest_detector',
    'stat_detector': 'feature_oneclass_detector',
    'feature_oneclass': 'feature_oneclass_detector',
    'conv_ae_detector': 'conv_autoencoder_detector',
    'conv_autoencoder': 'conv_autoencoder_detector',
    'tcn_ae_detector': 'tcn_autoencoder_detector',
    'tcn_autoencoder': 'tcn_autoencoder_detector',
    'lstm_ae_detector': 'legacy_lstm_autoencoder_detector',
    'arima_detector': 'legacy_arima_detector',
    'sst': 'legacy_sst_detector',
    'unscented_kalman_filter': 'legacy_kalman_detector',
    'functional_pca': 'legacy_functional_pca_detector',
}

CANONICAL_STAGE_DETECTION_MODELS: tuple[str, ...] = (
    'feature_iforest_detector',
    'feature_oneclass_detector',
    'conv_autoencoder_detector',
    'tcn_autoencoder_detector',
)

LEGACY_DETECTION_MODELS: tuple[str, ...] = (
    'legacy_lstm_autoencoder_detector',
    'legacy_arima_detector',
    'legacy_sst_detector',
    'legacy_kalman_detector',
    'legacy_functional_pca_detector',
)


def canonical_detection_model_name(name: str | None) -> str:
    normalized = str(name or '').strip().lower()
    return DETECTION_MODEL_ALIASES.get(normalized, normalized)


def detection_aliases_for(model_name: str) -> tuple[str, ...]:
    canonical = canonical_detection_model_name(model_name)
    aliases = [alias for alias, target in DETECTION_MODEL_ALIASES.items() if target == canonical]
    return tuple(sorted(dict.fromkeys([canonical, *aliases])))


def detection_family_for(model_name: str | None) -> str:
    canonical = canonical_detection_model_name(model_name)
    if canonical in {'feature_iforest_detector', 'feature_oneclass_detector'}:
        return 'feature_baseline'
    if canonical in {'conv_autoencoder_detector', 'tcn_autoencoder_detector'}:
        return 'neural_reconstruction'
    if canonical in LEGACY_DETECTION_MODELS:
        return 'legacy_detection'
    return 'detection'
