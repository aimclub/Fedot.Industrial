from importlib.util import find_spec

import numpy as np
import pytest

pytest.importorskip('fedot.core.data.data')

from fedot_ind.core.models.detection.modern_detectors import (
    ConvAutoencoderDetector,
    FeatureIsolationForestDetector,
    FeatureOneClassDetector,
    TCNAutoencoderDetector,
    build_detection_input_data,
)


def _series(length: int = 96) -> np.ndarray:
    time = np.arange(length, dtype=float)
    values = np.sin(time / 5.0) + 0.2 * np.cos(time / 13.0)
    values[64:72] += 5.0
    return values.reshape(-1, 1)


def test_feature_runtime_detector_publishes_scores_events_and_stage_diagnostics():
    input_data = build_detection_input_data(_series())
    detector = FeatureIsolationForestDetector(
        params={
            'window_length': 12,
            'calibration_strategy': 'quantile',
            'threshold_quantile': 0.85,
            'n_estimators': 50,
            'random_state': 42,
        }
    )

    detector.fit(input_data)
    probabilities = detector.predict_proba(input_data)
    diagnostics = detector.get_stage_diagnostics()
    risk_frame = detector.get_risk_feature_frame()

    assert probabilities.shape == (96, 2)
    assert diagnostics['calibration']['strategy'] == 'quantile'
    assert diagnostics['event_aggregation']['n_events'] >= 1
    assert risk_frame.metadata['n_events'] >= 1


def test_feature_oneclass_detector_exposes_stage_tuning_plan():
    detector = FeatureOneClassDetector(
        params={
            'window_length': 10,
            'nu': 0.1,
            'kernel': 'rbf',
        }
    )

    plan = detector.get_stage_tuning_plan()

    assert plan['canonical_model_name'] == 'feature_oneclass_detector'
    assert plan['family'] == 'feature_baseline'
    assert plan['groups'][0]['stage'] == 'representation'


@pytest.mark.skipif(find_spec('torch') is None, reason='torch is required for neural detector smoke tests')
def test_conv_autoencoder_detector_smoke():
    input_data = build_detection_input_data(_series(64))
    detector = ConvAutoencoderDetector(
        params={
            'window_length': 12,
            'epochs': 1,
            'batch_size': 8,
            'learning_rate': 1e-3,
            'device': 'cpu',
        }
    )

    detector.fit(input_data)
    probabilities = detector.predict_proba(input_data)

    assert probabilities.shape == (64, 2)


@pytest.mark.skipif(find_spec('torch') is None, reason='torch is required for neural detector smoke tests')
def test_tcn_autoencoder_detector_smoke():
    input_data = build_detection_input_data(_series(64))
    detector = TCNAutoencoderDetector(
        params={
            'window_length': 12,
            'epochs': 1,
            'batch_size': 8,
            'learning_rate': 1e-3,
            'device': 'cpu',
            'kernel_size': 3,
            'num_filters': 16,
            'num_levels': 2,
        }
    )

    detector.fit(input_data)
    probabilities = detector.predict_proba(input_data)

    assert probabilities.shape == (64, 2)
