import numpy as np

from fedot_ind.core.models.detection.runtime import (
    AnomalyScoreSeries,
    DetectionSplitKind,
    DetectionSplitSpec,
    build_detection_window_batch,
    build_risk_feature_frame,
    detect_events_from_score_series,
    estimate_detection_threshold,
    infer_regime_segments,
    split_detection_batch,
)


def _multichannel_series(length: int = 48) -> np.ndarray:
    time = np.arange(length, dtype=float)
    return np.column_stack(
        (
            np.sin(time / 4.0),
            np.cos(time / 5.0) + 0.1 * time,
        )
    )


def test_build_detection_window_batch_preserves_window_count_and_shape():
    batch = build_detection_window_batch(
        _multichannel_series(48),
        window_size=8,
        stride=2,
    )

    assert batch.windows.shape == (21, 8, 2)
    assert batch.window_indices.shape == (21, 2)
    assert batch.n_windows == 21
    assert batch.n_channels == 2
    assert batch.flattened_features.shape == (21, 16)
    assert batch.statistical_features.shape == (21, 10)


def test_split_detection_batch_temporal_prevents_cross_split_window_overlap():
    batch = build_detection_window_batch(
        _multichannel_series(64),
        window_size=10,
        stride=2,
    )
    train_batch, calibration_batch, test_batch = split_detection_batch(
        batch,
        DetectionSplitSpec(
            kind=DetectionSplitKind.TEMPORAL,
            train_fraction=0.5,
            calibration_fraction=0.25,
            prevent_future_leakage=True,
        ),
    )

    assert train_batch is not None
    assert calibration_batch is not None
    assert train_batch.window_indices[-1, 1] <= calibration_batch.window_indices[0, 0]
    if test_batch is not None and len(test_batch.window_indices):
        assert calibration_batch.window_indices[-1, 1] <= test_batch.window_indices[0, 0]


def test_split_detection_batch_random_holdout_is_deterministic_for_same_seed():
    batch = build_detection_window_batch(
        _multichannel_series(40),
        window_size=8,
        stride=1,
    )
    split_spec = DetectionSplitSpec(
        kind=DetectionSplitKind.HOLDOUT,
        train_fraction=0.6,
        calibration_fraction=0.2,
        random_seed=7,
        prevent_future_leakage=False,
    )

    first = split_detection_batch(batch, split_spec)
    second = split_detection_batch(batch, split_spec)

    assert first[0].metadata['selected_window_ids'] == second[0].metadata['selected_window_ids']
    assert first[1].metadata['selected_window_ids'] == second[1].metadata['selected_window_ids']
    if first[2] is not None and second[2] is not None:
        assert first[2].metadata['selected_window_ids'] == second[2].metadata['selected_window_ids']


def test_split_detection_batch_domain_holdout_uses_target_domain_windows_only_for_holdout():
    batch = build_detection_window_batch(
        _multichannel_series(24),
        window_size=6,
        stride=1,
        metadata={
            'window_domains': ['source'] * 10 + ['target'] * 9,
        },
    )

    train_batch, calibration_batch, test_batch = split_detection_batch(
        batch,
        DetectionSplitSpec(
            kind=DetectionSplitKind.DOMAIN_HOLDOUT,
            calibration_fraction=0.4,
            target_domain='target',
        ),
    )

    assert set(train_batch.metadata['window_domains']) == {'source'}
    assert set(calibration_batch.metadata['window_domains']) == {'target'}
    assert train_batch.metadata['split_kind'] == DetectionSplitKind.DOMAIN_HOLDOUT.value
    assert calibration_batch.metadata['split_name'] == 'calibration'
    if test_batch is not None:
        assert set(test_batch.metadata['window_domains']) == {'target'}
        assert test_batch.metadata['split_name'] == 'test'


def test_estimate_detection_threshold_quantile_is_monotonic():
    scores = np.linspace(0.0, 10.0, num=100)

    lower = estimate_detection_threshold(scores, strategy='quantile', quantile=0.9)
    higher = estimate_detection_threshold(scores, strategy='quantile', quantile=0.99)

    assert higher >= lower


def test_detect_events_and_risk_frame_preserve_regime_context():
    scores = np.array([0.1, 0.2, 1.5, 1.7, 0.4, 0.2, 2.2, 2.3, 2.0, 0.1], dtype=float)
    score_series = AnomalyScoreSeries(
        scores=tuple(scores.tolist()),
        labels=tuple(int(value >= 1.0) for value in scores.tolist()),
        threshold=1.0,
        calibration_strategy='mad',
    )
    regimes = infer_regime_segments(_multichannel_series(10))

    events = detect_events_from_score_series(
        score_series,
        min_event_length=2,
        regime_segments=regimes,
    )
    risk_frame = build_risk_feature_frame(
        events=events,
        regime_segments=regimes,
        score_series=score_series,
        node_name='mill_1',
        domain_name='mpsi',
    )

    assert len(events) == 2
    assert risk_frame.metadata['n_events'] == 2
    frame = risk_frame.to_frame()
    assert {'event_peak_score', 'regime_label', 'node_name', 'domain_name'} <= set(frame.columns)
    assert set(frame['node_name']) == {'mill_1'}
