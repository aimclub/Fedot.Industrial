import numpy as np
import pytest

pytest.importorskip('torch')

from dataclasses import replace

from fedot_ind.core.models.detection.stage_tuning import (
    DetectionStageName,
    build_detection_stage_tuning_plan
)

from fedot_ind.core.models.detection.runtime import (
    DetectionSplitSpec,
    DetectionSplitKind,
    ensure_detection_array,
    build_detection_window_batch,
    build_window_statistical_features,
    split_detection_batch,
    align_window_scores_to_points,
    build_anomaly_score_series,
    detect_events_from_score_series,
    infer_regime_segments,
    build_transfer_alignment_report,
    build_risk_feature_frame
)


def test_detection_array_contract():
    series = np.arange(100)

    arr = ensure_detection_array(series)

    assert arr.ndim == 2
    assert arr.shape == (100, 1)


def test_detection_window_batch_contract():
    series = np.linspace(0.0, 1.0, num=120)

    batch = build_detection_window_batch(
        series, 
        window_size=20, 
        stride=5
    )

    start, end = batch.window_indices[0]

    assert batch.windows.ndim == 3  # [N, W, C]
    assert batch.window_indices.shape[1] == 2
    assert batch.n_windows > 0
    assert batch.n_channels == 1
    assert end - start == batch.window_size


def test_statistical_features_shape():
    series = np.linspace(0, 1, 100)

    batch = build_detection_window_batch(
        series, 
        window_size=10)

    features = build_window_statistical_features(batch.windows)

    assert features.shape[0] == batch.n_windows
    assert features.shape[1] == 5 * batch.n_channels


def test_split_no_future_leakage():
    series = np.linspace(0, 1, 150)

    batch = build_detection_window_batch(
        series, 
        window_size=15, 
        stride=3
    )
    split_spec = DetectionSplitSpec(
        kind=DetectionSplitKind.TEMPORAL,
        train_fraction=0.6,
        calibration_fraction=0.2,
        prevent_future_leakage=True,
    )

    train, calib, test = split_detection_batch(batch, split_spec)
    train_end = train.window_indices[-1][1]
    calib_start = calib.window_indices[0][0]

    assert train_end <= calib_start

    if test is not None and len(test.window_indices) > 0:
        test_start = test.window_indices[0][0]
        assert calib.window_indices[-1][1] <= test_start


def test_split_detection_batch_contract_invariants():
    series = np.linspace(0, 1, 120)

    batch = build_detection_window_batch(
        series,
        window_size=10,
        stride=1
    )

    split_spec = DetectionSplitSpec(
        kind=DetectionSplitKind.TEMPORAL,
        train_fraction=0.6,
        calibration_fraction=0.2,
        prevent_future_leakage=True,
    )

    train, calib, test = split_detection_batch(batch, split_spec)
    train_ids = set(map(tuple, train.window_indices))
    calib_ids = set(map(tuple, calib.window_indices))
    test_ids = set(map(tuple, test.window_indices)) if test else set()
    total = len(train_ids | calib_ids | test_ids)

    assert train_ids.isdisjoint(calib_ids)
    assert train_ids.isdisjoint(test_ids)
    assert calib_ids.isdisjoint(test_ids)
    assert (
        max(train.window_indices[:, 1])
        <= min(calib.window_indices[:, 0])
    )   
    assert total <= batch.n_windows


def test_window_to_point_aggregation_contract():
    series = np.linspace(0, 1, 50)

    batch = build_detection_window_batch(
        series, 
        window_size=5, 
        stride=1
    )

    window_scores = np.arange(batch.n_windows)

    point_scores = align_window_scores_to_points(window_scores, batch)
    assert len(point_scores) == batch.original_length
    assert np.isfinite(point_scores).all()
    assert np.std(point_scores) > 0


def test_event_detection_contract_minimal():
    scores = np.array([0.0, 0.6, 0.9, 0.0])
    series = build_anomaly_score_series(
        scores,
        threshold=0.5,
        calibration_strategy='fixed',
    )

    events = detect_events_from_score_series(series, min_event_length=2)
    event = events[0]

    assert len(events) == 1
    assert event.start_index == 1
    assert event.end_index == 2
    assert event.peak_index == 2
    assert np.isclose(event.peak_score, max(scores[1:3]))


def test_regime_segmentation_covers_series():
    series = np.linspace(0, 10, 120)

    segments = infer_regime_segments(series)
    covered = []
    for seg in segments:
        covered.extend(range(seg.start_index, seg.end_index + 1))

    covered_sorted = sorted(covered)
    assert covered_sorted == list(range(len(series)))


def test_risk_feature_frame_contract():
    scores = np.zeros(100)
    scores[40:50] = 2.0

    series = build_anomaly_score_series(
        scores,
        threshold=1.0,
        calibration_strategy='fixed',
    )

    events = detect_events_from_score_series(series)
    regimes = infer_regime_segments(scores)

    frame = build_risk_feature_frame(
        events=events,
        regime_segments=regimes,
        score_series=series,
        node_name='node_1',
        domain_name='test_domain',
    )

    df = frame.to_frame()

    assert len(df) == len(events)
    assert 'event_start_index' in df.columns
    assert 'regime_label' in df.columns


def test_domain_holdout_split_contract():
    series = np.linspace(0, 1, 200)

    batch = build_detection_window_batch(
        series, 
        window_size=10, 
        stride=2
    )
    
    n_windows = batch.n_windows
    split_point = 40
    domains = np.array(["A"] * split_point + ["B"] * (n_windows - split_point))

    batch = replace(batch, metadata={"window_domains": domains})

    split_spec = DetectionSplitSpec(
        kind=DetectionSplitKind.DOMAIN_HOLDOUT,
        target_domain="B",
        calibration_fraction=0.3,
        prevent_future_leakage=True,
    )

    train, calib, test = split_detection_batch(batch, split_spec)

    assert all(d == "A" for d in train.metadata["window_domains"])
    assert all(d == "B" for d in calib.metadata["window_domains"])

    if test is not None:
        assert all(d == "B" for d in test.metadata["window_domains"])
        assert test.n_windows > 0

    if test is not None and test.n_windows > 0:
        calib_end = calib.window_indices[-1][1]
        test_start = test.window_indices[0][0]
        assert calib_end <= test_start


def test_transfer_alignment_report_contract():
    source = np.random.rand(100, 3)
    target = np.random.rand(100, 3)

    report = build_transfer_alignment_report(source, target)

    assert report.n_source == 100
    assert report.n_target == 100
    assert len(report.source_channel_mean) == 3
    assert len(report.target_channel_mean) == 3
    assert len(report.mean_shift) == 3
    assert np.allclose(
        np.array(report.mean_shift),
        np.array(report.target_channel_mean) - np.array(report.source_channel_mean)
    )


def test_stage_tuning_plan_structure():
    model_name = 'conv_autoencoder_detector'
    plan = build_detection_stage_tuning_plan(model_name)
    stages_in_plan = {group.stage for group in plan.groups}
    scoring_group = next(g for g in plan.groups if g.stage == DetectionStageName.ANOMALY_SCORING.value)
    
    assert plan.model_name == model_name
    assert plan.metadata['supports_stage_tuning'] is True
    assert DetectionStageName.REPRESENTATION.value in stages_in_plan
    assert DetectionStageName.REPRESENTATION.value in scoring_group.depends_on


def test_event_detection_deterministic():
    scores = np.array([0,1,1,0,1,1,0])

    series = build_anomaly_score_series(scores, threshold=0.5, calibration_strategy='fixed')

    e1 = detect_events_from_score_series(series)
    e2 = detect_events_from_score_series(series)

    assert e1 == e2


def test_stage_vocabulary_is_complete():
    expected_stages = {
        'data_quality',
        'regime_segmentation',
        'representation',
        'anomaly_scoring',
        'calibration',
        'event_aggregation',
        'transfer_alignment',
        'interpretation'
    }
    
    actual_stages = {stage.value for stage in DetectionStageName}
    
    assert actual_stages == expected_stages