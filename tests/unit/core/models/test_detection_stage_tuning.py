from fedot_ind.core.models.detection.stage_tuning import (
    DetectionStageName,
    build_detection_stage_tuning_plan,
)


def test_build_detection_stage_tuning_plan_for_iforest_orders_runtime_stages():
    plan = build_detection_stage_tuning_plan(
        'feature_iforest_detector',
        {'window_size': 16, 'stride': 2, 'threshold_quantile': 0.99},
    )

    stages = tuple(group.stage for group in plan.groups)

    assert plan.canonical_model_name == 'feature_iforest_detector'
    assert plan.family == 'feature_baseline'
    assert stages == (
        DetectionStageName.REPRESENTATION.value,
        DetectionStageName.ANOMALY_SCORING.value,
        DetectionStageName.CALIBRATION.value,
        DetectionStageName.EVENT_AGGREGATION.value,
    )
    assert plan.groups[-1].depends_on == (
        DetectionStageName.REPRESENTATION.value,
        DetectionStageName.CALIBRATION.value,
    )


def test_build_detection_stage_tuning_plan_canonicalizes_legacy_aliases():
    plan = build_detection_stage_tuning_plan('iforest_detector')

    assert plan.model_name == 'iforest_detector'
    assert plan.canonical_model_name == 'feature_iforest_detector'
    assert plan.metadata['supports_stage_tuning'] is True


def test_build_detection_stage_tuning_plan_marks_legacy_models_as_non_stage_native():
    plan = build_detection_stage_tuning_plan('legacy_lstm_autoencoder_detector')

    assert plan.family == 'legacy_detection'
    assert plan.groups == ()
    assert plan.metadata['supports_stage_tuning'] is False


def test_stage_tuning_plan_structure_K():
    model_name = 'conv_autoencoder_detector'
    plan = build_detection_stage_tuning_plan(model_name)
    stages_in_plan = {group.stage for group in plan.groups}
    scoring_group = next(g for g in plan.groups if g.stage == DetectionStageName.ANOMALY_SCORING.value)

    assert plan.model_name == model_name
    assert plan.metadata['supports_stage_tuning'] is True
    assert DetectionStageName.REPRESENTATION.value in stages_in_plan
    assert DetectionStageName.REPRESENTATION.value in scoring_group.depends_on


def test_stage_vocabulary_is_complete_K():
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
