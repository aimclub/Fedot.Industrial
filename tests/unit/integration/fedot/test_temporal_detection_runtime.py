from fedot_ind.integration.fedot.compatibility import DataTypesEnum, OutputData
from fedot_ind.core.models.detection.runtime import AnomalyScoreSeries
from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
from fedot_ind.integration.fedot import (
    DetectionMode,
    IntegrationContractError,
    IntegrationErrorCode,
    RuntimeState,
    create_detection_runtime,
    prepare_detection_data,
)
import numpy as np
import pandas as pd
import pytest
import fedot

pytestmark = pytest.mark.skipif(
    not hasattr(fedot, "create_data"),
    reason="The FEDOT TensorData profile is not installed.",
)


def _signal(length: int = 80) -> np.ndarray:
    time = np.arange(length, dtype=float)
    values = np.sin(time / 4.0) + 0.05 * np.cos(time / 7.0)
    values[52:58] += 4.0
    return values


def test_modern_detection_models_are_declared_in_extension_catalog():
    catalog = load_industrial_extension_catalog()

    for name in (
            "feature_iforest_detector",
            "feature_oneclass_detector",
            "conv_autoencoder_detector",
            "tcn_autoencoder_detector",
    ):
        declaration = catalog.operation(name)
        assert declaration is not None
        assert "anomaly_detection" in declaration.problems
        assert declaration.requires_target is False


def test_detection_runtime_rejects_unowned_operation_without_fallback():
    with pytest.raises(IntegrationContractError) as error:
        create_detection_runtime(
            profile="tensor",
            operation_name="iforest_detector",
        )

    assert error.value.code is IntegrationErrorCode.UNSUPPORTED_OPERATION


@pytest.mark.parametrize("mode, expected_shape", [
    (DetectionMode.LABELS, (80,)),
    (DetectionMode.SCORES, (80,)),
    (DetectionMode.PROBABILITIES, (80, 2)),
])
def test_detection_runtime_preserves_aligned_datetime_context(mode, expected_shape):
    index = pd.date_range("2026-06-01", periods=80, freq="min")
    frame = pd.DataFrame({"sensor": _signal()}, index=index)
    runtime = create_detection_runtime(
        profile="tensor",
        operation_name="feature_iforest_detector",
        mode=mode,
        causal=True,
        parameters={
            "window_length": 12,
            "threshold_quantile": 0.95,
            "random_state": 42,
            "n_estimators": 20,
        },
    )

    result = runtime.fit(frame).predict(frame)

    assert result.values.shape == expected_shape
    np.testing.assert_array_equal(result.time_idx, index.to_numpy())
    assert result.metadata["input_time_index_preserved"] is True
    assert result.metadata["operation_name"] == "feature_iforest_detector"
    if mode is DetectionMode.LABELS:
        assert all(interval.stop >
                   interval.start for interval in result.intervals)
    assert runtime.snapshot.state is RuntimeState.FITTED


def test_detection_runtime_preserves_resampled_time_grid_instead_of_input_index():
    index = pd.to_datetime([
        "2026-06-01 00:00:00",
        "2026-06-01 00:00:01",
        "2026-06-01 00:00:03",
        "2026-06-01 00:00:04",
        "2026-06-01 00:00:05",
        "2026-06-01 00:00:06",
        "2026-06-01 00:00:07",
        "2026-06-01 00:00:08",
        "2026-06-01 00:00:09",
        "2026-06-01 00:00:10",
        "2026-06-01 00:00:11",
        "2026-06-01 00:00:12",
        "2026-06-01 00:00:13",
        "2026-06-01 00:00:14",
        "2026-06-01 00:00:15",
        "2026-06-01 00:00:16",
        "2026-06-01 00:00:17",
        "2026-06-01 00:00:18",
        "2026-06-01 00:00:19",
        "2026-06-01 00:00:20",
    ])
    frame = pd.DataFrame(
        {"sensor": np.sin(np.arange(len(index)) / 3.0)}, index=index)
    runtime = create_detection_runtime(
        profile="tensor",
        operation_name="feature_iforest_detector",
        mode="scores",
        parameters={
            "window_length": 8,
            "target_sample_rate_hz": 1.0,
            "random_state": 42,
            "n_estimators": 10,
        },
    )

    result = runtime.fit(frame).predict(frame)

    assert len(result.time_idx) == len(result.values)
    assert len(result.time_idx) == 21
    assert result.metadata["input_time_index_preserved"] is False


def test_detection_runtime_restores_fedot_output_with_interval_context():
    signal = _signal()
    runtime = create_detection_runtime(
        profile="tensor",
        operation_name="feature_iforest_detector",
        mode="labels",
        parameters={
            "window_length": 12,
            "threshold_quantile": 0.95,
            "random_state": 42,
            "n_estimators": 10,
        },
    ).fit(signal)

    output = runtime.predict_output_data(signal)

    assert isinstance(output, OutputData)
    assert output.data_type is DataTypesEnum.table
    assert output.predict.shape == (80,)
    assert len(output.idx) == len(output.predict)
    contract = output.supplementary_data.temporal_contract
    assert contract["mode"] == "labels"
    assert all(interval["stop"] > interval["start"]
               for interval in contract["intervals"])


def test_detection_runtime_enforces_lifecycle():
    runtime = create_detection_runtime(
        profile="tensor",
        operation_name="feature_iforest_detector",
    )

    with pytest.raises(IntegrationContractError) as error:
        runtime.predict(_signal())
    assert error.value.code is IntegrationErrorCode.INVALID_STATE


def test_detection_runtime_rejects_prepared_plan_drift():
    prepared = prepare_detection_data(
        _signal(),
        stage="train",
        mode="scores",
        causal=False,
    )
    runtime = create_detection_runtime(
        profile="tensor",
        operation_name="feature_iforest_detector",
        mode="labels",
        causal=True,
    )

    with pytest.raises(IntegrationContractError) as error:
        runtime.fit(prepared)

    assert error.value.code is IntegrationErrorCode.PLAN_MISMATCH
    assert set(error.value.context["mismatches"]) == {"mode", "causal"}


def test_causal_detection_is_prefix_invariant_for_probabilities():
    train = _signal(100)
    prefix = _signal(64)
    extended = np.concatenate((prefix, np.linspace(-8.0, 12.0, 24)))
    runtime = create_detection_runtime(
        profile="tensor",
        operation_name="feature_iforest_detector",
        mode="probs",
        causal=True,
        parameters={
            "window_size": 12,
            "stride": 2,
            "threshold_quantile": 0.95,
            "random_state": 42,
            "n_estimators": 20,
        },
    ).fit(train)

    prefix_result = runtime.predict(prefix)
    extended_result = runtime.predict(extended)

    np.testing.assert_allclose(
        prefix_result.values,
        extended_result.values[:len(prefix)],
    )


def test_causal_detection_is_prefix_invariant_during_resampling():
    train = pd.DataFrame(
        {"sensor": _signal(100)},
        index=np.arange(100, dtype=float),
    )
    prefix = pd.DataFrame(
        {"sensor": _signal(64)},
        index=np.arange(64, dtype=float),
    )
    extended = pd.concat([
        prefix,
        pd.DataFrame({"sensor": [99.0]}, index=[63.4]),
    ])
    runtime = create_detection_runtime(
        profile="tensor",
        operation_name="feature_iforest_detector",
        mode="scores",
        causal=True,
        parameters={
            "window_size": 12,
            "stride": 2,
            "target_sample_rate_hz": 1.0,
            "random_state": 42,
            "n_estimators": 20,
        },
    ).fit(train)

    prefix_result = runtime.predict(prefix)
    extended_result = runtime.predict(extended)

    np.testing.assert_array_equal(
        extended_result.time_idx[:len(prefix_result.time_idx)],
        prefix_result.time_idx,
    )
    np.testing.assert_allclose(
        extended_result.values[:len(prefix_result.values)],
        prefix_result.values,
    )


def test_detection_runtime_preserves_external_gap_mask_in_training_windows():
    signal = _signal()
    gap_mask = np.zeros(len(signal), dtype=bool)
    gap_mask[20] = True
    runtime = create_detection_runtime(
        profile="tensor",
        operation_name="feature_iforest_detector",
        mode="scores",
        causal=True,
        parameters={
            "window_size": 10,
            "stride": 2,
            "random_state": 42,
            "n_estimators": 10,
        },
    ).fit(signal, gap_mask=gap_mask)

    operation = runtime._operation

    assert operation.training_batch_.mask is not None
    assert operation.training_batch_.mask.any()
    assert operation.get_stage_diagnostics()["data_quality"]["n_gap_points"] == 1


@pytest.mark.parametrize(
    "parameters",
    [
        {"gap_policy": "interpolate_linear"},
        {"transfer_strategy": "coral"},
    ],
)
def test_causal_detection_rejects_future_dependent_transforms(parameters):
    with pytest.raises(IntegrationContractError) as error:
        create_detection_runtime(
            profile="tensor",
            operation_name="feature_iforest_detector",
            causal=True,
            parameters=parameters,
        )

    assert error.value.code is IntegrationErrorCode.UNSUPPORTED_OPERATION


def test_detection_runtime_rejects_causal_parameter_conflict():
    with pytest.raises(IntegrationContractError) as error:
        create_detection_runtime(
            profile="tensor",
            operation_name="feature_iforest_detector",
            causal=True,
            parameters={"causal": False},
        )

    assert error.value.code is IntegrationErrorCode.PLAN_MISMATCH


def test_detection_runtime_uses_configured_minimum_event_length():
    runtime = create_detection_runtime(
        profile="tensor",
        operation_name="feature_iforest_detector",
        mode="labels",
        causal=True,
        parameters={
            "window_size": 8,
            "min_event_length": 2,
            "random_state": 42,
            "n_estimators": 10,
        },
    ).fit(_signal())
    runtime._operation.score_series_on_values = lambda _data: AnomalyScoreSeries(
        scores=(0.0, 2.0, 0.0, 3.0, 0.0),
        labels=(0, 1, 0, 1, 0),
        threshold=1.0,
        calibration_strategy="test",
    )

    values, intervals, _ = runtime._predict_values(None)

    assert values.tolist() == [0, 1, 0, 1, 0]
    assert intervals == ()
