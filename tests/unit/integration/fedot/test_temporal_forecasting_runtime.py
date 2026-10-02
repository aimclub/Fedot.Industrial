from fedot_ind.integration.fedot.compatibility import DataTypesEnum, OutputData
from fedot_ind.integration.fedot import (
    DataProfile,
    IntegrationContractError,
    IntegrationErrorCode,
    RuntimeState,
    create_forecasting_runtime,
    prepare_forecasting_data,
)
import numpy as np
import pandas as pd
import pytest
import fedot

pytestmark = pytest.mark.skipif(
    not hasattr(fedot, "create_data"),
    reason="The FEDOT TensorData profile is not installed.",
)


def test_forecasting_runtime_rejects_legacy_profile_and_unowned_operation():
    with pytest.raises(IntegrationContractError) as profile_error:
        create_forecasting_runtime(
            profile="legacy",
            operation_name="lagged_ridge_forecaster",
            horizon=3,
        )
    assert profile_error.value.code is IntegrationErrorCode.UNKNOWN_PROFILE

    with pytest.raises(IntegrationContractError) as operation_error:
        create_forecasting_runtime(
            profile="tensor",
            operation_name="ar",
            horizon=3,
        )
    assert operation_error.value.code is IntegrationErrorCode.UNSUPPORTED_OPERATION


def test_lagged_ridge_forecasting_runtime_preserves_future_datetime_coordinates():
    index = pd.date_range("2026-05-01", periods=80, freq="h")
    series = pd.Series(
        np.sin(np.arange(80) / 5.0) + np.arange(80) * 0.01,
        index=index,
    )
    runtime = create_forecasting_runtime(
        profile=DataProfile.TENSOR,
        operation_name="lagged_ridge_forecaster",
        horizon=4,
        parameters={"window_size": 16, "stride": 1, "alpha": 1.0},
    )

    result = runtime.fit(series).predict()

    assert result.values.shape == (4,)
    np.testing.assert_array_equal(
        result.forecast_time_idx,
        pd.date_range("2026-05-04 08:00", periods=4, freq="h").to_numpy(),
    )
    assert result.sample_idx.tolist() == [0]
    assert result.metadata["operation_name"] == "lagged_ridge_forecaster"
    assert runtime.snapshot.state is RuntimeState.FITTED
    output = runtime.predict_output_data()
    assert isinstance(output, OutputData)
    assert output.data_type is DataTypesEnum.table
    np.testing.assert_array_equal(output.idx, result.forecast_time_idx)
    np.testing.assert_allclose(output.predict, result.values)
    assert output.supplementary_data.temporal_contract["horizon"] == 4
    runtime.close()
    assert runtime.snapshot.state is RuntimeState.CLOSED


def test_forecasting_runtime_accepts_longer_prediction_history():
    train = np.sin(np.arange(80, dtype=float) / 5.0)
    predict = np.sin(np.arange(81, dtype=float) / 5.0)
    runtime = create_forecasting_runtime(
        profile="tensor",
        operation_name="lagged_ridge_forecaster",
        horizon=3,
        parameters={"window_size": 16, "alpha": 1.0},
    ).fit(train)

    result = runtime.predict(predict)

    assert result.values.shape == (3,)
    assert result.forecast_time_idx.tolist() == [81, 82, 83]
    assert result.metadata["history_length"] == 81


def test_forecasting_runtime_preserves_vector_shape_for_horizon_one():
    series = np.sin(np.arange(64, dtype=float) / 4.0)
    runtime = create_forecasting_runtime(
        profile="tensor",
        operation_name="lagged_ridge_forecaster",
        horizon=1,
        parameters={"window_size": 12, "alpha": 1.0},
    ).fit(series)

    result = runtime.predict()

    assert result.values.shape == (1,)
    assert result.forecast_time_idx.tolist() == [64]


@pytest.mark.parametrize(
    "data, orientation",
    [
        (np.arange(80, dtype=float).reshape(2, 40), "time_last"),
        (np.arange(80, dtype=float).reshape(40, 2), "time_first"),
    ],
)
def test_forecasting_runtime_rejects_ambiguous_multiseries_or_multichannel_input(
        data,
        orientation,
):
    runtime = create_forecasting_runtime(
        profile="tensor",
        operation_name="lagged_ridge_forecaster",
        horizon=2,
    )

    with pytest.raises(IntegrationContractError) as error:
        runtime.fit(data, orientation=orientation)

    assert error.value.code is IntegrationErrorCode.SCHEMA_MISMATCH


def test_forecasting_runtime_rejects_predict_schema_drift_before_model_call():
    train = prepare_forecasting_data(
        np.arange(60, dtype=float),
        horizon=3,
        sample_idx=["asset-a"],
    )
    predict = prepare_forecasting_data(
        np.column_stack((np.arange(60), np.arange(60) * 2.0)),
        horizon=3,
        stage="predict",
        sample_idx=["asset-a"],
        channel_names=["left", "right"],
    )
    runtime = create_forecasting_runtime(
        profile="tensor",
        operation_name="lagged_ridge_forecaster",
        horizon=3,
        parameters={"window_size": 12, "alpha": 1.0},
    ).fit(train)

    with pytest.raises(IntegrationContractError) as error:
        runtime.predict(predict)

    assert error.value.code is IntegrationErrorCode.SCHEMA_MISMATCH


def test_forecasting_runtime_enforces_lifecycle():
    runtime = create_forecasting_runtime(
        profile="tensor",
        operation_name="lagged_ridge_forecaster",
        horizon=2,
    )

    with pytest.raises(IntegrationContractError) as error:
        runtime.predict(np.arange(20))
    assert error.value.code is IntegrationErrorCode.INVALID_STATE
