"""TensorData-backed execution boundary for Industrial forecasters."""

from __future__ import annotations

from typing import Any

import numpy as np
from fedot_ind.core.operation.interfaces.forecasting_runtime_strategy import (
    IndustrialForecastingModelRuntimeStrategy,
    RUNTIME_FORECASTING_MODELS,
)
from fedot_ind.integration.fedot.contracts import (
    DataProfile,
    DataStage,
    IntegrationContractError,
    IntegrationErrorCode,
    RuntimeState,
)
from fedot_ind.integration.fedot.compatibility import OperationParameters
from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
from fedot_ind.integration.fedot.temporal_contracts import (
    ForecastingExecutionPlan,
    ForecastingPrediction,
    PreparedForecastingData,
    TemporalOrientation,
    TemporalRuntimeSnapshot,
)
from fedot_ind.integration.fedot.temporal_planning import (
    infer_future_time_index,
    prepare_forecasting_data,
)
from fedot_ind.integration.fedot.temporal_tensor import (
    ForecastingTensorData,
    create_forecasting_tensor_data,
    forecasting_prediction_to_output_data,
)


class TensorForecastingRuntime:
    """Execute one explicitly supported Industrial forecaster via TensorData."""

    def __init__(self, plan: ForecastingExecutionPlan) -> None:
        _validate_operation(plan.operation_name)
        self.plan = plan
        self._state = RuntimeState.CREATED
        self._strategy: IndustrialForecastingModelRuntimeStrategy | None = None
        self._operation: Any | None = None
        self._train_data: ForecastingTensorData | None = None

    @property
    def snapshot(self) -> TemporalRuntimeSnapshot:
        return TemporalRuntimeSnapshot(
            profile=self.plan.profile,
            state=self._state,
            operation_name=self.plan.operation_name,
            plan=None if self._train_data is None else self._train_data.prepared.plan,
        )

    def fit(
            self,
            data,
            *,
            orientation: TemporalOrientation | str = TemporalOrientation.TIME_FIRST,
            time_idx=None,
            sample_idx=None,
            channel_names=None,
    ) -> "TensorForecastingRuntime":
        self._require_state(RuntimeState.CREATED, "fit")
        prepared = data if isinstance(data, PreparedForecastingData) else prepare_forecasting_data(
            data,
            horizon=self.plan.horizon,
            stage=DataStage.TRAIN,
            orientation=orientation,
            time_idx=time_idx,
            sample_idx=sample_idx,
            channel_names=channel_names,
        )
        if prepared.plan.stage is not DataStage.TRAIN:
            raise IntegrationContractError(
                IntegrationErrorCode.PLAN_MISMATCH,
                "Forecasting fit requires a training data plan.",
                context={"stage": prepared.plan.stage.value},
            )
        self._validate_data_plan(prepared)
        tensor_data = create_forecasting_tensor_data(prepared)
        strategy = IndustrialForecastingModelRuntimeStrategy(
            self.plan.operation_name,
            params=OperationParameters(**self.plan.runtime_parameters()),
        )
        try:
            operation = strategy.fit(tensor_data.to_input_data())
        except Exception as error:
            raise _runtime_error(self.plan.operation_name,
                                 "fit", error) from error
        self._train_data = tensor_data
        self._strategy = strategy
        self._operation = operation
        self._state = RuntimeState.FITTED
        return self

    def predict(
            self,
            data=None,
            *,
            orientation: TemporalOrientation | str = TemporalOrientation.TIME_FIRST,
            time_idx=None,
            sample_idx=None,
            channel_names=None,
            time_step=None,
    ) -> ForecastingPrediction:
        self._require_state(RuntimeState.FITTED, "predict")
        assert self._train_data is not None
        assert self._strategy is not None
        assert self._operation is not None
        if data is None:
            prepared = prepare_forecasting_data(
                np.array(self._train_data.prepared.values, copy=True),
                horizon=self.plan.horizon,
                stage=DataStage.PREDICT,
                orientation=TemporalOrientation.TIME_LAST,
                time_idx=self._train_data.prepared.history_time_idx,
                sample_idx=self._train_data.prepared.sample_idx,
                channel_names=self._train_data.prepared.plan.schema.channel_names,
            )
        elif isinstance(data, PreparedForecastingData):
            prepared = data
        else:
            prepared = prepare_forecasting_data(
                data,
                horizon=self.plan.horizon,
                stage=DataStage.PREDICT,
                orientation=orientation,
                time_idx=time_idx,
                sample_idx=sample_idx,
                channel_names=channel_names,
            )
        if prepared.plan.stage is not DataStage.PREDICT:
            raise IntegrationContractError(
                IntegrationErrorCode.PLAN_MISMATCH,
                "Forecasting predict requires a prediction data plan.",
                context={"stage": prepared.plan.stage.value},
            )
        self._validate_data_plan(prepared)
        _validate_predict_schema(self._train_data.prepared, prepared)
        tensor_data = create_forecasting_tensor_data(
            prepared, from_data=self._train_data)
        try:
            output = self._strategy.predict(
                self._operation, tensor_data.to_input_data())
        except Exception as error:
            raise _runtime_error(self.plan.operation_name,
                                 "predict", error) from error
        raw_prediction = getattr(output, "predict", output)
        values = _normalize_forecast_values(
            raw_prediction,
            horizon=self.plan.horizon,
            series_count=prepared.plan.schema.series_count,
        )
        forecast_idx = infer_future_time_index(
            prepared.history_time_idx,
            self.plan.horizon,
            step=time_step,
        )
        return ForecastingPrediction(
            values=values,
            forecast_time_idx=forecast_idx,
            sample_idx=prepared.sample_idx,
            horizon=self.plan.horizon,
            metadata={
                "operation_name": self.plan.operation_name,
                "history_length": prepared.plan.schema.history_length,
            },
        )

    def close(self) -> None:
        self._operation = None
        self._strategy = None
        self._train_data = None
        self._state = RuntimeState.CLOSED

    def predict_output_data(self, data=None, **kwargs):
        """Return the forecast through FEDOT's public output container."""
        prediction = self.predict(data, **kwargs)
        assert self._train_data is not None
        return forecasting_prediction_to_output_data(prediction, self._train_data)

    def _require_state(self, expected: RuntimeState, phase: str) -> None:
        if self._state is not expected:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_STATE,
                f"Forecasting runtime cannot {phase} in its current state.",
                context={"state": self._state.value,
                         "expected": expected.value},
            )

    def _validate_data_plan(self, prepared: PreparedForecastingData) -> None:
        if prepared.plan.horizon != self.plan.horizon:
            raise IntegrationContractError(
                IntegrationErrorCode.INVALID_HORIZON,
                "Forecasting data horizon must match the execution plan.",
                context={"execution": self.plan.horizon,
                         "data": prepared.plan.horizon},
            )
        schema = prepared.plan.schema
        if schema.series_count != 1 or schema.channel_count != 1:
            raise IntegrationContractError(
                IntegrationErrorCode.SCHEMA_MISMATCH,
                "The current forecasting runtime supports one univariate series.",
                context={
                    "series_count": schema.series_count,
                    "channel_count": schema.channel_count,
                    "operation": self.plan.operation_name,
                },
            )


def create_forecasting_runtime(
        *,
        profile: DataProfile | str,
        operation_name: str,
        horizon: int,
        parameters: dict[str, Any] | None = None,
) -> TensorForecastingRuntime:
    """Build a validated forecasting runtime without a legacy fallback."""
    try:
        normalized_profile = profile if isinstance(
            profile, DataProfile) else DataProfile(profile)
    except (TypeError, ValueError) as error:
        raise IntegrationContractError(
            IntegrationErrorCode.UNKNOWN_PROFILE,
            "Forecasting runtime profile is unsupported.",
            context={"profile": profile},
            cause=error,
        ) from error
    return TensorForecastingRuntime(ForecastingExecutionPlan(
        profile=normalized_profile,
        operation_name=operation_name,
        horizon=horizon,
        parameters=parameters or {},
    ))


def _validate_operation(operation_name: str) -> None:
    declaration = load_industrial_extension_catalog().operation(operation_name)
    if (
            declaration is None
            or "ts_forecasting" not in declaration.tasks
            or operation_name not in RUNTIME_FORECASTING_MODELS
    ):
        raise IntegrationContractError(
            IntegrationErrorCode.UNSUPPORTED_OPERATION,
            "Forecasting operation is not available through the TensorData runtime.",
            context={"operation": operation_name},
        )


def _validate_predict_schema(
        train: PreparedForecastingData,
        predict: PreparedForecastingData,
) -> None:
    train_schema = train.plan.schema
    predict_schema = predict.plan.schema
    mismatches = {
        field: {"train": getattr(train_schema, field),
                "predict": getattr(predict_schema, field)}
        for field in ("series_count", "channel_count", "channel_names")
        if getattr(train_schema, field) != getattr(predict_schema, field)
    }
    if mismatches or not np.array_equal(train.sample_idx, predict.sample_idx):
        raise IntegrationContractError(
            IntegrationErrorCode.SCHEMA_MISMATCH,
            "Forecasting prediction data must preserve fitted series and channels.",
            context={"mismatches": mismatches},
        )


def _normalize_forecast_values(values: Any, *, horizon: int, series_count: int) -> np.ndarray:
    array = values.detach().cpu().numpy() if hasattr(
        values, "detach") else np.asarray(values)
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        if series_count != 1 or len(array) != horizon:
            raise _forecast_length_error(array, horizon, series_count)
        return array
    if array.ndim == 2:
        if series_count == 1 and array.shape in {(1, horizon), (horizon, 1)}:
            return array.reshape(horizon)
        if array.shape == (series_count, horizon):
            return array
        if array.shape == (horizon, series_count):
            return array.T
    raise _forecast_length_error(array, horizon, series_count)


def _forecast_length_error(array: np.ndarray, horizon: int, series_count: int) -> IntegrationContractError:
    return IntegrationContractError(
        IntegrationErrorCode.LENGTH_MISMATCH,
        "Forecast model output does not match the requested horizon.",
        context={"shape": list(array.shape), "horizon": horizon,
                 "series_count": series_count},
    )


def _runtime_error(operation: str, phase: str, error: Exception) -> IntegrationContractError:
    if isinstance(error, IntegrationContractError):
        return error
    return IntegrationContractError(
        IntegrationErrorCode.RUNTIME_FAILURE,
        "Forecasting runtime execution failed.",
        context={"operation": operation, "phase": phase,
                 "cause_type": type(error).__name__},
        cause=error,
    )


__all__ = ["TensorForecastingRuntime", "create_forecasting_runtime"]
