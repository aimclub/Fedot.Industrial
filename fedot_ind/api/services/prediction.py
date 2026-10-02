"""Prediction routing service for ``FedotIndustrial``."""

from typing import Any

from fedot_ind.integration.fedot.compatibility import (
    OutputData,
    TensorData,
    as_numpy,
    ensure_fedot_tensor_data,
)


class PredictionService:
    """Route prediction calls across FEDOT, Pipeline and custom strategy solvers."""

    def predict_output(
            self,
            *,
            manager: Any,
            target_encoder: Any,
            predict_data: Any,
            predict_mode: str,
    ) -> Any:
        """Route prediction to the configured solver and return its prediction values.

        Custom solvers receive the original data and ignore ``predict_mode``.
        FEDOT solvers receive data converted with the stored training reference;
        FEDOT output containers are unwrapped to NumPy values. If an encoder is
        active, decode predictions and update any target on the runtime data.

        Solver and decoder errors propagate. Forecasts with an observed length
        other than the requested horizon raise ValueError.
        """
        condition_check = manager.condition_check
        solver = manager.solver
        have_encoder = condition_check.solver_have_target_encoder(target_encoder)
        custom_predict = all([
            not condition_check.solver_is_fedot_class(solver),
            not condition_check.solver_is_pipeline_class(solver),
        ])

        if custom_predict:
            runtime_data = predict_data
            prediction = solver.predict(predict_data)
        else:
            runtime_data = ensure_fedot_tensor_data(
                predict_data,
                fit_stage=False,
                reference_data=getattr(manager, "fedot_train_data", None),
            )
            prediction = self._predict_with_solver(
                manager=manager,
                predict_data=runtime_data,
                predict_mode=predict_mode,
            )

        output_is_data = isinstance(prediction, (OutputData, TensorData))
        prediction_value = prediction.predict if output_is_data else prediction
        if have_encoder:
            prediction_value = self._inverse_encoder_transform(
                prediction=prediction_value,
                target_encoder=target_encoder,
                predict_data=runtime_data,
            )
        if output_is_data:
            prediction_value = as_numpy(prediction_value)
        if self._is_forecasting_data(runtime_data):
            self._validate_forecast_length(prediction_value, runtime_data)
        return prediction_value

    @staticmethod
    def _predict_with_solver(*, manager: Any, predict_data: Any, predict_mode: str) -> Any:
        solver = manager.solver
        if manager.condition_check.solver_is_pipeline_class(solver):
            return solver.predict(predict_data, predict_mode)
        if predict_mode in ["labels"]:
            return solver.predict(predict_data)
        current_pipeline = getattr(solver, "current_pipeline", None)
        if current_pipeline is not None:
            return current_pipeline.predict(predict_data, output_mode=predict_mode)
        return solver.predict_proba(predict_data)

    @staticmethod
    def _inverse_encoder_transform(*, prediction: Any, target_encoder: Any, predict_data: Any) -> Any:
        """Decode predictions and, when present, replace the input target in place.

        Return decoded predictions; conversion and encoder errors propagate.
        """
        predicted_labels = target_encoder.inverse_transform(as_numpy(prediction))
        if getattr(predict_data, "target", None) is not None:
            predict_data.target = target_encoder.inverse_transform(as_numpy(predict_data.target))
        return predicted_labels

    @staticmethod
    def _is_forecasting_data(predict_data: Any) -> bool:
        task = getattr(predict_data, "task", None)
        task_type = getattr(task, "task_type", None)
        task_value = getattr(task_type, "value", "")
        return "forecasting" in str(task_value)

    @staticmethod
    def _validate_forecast_length(prediction: Any, predict_data: Any) -> None:
        """Raise ValueError when the observed output length differs from the horizon.

        For multidimensional outputs, use the last axis when it equals the
        horizon, otherwise use the first axis. Zero-dimensional arrays have observed length zero.
        """
        horizon = int(predict_data.task.task_params.forecast_length)
        shape = getattr(prediction, "shape", None)
        if shape is None:
            observed = len(prediction)
        elif len(shape) == 0:
            observed = 0
        elif len(shape) == 1:
            observed = shape[0]
        else:
            observed = shape[-1] if shape[-1] == horizon else shape[0]
        if observed != horizon:
            raise ValueError(
                f"Forecast output must contain exactly {horizon} steps, got {observed}."
            )
