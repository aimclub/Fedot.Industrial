"""Prediction routing service for ``FedotIndustrial``."""

from typing import Any

from fedot_ind.integration.fedot.compatibility import OutputData


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
        condition_check = manager.condition_check
        solver = manager.solver
        have_encoder = condition_check.solver_have_target_encoder(target_encoder)
        custom_predict = all([
            not condition_check.solver_is_fedot_class(solver),
            not condition_check.solver_is_pipeline_class(solver),
        ])

        if custom_predict:
            prediction = solver.predict(predict_data)
        else:
            prediction = self._predict_with_solver(
                manager=manager,
                predict_data=predict_data,
                predict_mode=predict_mode,
            )

        output_is_data = isinstance(prediction, OutputData)
        if have_encoder:
            prediction = self._inverse_encoder_transform(
                prediction=prediction,
                target_encoder=target_encoder,
                predict_data=predict_data,
            )
        if output_is_data:
            prediction = prediction.predict
        if self._is_forecasting_data(predict_data):
            self._validate_forecast_length(prediction, predict_data)
        return prediction

    @staticmethod
    def _predict_with_solver(*, manager: Any, predict_data: Any, predict_mode: str) -> Any:
        solver = manager.solver
        if manager.condition_check.solver_is_pipeline_class(solver):
            return solver.predict(predict_data, predict_mode)
        if predict_mode in ["labels"]:
            return solver.predict(predict_data)
        return solver.predict_proba(predict_data)

    @staticmethod
    def _inverse_encoder_transform(*, prediction: Any, target_encoder: Any, predict_data: Any) -> Any:
        predicted_labels = target_encoder.inverse_transform(prediction)
        predict_data.target = target_encoder.inverse_transform(predict_data.target)
        return predicted_labels

    @staticmethod
    def _is_forecasting_data(predict_data: Any) -> bool:
        task = getattr(predict_data, "task", None)
        task_type = getattr(task, "task_type", None)
        task_value = getattr(task_type, "value", "")
        return "forecasting" in str(task_value)

    @staticmethod
    def _validate_forecast_length(prediction: Any, predict_data: Any) -> None:
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
