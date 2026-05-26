"""Prediction routing service for ``FedotIndustrial``."""

from typing import Any

from fedot.core.data.data import OutputData


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
            prediction = prediction[-predict_data.task.task_params.forecast_length:]
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
