"""Metric evaluation service for ``FedotIndustrial``."""

from typing import Any

FEDOT_GET_METRICS = {}


def _metric_registry():
    global FEDOT_GET_METRICS
    if not FEDOT_GET_METRICS:
        from fedot_ind.core.repository.constanst_repository import FEDOT_GET_METRICS as registry
        FEDOT_GET_METRICS = registry
    return FEDOT_GET_METRICS


class MetricEvaluationService:
    """Evaluate Industrial API metrics for labels/probabilities."""

    def evaluate(
            self,
            *,
            target: Any,
            predicted_labels: Any,
            predicted_probs: Any,
            problem: str,
            metric_names: Any,
            rounding_order: int,
            train_data: Any,
            seasonality: int,
            condition_check: Any,
            target_encoder: Any,
    ) -> Any:
        valid_shape = target.shape
        metrics = _metric_registry()
        if isinstance(predicted_labels, dict):
            return {
                model_name: metrics[problem](
                    target=target,
                    metric_names=metric_names,
                    rounding_order=rounding_order,
                    labels=model_result,
                    probs=predicted_probs,
                )
                for model_name, model_result in predicted_labels.items()
            }

        if condition_check.solver_have_target_encoder(target_encoder):
            new_target = target_encoder.transform(target.flatten())
            labels = target_encoder.transform(predicted_labels).reshape(valid_shape)
        else:
            new_target = target.flatten()
            labels = predicted_labels.reshape(valid_shape)

        return metrics[problem](
            target=new_target,
            metric_names=metric_names,
            rounding_order=rounding_order,
            labels=labels,
            probs=predicted_probs,
            train_data=train_data,
            seasonality=seasonality,
        )
