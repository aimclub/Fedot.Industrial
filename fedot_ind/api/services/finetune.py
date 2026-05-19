"""Finetune service for ``FedotIndustrial``."""

from dataclasses import dataclass
from typing import Any, Callable, Dict

FEDOT_TUNER_STRATEGY = {}
FEDOT_TUNING_METRICS = {}


def _tuner_strategy_registry():
    global FEDOT_TUNER_STRATEGY
    if not FEDOT_TUNER_STRATEGY:
        from fedot_ind.core.repository.constanst_repository import FEDOT_TUNER_STRATEGY as registry
        FEDOT_TUNER_STRATEGY = registry
    return FEDOT_TUNER_STRATEGY


def _tuning_metrics_registry():
    global FEDOT_TUNING_METRICS
    if not FEDOT_TUNING_METRICS:
        from fedot_ind.core.repository.constanst_repository import FEDOT_TUNING_METRICS as registry
        FEDOT_TUNING_METRICS = registry
    return FEDOT_TUNING_METRICS


def build_tuner(api: Any, **kwargs: Any) -> Any:
    from fedot_ind.core.repository.industrial_implementations.abstract import (
        build_tuner as industrial_build_tuner,
    )
    return industrial_build_tuner(api, **kwargs)


@dataclass(frozen=True)
class FinetunePayload:
    """Prepared inputs for tuner construction or direct model fitting."""

    train_data: Any
    model_to_tune: Any
    tuning_params: Dict[str, Any]

    def as_kwargs(self) -> Dict[str, Any]:
        return {
            "train_data": self.train_data,
            "model_to_tune": self.model_to_tune,
            "tuning_params": self.tuning_params,
        }


class FinetuneService:
    """Prepare finetune data and run either direct fitting or tuner construction."""

    def prepare_tuning_params(self, tuning_params: Dict[str, Any], task: str) -> Dict[str, Any]:
        tuning_params["metric"] = _tuning_metrics_registry()[task]
        tuning_params["tuner"] = _tuner_strategy_registry()[
            tuning_params.get("tuner", "sequential")
        ]
        return tuning_params

    def prepare_payload(
            self,
            *,
            train_data: Any,
            tuning_params: Dict[str, Any],
            model_to_tune: Any,
            task: str,
            is_fedot_datatype: bool,
            process_input: Callable[[Any], Any],
            init_backend: Callable[[Any], Any],
    ) -> FinetunePayload:
        if model_to_tune is None:
            raise ValueError("model_to_tune must be provided for finetune")

        processed_data = train_data if is_fedot_datatype else process_input(train_data)
        processed_data = init_backend(processed_data)
        prepared_params = self.prepare_tuning_params(tuning_params, task)
        return FinetunePayload(
            train_data=processed_data,
            model_to_tune=self._materialize_model(model_to_tune),
            tuning_params=prepared_params,
        )

    def run(
            self,
            *,
            api: Any,
            payload: FinetunePayload,
            return_only_fitted: bool,
    ) -> Any:
        if return_only_fitted:
            payload.model_to_tune.fit(payload.train_data)
            return payload.model_to_tune
        return build_tuner(api, **payload.as_kwargs())

    @staticmethod
    def _materialize_model(model_to_tune: Any) -> Any:
        if hasattr(model_to_tune, "build"):
            return model_to_tune.build()
        return model_to_tune
