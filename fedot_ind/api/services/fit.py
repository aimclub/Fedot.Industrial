"""Fit service for ``FedotIndustrial``."""

from typing import Any, Callable

from fedot_ind.integration.fedot.compatibility import ensure_fedot_tensor_data


class IndustrialFitService:
    """Run model fitting through either custom strategy or FEDOT solver."""

    def fit(self, manager: Any, train_data: Any) -> Any:
        strategy = manager.industrial_config.strategy
        if isinstance(strategy, Callable):
            return strategy.fit(train_data)
        tensor_data = ensure_fedot_tensor_data(train_data, fit_stage=True)
        manager.fedot_train_data = tensor_data
        return manager.solver.fit(tensor_data)
