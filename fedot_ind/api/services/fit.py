"""Fit service for ``FedotIndustrial``."""

from typing import Any, Callable


class IndustrialFitService:
    """Run model fitting through either custom strategy or FEDOT solver."""

    def fit(self, manager: Any, train_data: Any) -> Any:
        strategy = manager.industrial_config.strategy
        if isinstance(strategy, Callable):
            return strategy.fit(train_data)
        return manager.solver.fit(train_data)
