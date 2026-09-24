"""FEDOT extension activation service for ``FedotIndustrial``."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from fedot_ind.integration.fedot.extensions import (
    IndustrialExtensionResult,
    IndustrialExtensionSession,
)


@dataclass(frozen=True)
class RepositoryActivationResult:
    """Result of configuring one Industrial execution context."""

    extension: IndustrialExtensionResult | None
    input_data: Any = None


class IndustrialRepositoryInitializer:
    """Own extension registration and optimizer selection for one API instance."""

    def __init__(
            self,
            session_factory: Callable[[], IndustrialExtensionSession] = IndustrialExtensionSession,
    ) -> None:
        self._session = session_factory()

    def ensure_active(self, *, industrial_context: bool) -> IndustrialExtensionResult | None:
        """Activate the Industrial manifest when the selected context needs it."""
        return self._session.activate() if industrial_context else None

    def close(self) -> None:
        self._session.close()

    def activate(self, *, manager: Any, logger: Any, input_data: Any = None) -> RepositoryActivationResult:
        logger.info('-' * 50)
        logger.info('Initialising Industrial Repository')
        if manager.industrial_config.is_default_fedot_context:
            logger.info('-------------------------------------------------')
            logger.info('Initialising Fedot Evolutionary Optimisation params')
            manager.automl_config.optimisation_strategy = manager.optimisation_agent['Fedot']
            return RepositoryActivationResult(extension=None, input_data=input_data)

        logger.info('-------------------------------------------------')
        logger.info('Initialising Industrial Evolutionary Optimisation params')
        extension = self.ensure_active(industrial_context=True)
        optimisation_agent = manager.automl_config.optimisation_strategy['optimisation_agent']
        optimisation_params = {
            **manager.automl_config.optimisation_strategy['optimisation_strategy'],
            'initial_graphs_prevalidated': True,
        }
        manager.automl_config.optimisation_strategy = partial(
            manager.optimisation_agent[optimisation_agent],
            optimisation_params=optimisation_params,
        )
        return RepositoryActivationResult(extension=extension, input_data=input_data)
