"""Repository activation service for ``FedotIndustrial``."""

from dataclasses import dataclass
from functools import partial
from typing import Any

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


@dataclass(frozen=True)
class RepositoryActivationResult:
    """Result of activating the FEDOT/Industrial operation repository."""

    repo: Any
    input_data: Any = None


class IndustrialRepositoryInitializer:
    """Activate the correct operation repository for the current context."""

    def __init__(self, industrial_models_factory=IndustrialModels):
        self.industrial_models_factory = industrial_models_factory

    def setup_repository(self, *, backend: str) -> Any:
        """Set up the Industrial repository without touching optimizer config."""
        return self.industrial_models_factory().setup_repository(backend=backend)

    def activate(self, *, manager: Any, logger: Any, input_data: Any = None) -> RepositoryActivationResult:
        logger.info('-' * 50)
        logger.info('Initialising Industrial Repository')
        industrial_models = self.industrial_models_factory()

        if manager.industrial_config.is_default_fedot_context:
            logger.info('-------------------------------------------------')
            logger.info('Initialising Fedot Evolutionary Optimisation params')
            repo = industrial_models.setup_default_repository()
            manager.automl_config.optimisation_strategy = manager.optimisation_agent['Fedot']
            return RepositoryActivationResult(repo=repo, input_data=input_data)

        logger.info('-------------------------------------------------')
        logger.info('Initialising Industrial Evolutionary Optimisation params')
        repo = industrial_models.setup_repository(backend=manager.compute_config.backend)
        optimisation_agent = manager.automl_config.optimisation_strategy['optimisation_agent']
        optimisation_params = manager.automl_config.optimisation_strategy['optimisation_strategy']
        manager.automl_config.optimisation_strategy = partial(
            manager.optimisation_agent[optimisation_agent],
            optimisation_params=optimisation_params,
        )
        return RepositoryActivationResult(repo=repo, input_data=input_data)
