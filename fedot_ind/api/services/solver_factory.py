"""FEDOT solver factory service."""

from typing import Any

from fedot.api.main import Fedot

from fedot_ind.api.flow import IndustrialTaskKind, SolverInitPlan, unwrap_or_raise


def build_solver_init_plan(manager: Any) -> SolverInitPlan:
    """Build a typed solver initialization plan from the API manager."""
    task_kind = unwrap_or_raise(IndustrialTaskKind.normalize(manager.automl_config.config['task']))
    task_params = (
        manager.industrial_config.task_params
        if manager.industrial_config.is_forecasting_context
        else manager.automl_config.config['task_params']
    )
    return SolverInitPlan.create(
        task_kind=task_kind,
        learning_strategy_params=manager.learning_config.config['learning_strategy_params'],
        optimisation_loss=manager.learning_config.config['optimisation_loss'],
        task_params=task_params,
        optimizer=manager.automl_config.optimisation_strategy,
        available_operations=manager.automl_config.config['available_operations'],
        initial_assumption=manager.automl_config.config['initial_assumption'],
    )


class SolverFactory:
    """Create FEDOT solvers from typed initialization plans."""

    def __init__(self, fedot_cls=Fedot):
        self.fedot_cls = fedot_cls

    def create(self, plan: SolverInitPlan) -> Fedot:
        return self.fedot_cls(
            **plan.learning_strategy_params,
            metric=plan.optimisation_loss,
            problem=plan.fedot_problem,
            task_params=plan.task_params,
            optimizer=plan.optimizer,
            available_operations=list(plan.available_operations),
            initial_assumption=plan.initial_assumption,
        )
