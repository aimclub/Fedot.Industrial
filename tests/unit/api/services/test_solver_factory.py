from __future__ import annotations

from types import SimpleNamespace

import pytest

from fedot_ind.api.flow import IndustrialTaskKind, SolverInitPlan
from fedot_ind.api.services.solver_factory import SolverFactory, build_solver_init_plan


def test_solver_factory_passes_plan_fields_to_fedot_constructor():
    created = {}

    class FakeFedot:
        def __init__(self, **kwargs):
            created.update(kwargs)

    plan = SolverInitPlan.create(
        task_kind=IndustrialTaskKind.CLASSIFICATION,
        learning_strategy_params={"timeout": 1, "n_jobs": 2},
        optimisation_loss={"quality_loss": "accuracy"},
        task_params={"task": "params"},
        optimizer="optimizer",
        available_operations=["rf"],
        initial_assumption="pipeline",
    )

    solver = SolverFactory(FakeFedot).create(plan)

    assert isinstance(solver, FakeFedot)
    assert created == {
        "timeout": 1,
        "n_jobs": 2,
        "metric": {"quality_loss": "accuracy"},
        "problem": "classification",
        "task_params": {"task": "params"},
        "optimizer": "optimizer",
        "available_operations": ["rf"],
        "initial_assumption": "pipeline",
    }


def test_build_solver_init_plan_uses_automl_task_params_for_classification():
    manager = SimpleNamespace(
        automl_config=SimpleNamespace(
            config={
                "task": "classification",
                "task_params": {"automl": True},
                "available_operations": ["rf"],
                "initial_assumption": "pipeline",
            },
            optimisation_strategy="optimizer",
        ),
        industrial_config=SimpleNamespace(
            is_forecasting_context=False,
            task_params={"industrial": True},
        ),
        learning_config=SimpleNamespace(
            config={
                "learning_strategy_params": {"timeout": 1},
                "optimisation_loss": {"quality_loss": "accuracy"},
            }
        ),
    )

    plan = build_solver_init_plan(manager)

    assert plan.task_kind is IndustrialTaskKind.CLASSIFICATION
    assert plan.task_params == {"automl": True}
    assert plan.available_operations == ("rf",)
    assert plan.initial_assumption == "pipeline"


def test_build_solver_init_plan_uses_industrial_task_params_for_forecasting():
    manager = SimpleNamespace(
        automl_config=SimpleNamespace(
            config={
                "task": "ts_forecasting",
                "task_params": {"automl": True},
                "available_operations": ["lagged"],
                "initial_assumption": None,
            },
            optimisation_strategy="optimizer",
        ),
        industrial_config=SimpleNamespace(
            is_forecasting_context=True,
            task_params={"forecast_length": 14},
        ),
        learning_config=SimpleNamespace(
            config={
                "learning_strategy_params": {"timeout": 1},
                "optimisation_loss": {"quality_loss": "rmse"},
            }
        ),
    )

    plan = build_solver_init_plan(manager)

    assert plan.task_kind is IndustrialTaskKind.TS_FORECASTING
    assert plan.task_params == {"forecast_length": 14}


def test_build_solver_init_plan_rejects_invalid_task_kind():
    manager = SimpleNamespace(
        automl_config=SimpleNamespace(
            config={
                "task": "ranking",
                "task_params": {},
                "available_operations": [],
                "initial_assumption": None,
            },
            optimisation_strategy=None,
        ),
        industrial_config=SimpleNamespace(is_forecasting_context=False, task_params={}),
        learning_config=SimpleNamespace(
            config={"learning_strategy_params": {}, "optimisation_loss": None}
        ),
    )

    with pytest.raises(ValueError, match="invalid_task_kind"):
        build_solver_init_plan(manager)
