from __future__ import annotations

import pytest

from fedot_ind.api.flow import (
    IndustrialFlowError,
    IndustrialTaskKind,
    InitialAssumptionSource,
    PredictionModePlan,
    PredictionOutputMode,
    SolverInitPlan,
    plan_initial_assumption,
    unwrap_or_raise,
)


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("classification", IndustrialTaskKind.CLASSIFICATION),
        ("ts_classification", IndustrialTaskKind.CLASSIFICATION),
        ("regression", IndustrialTaskKind.REGRESSION),
        ("ts_regression", IndustrialTaskKind.REGRESSION),
        ("forecasting", IndustrialTaskKind.TS_FORECASTING),
        ("ts_forecasting", IndustrialTaskKind.TS_FORECASTING),
        (IndustrialTaskKind.CLASSIFICATION, IndustrialTaskKind.CLASSIFICATION),
    ],
)
def test_task_kind_normalizes_known_task_strings(raw_value, expected):
    assert unwrap_or_raise(IndustrialTaskKind.normalize(raw_value)) is expected


def test_task_kind_invalid_value_returns_stable_domain_error():
    result = IndustrialTaskKind.normalize("ranking")

    assert result.is_left()
    error = result.monoid[0]
    assert isinstance(error, IndustrialFlowError)
    assert error.code == "invalid_task_kind"
    assert error.value == "ranking"
    with pytest.raises(ValueError, match="invalid_task_kind"):
        unwrap_or_raise(result)


def test_initial_assumption_plan_prefers_automl_config():
    plan = plan_initial_assumption(automl_initial="automl", industrial_initial="industrial")

    assert plan.source is InitialAssumptionSource.AUTOML_CONFIG
    assert plan.assumption == "automl"
    assert plan.has_assumption


def test_initial_assumption_plan_uses_industrial_config_as_fallback():
    plan = plan_initial_assumption(automl_initial=None, industrial_initial="industrial")

    assert plan.source is InitialAssumptionSource.INDUSTRIAL_CONFIG
    assert plan.assumption == "industrial"
    assert plan.has_assumption


def test_initial_assumption_plan_represents_absence_explicitly():
    plan = plan_initial_assumption()

    assert plan.source is InitialAssumptionSource.NONE
    assert plan.assumption is None
    assert not plan.has_assumption


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        (None, PredictionOutputMode.DEFAULT),
        ("labels", PredictionOutputMode.LABELS),
        ("probabilities", PredictionOutputMode.PROBABILITIES),
        (PredictionOutputMode.PROBABILITIES, PredictionOutputMode.PROBABILITIES),
    ],
)
def test_prediction_output_mode_normalizes_aliases(raw_value, expected):
    assert unwrap_or_raise(PredictionOutputMode.normalize(raw_value)) is expected


def test_prediction_output_mode_invalid_value_is_domain_error():
    result = PredictionOutputMode.normalize("scores")

    assert result.is_left()
    assert result.monoid[0].code == "invalid_prediction_mode"


def test_prediction_mode_plan_names_probability_requirement():
    plan = PredictionModePlan(
        task_kind=IndustrialTaskKind.CLASSIFICATION,
        output_mode=PredictionOutputMode.PROBABILITIES,
    )

    assert plan.requires_probabilities


def test_solver_init_plan_is_deterministic_and_tuple_normalized():
    plan = SolverInitPlan.create(
        task_kind=IndustrialTaskKind.CLASSIFICATION,
        learning_strategy_params={"timeout": 1},
        optimisation_loss={"quality_loss": "accuracy"},
        available_operations=["rf", "scaling"],
        initial_assumption="pipeline",
    )

    assert plan.fedot_problem == "classification"
    assert plan.learning_strategy_params == {"timeout": 1}
    assert plan.available_operations == ("rf", "scaling")
    assert "SolverInitPlan" in repr(plan)
