import sys

import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier

from fedot_ind.core.models.pdl.base_models import (
    available_pdl_base_models,
    create_pdl_base_model,
)


def test_default_pdl_models_are_resolved_inside_the_thematic_package():
    classifier = create_pdl_base_model(
        "classification", "rf", {"n_estimators": 2, "random_state": 42}
    )
    regressor = create_pdl_base_model(
        "regression", "treg", {"n_estimators": 2, "random_state": 42}
    )

    assert isinstance(classifier, RandomForestClassifier)
    assert isinstance(regressor, ExtraTreesRegressor)
    assert "rf" in available_pdl_base_models("classification")
    assert "treg" in available_pdl_base_models("regression")


def test_optional_estimator_modules_are_not_loaded_by_default(monkeypatch):
    monkeypatch.delitem(sys.modules, "lightgbm.sklearn", raising=False)
    monkeypatch.delitem(sys.modules, "xgboost", raising=False)

    create_pdl_base_model("classification", "dt")
    create_pdl_base_model("regression", "ridge")

    assert "lightgbm.sklearn" not in sys.modules
    assert "xgboost" not in sys.modules


@pytest.mark.parametrize(
    ("task", "name", "message"),
    [
        ("classification", "missing", "Unknown PDL classification base model"),
        ("regression", "missing", "Unknown PDL regression base model"),
        ("forecasting", "ridge", "Unknown PDL task"),
    ],
)
def test_unknown_pdl_base_model_requests_are_explicit(task, name, message):
    with pytest.raises(ValueError, match=message):
        create_pdl_base_model(task, name)
