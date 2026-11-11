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


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_every_advertised_pdl_model_supports_default_seeded_construction(task):
    from fedot_ind.core.models.pdl.pairwise_model import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor

    estimator = PairwiseDifferenceClassifier if task == 'classification' else PairwiseDifferenceRegressor
    names = available_pdl_base_models(task)
    assert names
    assert 'catboost' not in names
    assert 'catboostreg' not in names
    for name in names:
        model = estimator({'model': name})
        assert model.config.random_state == 42
        assert model.base_model.get_params(deep=False)['random_state'] == 42
        assert callable(model.base_model.fit)
        assert callable(model.base_model.predict)


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
