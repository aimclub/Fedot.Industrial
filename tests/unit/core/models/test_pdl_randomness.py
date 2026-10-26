"""The public PDL seed must reach the actual downstream estimator."""

from copy import deepcopy
import json

import numpy as np
import pytest
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.models.pdl.config import PairwiseLearningConfig
from fedot_ind.core.models.pdl.pairwise_model import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor
from fedot_ind.core.models.pdl.base_models import create_pdl_base_model


@pytest.mark.parametrize('estimator,name', [
    (PairwiseDifferenceClassifier, 'dt'), (PairwiseDifferenceRegressor, 'dtreg')])
@pytest.mark.parametrize('operation_parameters', [False, True])
@pytest.mark.parametrize('seed', ['omitted', 42, 123, 0, None])
def test_public_seed_reaches_base_estimator_without_mutating_inputs(estimator, name, operation_parameters, seed):
    raw = {'model': name, 'max_pairs': 512, 'max_depth': 2}
    if seed != 'omitted':
        raw['random_state'] = seed
    original = deepcopy(raw)
    params = OperationParameters(**raw) if operation_parameters else raw
    model = estimator(params)
    expected = 42 if seed == 'omitted' else seed
    assert model.config.random_state == expected
    assert model.base_model.random_state == expected
    assert model.base_model.max_depth == 2
    assert model.base_model_params == {'max_depth': 2}
    assert raw == original
    if operation_parameters:
        assert params.to_dict() == original


@pytest.mark.parametrize('estimator,name', [
    (PairwiseDifferenceClassifier, 'rf'), (PairwiseDifferenceRegressor, 'treg')])
def test_seeded_pdl_fits_independently_of_ambient_numpy_rng(estimator, name):
    features = np.column_stack([np.arange(18), np.arange(18) % 4, np.arange(18)]) / 18
    target = np.arange(18) % 3 if name == 'rf' else np.sin(np.arange(18))
    ambient = np.random.get_state()
    try:
        predictions = []
        for ambient_seed in (11, 989):
            np.random.seed(ambient_seed)
            np.random.random(70)
            model = estimator({'model': name, 'random_state': 0, 'n_estimators': 3,
                               'max_pairs': 512, 'backend': 'numpy'})
            model.fit(features, target)
            predictions.append(model.predict(features + .01))
            diagnostic = model.get_diagnostics()
            assert diagnostic['resolved_base_model_params']['random_state'] == 0
            assert diagnostic['base_model_class'].endswith(type(model.base_model).__name__)
            assert diagnostic['base_model_params'] == {'n_estimators': 3}
            json.dumps(diagnostic, allow_nan=False)
        np.testing.assert_array_equal(*predictions)
    finally:
        np.random.set_state(ambient)


@pytest.mark.parametrize('seed', [True, False, -1, 2**32, 1.5, '42'])
def test_pdl_seed_rejects_ambiguous_or_out_of_range_values(seed):
    with pytest.raises(ValueError, match='random_state'):
        PairwiseLearningConfig(random_state=seed).normalized()


def test_numpy_integer_seed_normalizes_to_json_compatible_integer():
    config = PairwiseLearningConfig(random_state=np.int64(123)).normalized()
    assert type(config.random_state) is int
    assert config.random_state == 123
    json.dumps(config.to_dict())


def test_direct_factory_preserves_existing_seed_and_rejects_conflicting_context():
    assert create_pdl_base_model('regression', 'dtreg').random_state is None
    assert create_pdl_base_model('regression', 'dtreg', {'random_state': 7}).random_state == 7
    assert create_pdl_base_model('regression', 'dtreg', random_state=0).random_state == 0
    with pytest.raises(ValueError, match='Conflicting'):
        create_pdl_base_model('regression', 'dtreg', {'random_state': 7}, random_state=42)


@pytest.mark.parametrize('task,name', [('classification', 'catboost'), ('regression', 'catboostreg')])
def test_seeded_array_contract_does_not_silently_accept_inputdata_only_wrappers(task, name):
    with pytest.raises(ValueError, match='seeded array estimator contract'):
        create_pdl_base_model(task, name, random_state=42)


def test_existing_xgboost_classifier_alias_is_not_reinterpreted():
    model = PairwiseDifferenceClassifier({'model': 'xgboost', 'random_state': 123})
    assert type(model.base_model).__name__ == 'GradientBoostingClassifier'
    assert model.base_model.random_state == 123
