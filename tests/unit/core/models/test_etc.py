import pytest

from fedot_ind.core.models.early_tc.prob_threshold import ProbabilityThresholdClassifier
from fedot_ind.core.models.early_tc.ecec import ECEC
from fedot_ind.core.models.early_tc.economy_k import EconomyK
from fedot_ind.core.models.early_tc.teaser import TEASER
import numpy as np

_N_FEATURES = 73
_N_SAMPLES = 133
_N_CLASSES = 3
_INTERVAL_LENGTH = 7
MODELS = {
    'economy_k': EconomyK,
    'ecec': ECEC,
    'teaser': TEASER,
    'proba_threshold_etc': ProbabilityThresholdClassifier
}


@pytest.fixture
def data():
    X, y = np.random.randn(_N_SAMPLES, _N_FEATURES), np.random.randint(0, _N_CLASSES, size=_N_SAMPLES)
    return X, y


def test_compute_prediction_points(data):
    X, y = data
    pthr = ProbabilityThresholdClassifier({'interval_percentage': 10})
    pthr._init_model(X, y)
    prediction_idx = pthr.prediction_idx
    assert len(prediction_idx) == _N_FEATURES // _INTERVAL_LENGTH, 'wrong number of points'


@pytest.mark.parametrize('training,prediction_mode,expected_num', [
    (True, 'last_available', None),
    (False, 'last_available', 1),
    (False, 'best_by_metrics_mean', 1),
    (False, 'all', None),

])
def test_select_estimators(data, training, prediction_mode, expected_num):
    X, y = data
    pthr = ProbabilityThresholdClassifier({'prediction_mode': prediction_mode})
    pthr._init_model(X, y)
    if expected_num is None:
        expected_num = pthr.n_pred
    idx, _ = pthr._select_estimators(X, training)
    assert len(idx) == expected_num, f'selection went wrong: got {len(idx)}, expected {expected_num}'


@pytest.mark.parametrize('model',
                         ['proba_threshold_etc', 'ecec', 'economy_k', 'teaser'])
def test_fit_predict(data, model):
    X, y = data
    model = MODELS[model]({'prediction_mode': 'all'})
    model.fit(X, y)
    prediction = model.predict_proba(X)
    ind = model._select_estimators(X, training=False)[0]
    assert (not np.isnan(prediction).any() and
            (prediction.shape == (2, len(ind), len(y), _N_CLASSES))), 'Prediction went wrong'

# ECEC TESTS


def test_select_thrs():
    model = ECEC()
    selection = model._select_thrs(np.random.randn(40))
    assert len(selection), 'No candidates were chosen!'

# Proba Thr


def test_consecutive(data):
    X, y = data
    pthr = ProbabilityThresholdClassifier({'prediction_mode': 'last_available',
                                           'consecutive_predictions': 1})
    pthr.fit(X, y)
    prediction, scores = pthr.predict(X)
    assert -1 not in prediction, 'Setting uncertainty while it is impossible'

# Economy K


def test_specific_economyk(data):
    X, y = data
    model = EconomyK()
    model.fit(X, y)
    assert not np.isnan(
        model._EconomyK__cluster_probas(X, model._clusterizer.cluster_centers_)
    ).any(), '__cluster_probas doesn\'t function correctly'

    i = model.n_pred - 1
    times = model._get_prediction_time(X, model._clusterizer.cluster_centers_, i)[0]
    assert not np.isnan(times).any()
    assert ((model.prediction_idx[0] <= times) & (times <= model.prediction_idx[-1])).all(), \
        f'(_get_prediction_time) case of the last prediction point:' + \
        ' times cannot exceed the limits of time predictions.' + \
        f'current lies in [{times.min()}, {times.max()}]'

# TEASER


def test_form_X_oc():
    probas = np.random.randint(0, 10, size=(_N_SAMPLES, _N_CLASSES)).astype(float)
    probas /= probas.sum(1, keepdims=True) + 1e-5
    model = TEASER()
    X_oc = model._form_X_oc(probas)
    assert X_oc.shape == (_N_SAMPLES, _N_CLASSES + 1), 'Wrong number of features'
    assert ((0 <= X_oc) & (X_oc <= 1)).all(), 'In original paper outputs lie in [0, 1]'
