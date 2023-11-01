import os.path

import numpy as np
import pandas as pd
import pytest
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline

from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from fedot_ind.core.models import QuantileExtractor

N_SAMPLES = 10
METRIC_LIST = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']


@pytest.fixture
def params():
    return dict(task='ts_classification',
                dataset='Ham',
                strategy='quantile',
                use_cache=False,
                timeout=0.5,
                n_jobs=-1,
                window_mode=True,
                window_size=20,
                output_folder='./results')


@pytest.fixture
def generator():
    return QuantileExtractor({'window_size': 0})


@pytest.fixture
def model_hyperparams():
    return {'problem': 'classification',
            'seed': 42,
            'timeout': 0.1,
            'max_depth': 10,
            'max_arity': 4,
            'cv_folds': 2,
            'logging_level': 20,
            'n_jobs': -1,
            'available_operations': ['scaling', 'mlp', 'pca', 'rf']}


@pytest.fixture
def classifier(params):
    return TimeSeriesClassifier(params)


@pytest.fixture
def features_n_target():
    return np.random.rand(N_SAMPLES, 20), np.random.randint(0, 3, N_SAMPLES)


def test_init(classifier):
    assert classifier.strategy == 'quantile'
    assert classifier.dataset_name == 'Ham'
    assert classifier.logger.name == 'TimeSeriesClassifier'

    none_cls_attrs = [_ for _ in classifier.__dict__.keys() if _ not in ['strategy', 'dataset_name',
                                                                         'logger', 'saver', 'datacheck','output_folder']]
    cls_attrs = ['strategy', 'dataset_name', 'logger',
                 'saver', 'datacheck', 'output_folder']

    for attr in none_cls_attrs:
        assert classifier.__dict__[attr] is None
    for attr in cls_attrs:
        assert classifier.__dict__[attr] is not None


def test_fit(classifier, generator, model_hyperparams, features_n_target):
    features, target = features_n_target
    features = pd.DataFrame(features)
    classifier.generator_runner = generator
    classifier.model_hyperparams = model_hyperparams
    model = classifier.fit(features, target)

    model.current_pipeline.save('./ppl.json')

    assert isinstance(model, Fedot)
    assert classifier.train_features is not None
    assert classifier.train_features.shape[0] == N_SAMPLES
    assert classifier.predictor is model


def test_fit_model(classifier, model_hyperparams, features_n_target):
    features, target = features_n_target
    features = pd.DataFrame(features)
    classifier.model_hyperparams = model_hyperparams
    model = classifier._fit_baseline_model(features, target)
    assert isinstance(model, Pipeline)


def test_fit_baseline_model(classifier, features_n_target):
    features, target = features_n_target
    features = pd.DataFrame(features)
    model = classifier._fit_baseline_model(features, target)
    assert isinstance(model, Pipeline)


def test_predict(classifier, generator, model_hyperparams, features_n_target):
    classifier.generator_runner = generator
    classifier.predictor = Pipeline().load('./ppl.json')
    features, target = features_n_target
    features = pd.DataFrame(features)
    labels = classifier.predict(features=features, target=target)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == N_SAMPLES


def test_predict_proba(classifier, generator, model_hyperparams, features_n_target):
    classifier.generator_runner = generator
    classifier.predictor = Pipeline().load('./ppl.json')
    features, target = features_n_target
    features = pd.DataFrame(features)
    proba = classifier.predict_proba(features=features, target=target)
    assert isinstance(proba, np.ndarray)
    assert proba.shape[0] == N_SAMPLES


@pytest.mark.parametrize('mode', ['labels', 'probs'])
def test_predict_abstraction(classifier, mode, features_n_target, generator):
    classifier.generator_runner = generator
    classifier.predictor = Pipeline().load('./ppl.json')
    features, target = features_n_target
    features = pd.DataFrame(features)
    prediction = classifier._predict_abstraction(test_features=features,
                                                 target=target,
                                                 mode=mode)
    if mode == 'probs':
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape[0] == N_SAMPLES
    else:
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == N_SAMPLES


def test_get_metrics(classifier):
    target = np.array([1, 1, 2, 2]).reshape(-1, 1)
    classifier.prediction_label = np.array([1, 2, 1, 2]).reshape(-1, 1)
    classifier.prediction_proba = np.array([[0.5, 0.3], [0.3, 0.5], [0.5, 0.3], [0.3, 0.5]])
    metrics = classifier.get_metrics(target=target,
                                     metric_names=METRIC_LIST)

    assert np.all([_ in METRIC_LIST for _ in metrics.keys()])
    assert np.all([isinstance(value, float) for value in metrics.values()])


@pytest.mark.parametrize('kind', ('labels', 'probs'))
def test_save_prediction(classifier, kind):
    prediction_label = np.array([1, 2, 1, 2]).reshape(-1, 1)
    prediction_proba = np.array([[0.5, 0.3], [0.3, 0.5], [0.5, 0.3], [0.3, 0.5]])
    if kind == 'labels':
        classifier.save_prediction(predicted_data=prediction_label, kind=kind)
    else:
        classifier.save_prediction(predicted_data=prediction_proba, kind=kind)
    expected_file_path = os.path.join(classifier.saver.path, f'{kind}.csv')
    assert os.path.isfile(expected_file_path)


def test_save_metrics(classifier):
    classifier.save_metrics(metrics=dict(f1=0.5, roc_auc=0.4))
    expected_file_path = os.path.join(classifier.saver.path, 'metrics.csv')
    assert os.path.isfile(expected_file_path)


