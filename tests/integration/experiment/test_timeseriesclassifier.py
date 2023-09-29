import numpy as np
import pytest
from fedot.core.pipelines.pipeline import Pipeline

from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor


@pytest.fixture
def params():
    return dict(dataset='custom',
                strategy='quantile',
                use_cache=False,
                window_mode=False,
                model_params={'problem': 'classification',
                              'timeout': 1,
                              'n_jobs': -1})


@pytest.fixture
def features_extractor():
    return QuantileExtractor({'window_mode': False})


@pytest.fixture
def classifier(params):
    return TimeSeriesClassifier(params)


@pytest.fixture
def dataset():
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=30,
                                                                       max_ts_len=50,
                                                                       n_classes=np.random.choice([2,3]),
                                                                       test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


def test_fit_predict(classifier, features_extractor, dataset):
    X_train, y_train, X_test, y_test = dataset
    classifier.generator_runner = features_extractor
    pipeline = classifier.fit(features=X_train, target=y_train)
    labels = classifier.predict(features=X_test, target=y_test)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(y_test)
    assert isinstance(pipeline, Pipeline)
