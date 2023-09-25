import pytest

from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier


@pytest.fixture
def params():
    return dict(task='ts_classification',
                dataset='Ham',
                strategy='quantile',
                use_cache=False,
                timeout=1,
                n_jobs=-1,
                window_mode=True,
                window_size=20)


@pytest.fixture
def classifier(params):
    return TimeSeriesClassifier(params)


def test_init(classifier):
    assert classifier.strategy == 'quantile'
    assert classifier.model_hyperparams is None
    assert classifier.generator_runner is None
    assert classifier.dataset_name == 'Ham'
    assert classifier.output_folder is None
    assert classifier.saver is not None
    assert classifier.logger is not None
    assert classifier.datacheck is not None
    assert classifier.prediction_proba is None
    assert classifier.test_predict_hash is None
    assert classifier.prediction_label is None
    assert classifier.predictor is None
    assert classifier.y_train is None
    assert classifier.train_features is None
    assert classifier.test_features is None
    assert classifier.input_test_data is None
    assert classifier.logger.name == 'TimeSeriesClassifier'
