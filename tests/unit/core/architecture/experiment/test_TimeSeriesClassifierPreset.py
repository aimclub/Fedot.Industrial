import pytest
from fedot.core.pipelines.pipeline import Pipeline

from fedot_ind.core.architecture.experiment.TimeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def dataset_uni():
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=30,
                                                                       max_ts_len=50,
                                                                       binary=True,
                                                                       test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


def dataset_multi():
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=30,
                                                                       max_ts_len=50,
                                                                       binary=True,
                                                                       test_size=0.5,
                                                                       multivariate=True).generate_data()
    return X_train, y_train, X_test, y_test


@pytest.fixture
def params():
    return dict(branch_nodes=['eigen_basis'],
                dataset='custom',
                model_params={'problem': 'classification',
                              'n_jobs': -1,
                              'timeout': 0.1},
                output_folder='.',
                tuning_iterations=1,
                tuning_timeout=0.1)


@pytest.fixture
def classifier(params):
    return TimeSeriesClassifierPreset(params)


def test_init(classifier):
    assert classifier.branch_nodes == ['eigen_basis']
    assert classifier.tuning_iterations == 1
    assert classifier.tuning_timeout == 0.1
    assert isinstance(classifier.preprocessing_pipeline, Pipeline)
    assert classifier.output_folder == '.'


@pytest.mark.parametrize('dataset', [dataset_uni(), dataset_multi()])
def test_fit_predict(classifier, dataset):
    X_train, y_train, X_test, y_test = dataset
    model = classifier.fit(features=X_train, target=y_train)
    labels = classifier.predict(features=X_test, target=y_test)
    probs = classifier.predict_proba(features=X_test, target=y_test)
    metrics = classifier.get_metrics(target=y_test, metric_names=['f1', 'roc_auc'])
    for metric in metrics:
        assert metric in ['f1', 'roc_auc']

    assert len(labels) == len(y_test)
