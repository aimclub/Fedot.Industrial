import pytest

from fedot_ind.core.architecture.experiment.TimeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


@pytest.fixture
def dataset():
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=30,
                                                                       max_ts_len=50,
                                                                       n_classes=2,
                                                                       test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


@pytest.fixture
def classifier_for_fit():
    params = dict(branch_nodes=['eigen_basis'],
                  dataset='dataset',
                  tuning_iters=2,
                  tuning_timeout=2,
                  model_params={'problem': 'classification',
                                'n_jobs': -1,
                                'timeout': 1},
                  output_folder='.')
    return TimeSeriesClassifierPreset(params)


def test_fit_predict(classifier_for_fit, dataset):
    X_train, y_train, X_test, y_test = dataset
    classifier_for_fit.fit(features=X_train, target=y_train)
    labels = classifier_for_fit.predict(features=X_test, target=y_test)
    assert len(labels) == len(y_test)
    assert classifier_for_fit.preprocessing_pipeline.is_fitted is True
    assert classifier_for_fit.predictor.current_pipeline.is_fitted is True
