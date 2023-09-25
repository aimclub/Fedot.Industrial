import pytest
from fedot.core.pipelines.pipeline import Pipeline

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
def params():
    return dict(branch_nodes=['eigen_basis'],
                dataset='FordA',
                model_params={'task': 'classification',
                              'n_jobs': -1,
                              'timeout': 1},
                output_folder='.')


@pytest.fixture
def classifier(params):
    return TimeSeriesClassifierPreset(params)


def test_init(classifier):
    assert classifier.branch_nodes == ['eigen_basis']
    assert classifier.tuning_iters == 30
    assert classifier.tuning_timeout == 15.0
    assert isinstance(classifier.preprocessing_pipeline, Pipeline)
    assert classifier.output_folder == '.'
