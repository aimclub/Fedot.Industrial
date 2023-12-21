import os

import numpy as np
import pytest

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.architecture.experiment.TimeSeriesRegression import TimeSeriesRegression
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.tools.loader import DataLoader

@pytest.fixture
def params():
    return dict(strategy='quantile',
                model_params={'problem': 'regression',
                              'timeout': 0.5,
                              'n_jobs': 2,
                              'metric': 'rmse'},
                generator_class=QuantileExtractor({'window_mode': True, 'window_size': 20}),
                use_cache=True,
                dataset='ApplianceEnergy',
                output_folder='.',
                explained_variance=0.9,)


@pytest.fixture
def regressor(params):
    return TimeSeriesRegression(params)


@pytest.fixture()
def dataset():
    path = os.path.join(PROJECT_PATH, 'examples/data/')
    loader = DataLoader(dataset_name='BitcoinSentiment',
                        folder=path)
    return loader.load_data()


def test_init(regressor):
    assert regressor.dataset_name == 'ApplianceEnergy'
    assert isinstance(regressor.generator_runner, QuantileExtractor)
    assert regressor.strategy == 'quantile'
    assert regressor.use_cache is True
    assert regressor.pca.n_components == 0.9
    assert regressor.pca.svd_solver == 'full'
    assert regressor.model_hyperparams['metric'] == 'rmse'


# def test_fit_predict(regressor, dataset):
#     (X_train, y_train), (X_test, y_test) = dataset
#     regressor.fit(X_train, y_train)
#     predict = regressor.predict(X_test, y_test)
#     metrics = regressor.get_metrics(target=y_test, metric_names=['rmse', 'mae', 'r2'])
#
#     assert isinstance(predict, np.ndarray)
#     assert isinstance(metrics, dict)
