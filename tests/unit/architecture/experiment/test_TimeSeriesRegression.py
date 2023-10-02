import pytest

from fedot_ind.core.architecture.experiment.TimeSeriesRegression import TimeSeriesRegression
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor


@pytest.fixture
def params():
    return dict(strategy='quantile',
                model_params={'problem': 'regression',
                              'timeout': 1,
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


def test_init(regressor):
    assert regressor.dataset_name == 'ApplianceEnergy'
    assert isinstance(regressor.generator_runner, QuantileExtractor)
    assert regressor.strategy == 'quantile'
    assert regressor.use_cache is True
    assert regressor.pca.n_components == 0.9
    assert regressor.pca.svd_solver == 'full'
    assert regressor.model_hyperparams['metric'] == 'rmse'
