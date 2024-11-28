import json

import numpy as np
import pytest
from fedot.core.data.data import OutputData

from fedot_ind.api.utils.data import init_input_data
from fedot_ind.api.utils.path_lib import PATH_TO_DEFAULT_PARAMS
from fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding import RiemannExtractor
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


@pytest.fixture(scope='session')
def input_data():
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(num_samples=20,
                                                                       max_ts_len=50,
                                                                       binary=True,
                                                                       multivariate=True,
                                                                       test_size=0.5).generate_data()
    return init_input_data(X_train, y_train), init_input_data(X_test, y_test)


@pytest.fixture(scope='session')
def default_params():
    with open(PATH_TO_DEFAULT_PARAMS, 'r') as file:
        default_params = json.load(file)['riemann_extractor']
    return default_params


def test__init_riemann_extractor(default_params):
    riemann_extractor = RiemannExtractor(default_params)
    assert riemann_extractor is not None


@pytest.mark.parametrize('what_is_none', ('SPD_space', 'tangent_space', 'both'))
def test__init_spaces(default_params, what_is_none):
    if what_is_none == 'both':
        default_params['SPD_space'] = None
        default_params['tangent_space'] = None
    else:
        default_params['what_is_none'] = None

    riemann_extractor = RiemannExtractor(default_params)
    assert riemann_extractor.spd_space is not None
    assert riemann_extractor.tangent_space is not None
    assert riemann_extractor.shrinkage is not None


@pytest.mark.parametrize('fit_stage', (True, False))
def test_extract_riemann_features(input_data, default_params, fit_stage):
    riemann_extractor = RiemannExtractor(default_params)
    riemann_extractor.fit_stage = fit_stage
    train, test = input_data
    ref_point = riemann_extractor.extract_riemann_features(train)
    assert ref_point is not None
    assert isinstance(ref_point, np.ndarray)


@pytest.mark.parametrize('fit_stage', (True, False))
def test_extract_centroid_distance(input_data, default_params, fit_stage):
    riemann_extractor = RiemannExtractor(default_params)
    riemann_extractor.fit_stage = fit_stage
    train, test = input_data
    riemann_extractor.classes_ = np.unique(train.target)
    ref_point = riemann_extractor.extract_centroid_distance(train)
    assert ref_point is not None
    assert isinstance(ref_point, np.ndarray)


@pytest.mark.parametrize('extraction_strategy', ('mdm', 'tangent', 'ensemble'))
def test__init_extraction_func(default_params, extraction_strategy):
    default_params['extraction_strategy'] = extraction_strategy
    riemann_extractor = RiemannExtractor(default_params)
    assert riemann_extractor.extraction_func is not None


@pytest.mark.parametrize('fit_stage', (True, False))
def test__ensemble_features(input_data, default_params, fit_stage):
    riemann_extractor = RiemannExtractor(default_params)
    riemann_extractor.fit_stage = fit_stage
    train, test = input_data
    riemann_extractor.classes_ = np.unique(train.target)
    ref_point = riemann_extractor._ensemble_features(train)
    assert ref_point is not None
    assert isinstance(ref_point, np.ndarray)


@pytest.mark.parametrize('fit_stage', (True, False))
def test__transform(input_data, default_params, fit_stage):
    riemann_extractor = RiemannExtractor(default_params)
    riemann_extractor.fit_stage = fit_stage
    train, test = input_data
    riemann_extractor.classes_ = np.unique(train.target)
    ref_point = riemann_extractor.transform(train)
    assert ref_point is not None
    assert isinstance(ref_point, OutputData)
