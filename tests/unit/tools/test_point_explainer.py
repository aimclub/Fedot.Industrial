import math
import warnings

import pytest
from matplotlib import get_backend, pyplot as plt

from fedot_ind.api.main import FedotIndustrial as FI
from fedot_ind.tools.explain.distances import DistanceTypes
from fedot_ind.tools.explain.explain import PointExplainer
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator

distances = DistanceTypes.keys()


@pytest.fixture()
def data():
    generator = TimeSeriesDatasetsGenerator(num_samples=14,
                                            max_ts_len=50,
                                            binary=True)
    train_data, test_data = generator.generate_data()
    X_test, y_test = test_data
    X_train, y_train = train_data
    return X_train, y_train, X_test, y_test


@pytest.fixture()
def model(data):
    available_operations = ['scaling',
                            'normalization',
                            'xgboost',
                            'rfr',
                            'rf',
                            'logit',
                            'mlp',
                            'knn',
                            'lgbm',
                            'pca']

    stat_model = FI(problem='classification',
                    dataset='dataset',
                    timeout=0.1,
                    n_jobs=-1,
                    logging_level=50)
    x_train, y_train, x_test, y_test = data
    stat_model.fit((x_train, y_train))
    return stat_model, x_test, y_test


@pytest.mark.parametrize('distance, window', [(d, w) for d in distances for w in [0, 30]])
def test_explain(data, model, distance, window):
    # switch to non-Gui, preventing plots being displayed
    # suppress UserWarning that agg cannot show plots
    curr_backend = get_backend()
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", "Matplotlib is currently using agg")

    stat_model, X_test, y_test = model
    distance = distance
    explainer = PointExplainer(stat_model, X_test, y_test)
    explainer.explain(n_samples=1, window=window, method=distance)
    explainer.visual(threshold=0, name='Custom' + '_' + distance)

    ts_len = X_test.shape[1]
    expected_n_parts = math.ceil(ts_len / (window * ts_len // 100)) if window != 0 else ts_len
    assert explainer.scaled_vector.shape[0] == expected_n_parts
