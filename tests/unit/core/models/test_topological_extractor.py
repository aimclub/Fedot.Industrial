import pytest
from fedot.core.data.data import InputData, OutputData

from fedot_ind.api.utils.data import init_input_data
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.representation.topological import TopologicalExtractor
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def dataset(binary):
    (X_train, y_train), (X_test, y_test) = TimeSeriesDatasetsGenerator(
        num_samples=20, max_ts_len=50, binary=True, test_size=0.5).generate_data()
    return X_train, y_train, X_test, y_test


@pytest.fixture
def input_data():
    N_CLASSES = np.random.choice([2, 3])
    X_train, y_train, X_test, y_test = dataset(N_CLASSES)
    input_train_data = init_input_data(X_train, y_train)
    return input_train_data


@pytest.fixture
def topological_extractor():
    return TopologicalExtractor({'window_size': 50})


def test_transform(topological_extractor, input_data):
    train_features = topological_extractor.transform(input_data=input_data)
    assert train_features is not None
    assert isinstance(train_features, OutputData)


def test_generate_topological_features(topological_extractor, input_data):
    sample = input_data.features[0]
    train_features = topological_extractor.generate_topological_features(
        sample)
    assert train_features is not None
    assert isinstance(train_features, InputData)
    assert train_features.features.shape[0] == 1
