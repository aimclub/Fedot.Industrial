import pytest
from fedot.core.data.data import InputData

from fedot_ind.core.architecture.preprocessing.data_convertor import FedotConverter
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


@pytest.fixture
def data():
    ts_generator = TimeSeriesDatasetsGenerator()
    train_data, test_data = ts_generator.generate_data()
    return train_data, test_data


def test_fedot_converter(data):
    train_data, test_data = data
    converter = FedotConverter(data=train_data)

    assert isinstance(converter.input_data, InputData)
