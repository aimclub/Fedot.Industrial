import numpy as np
import pytest

fedot_data = pytest.importorskip('fedot.core.data.data')
InputData = fedot_data.InputData
OutputData = fedot_data.OutputData
OperationParameters = pytest.importorskip('fedot.core.operations.operation_parameters').OperationParameters
DataTypesEnum = pytest.importorskip('fedot.core.repository.dataset_types').DataTypesEnum

from fedot_ind.core.operation.interfaces.detection_runtime_strategy import (
    IndustrialDetectionModelRuntimeStrategy,
    build_detection_boundary_batch,
    _invoke_detection_prediction
)

from fedot_ind.core.models.detection.runtime import DetectionWindowBatch


class SimplePredictModel:
    def __init__(self, params=None): self.params = params
    def fit(self, data): pass
    def predict(self, data, output_mode='default'): 
        return np.zeros(len(data.features))


class ScorerModel:
    def __init__(self, params=None): self.params = params
    def fit(self, data): pass
    def predict(self, data, output_mode='default'): 
        return np.zeros(len(data.features))
    def score_samples(self, data):
        return np.ones(len(data.features)) * 0.5


def _build_detection_input(length: int = 50) -> InputData:
    features = np.random.rand(length, 2)
    return InputData(
        idx=np.arange(length),
        features=features,
        target=None,
        task=None,
        data_type=DataTypesEnum.table,
    )


def test_build_detection_boundary_batch_logic():
    input_data = _build_detection_input(length=100)
    params = OperationParameters(window_size=20, stride=5)
    batch = build_detection_boundary_batch(input_data, params)
    
    assert batch.window_size == 20
    assert batch.stride == 5


def test_invoke_prediction_calibration_contract():
    model = ScorerModel()
    input_data = _build_detection_input()
 
    pred_scores = _invoke_detection_prediction(model, input_data, output_mode='scores')
    assert np.all(pred_scores == 0.5)

    pred_default = _invoke_detection_prediction(model, input_data, output_mode='default')
    assert np.all(pred_default == 0.5) 

    simple_model = SimplePredictModel()
    pred_simple = _invoke_detection_prediction(simple_model, input_data, output_mode='default')
    assert np.all(pred_simple == 0)


def test_strategy_predict_format():
    strategy = IndustrialDetectionModelRuntimeStrategy('iforest_detector')
    strategy.operation_impl = lambda params: ScorerModel(params)
    
    input_data = _build_detection_input()
    trained = strategy.fit(input_data)
    
    output = strategy.predict(trained, input_data)
    
    assert isinstance(output, OutputData)
    assert output.data_type is DataTypesEnum.table
    assert np.all(output.predict == 0.5)