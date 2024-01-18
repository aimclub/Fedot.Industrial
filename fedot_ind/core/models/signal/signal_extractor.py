from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.metrics.metrics_implementation import *
from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation


class SignalExtractor(BaseExtractor):
    """Class responsible for wavelet feature generator experiment.

    Attributes:
        wavelet_basis (WaveletBasisImplementation): class to transform time series
        wavelet (str): current wavelet type
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.wavelet_basis = WaveletBasisImplementation
        self.n_components = params.get('n_components')
        self.wavelet = params.get('wavelet')

    def _transform(self, input_data: InputData) -> np.array:
        wavelet_basis = self.wavelet_basis({'n_components': self.n_components,
                                            'wavelet': self.wavelet})
        transformed_features = wavelet_basis.transform(input_data)
        predict = self._clean_predict(transformed_features.predict)
        return predict
