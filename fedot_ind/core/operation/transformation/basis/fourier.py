from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation


class FourierBasisImplementation(BasisDecompositionImplementation):
    """A class for decomposing data on the Fourier basis and evaluating the derivative of the resulting decomposition.

    Example::
        ts1 = np.random.rand(200)
        ts2 = np.random.rand(200)
        ts = [ts1, ts2]
        bss = FourierBasisImplementation({'threshold': 20000'})
        basis_multi = bss.transform(ts)
        basis_1d = bss.transform(ts1)

    """

    def __repr__(self):
        return 'FourierBasisImplementation'

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.threshold = params.get('threshold')
        self.approximation = params.get('approximation', 'smooth')
        self.basis = None

        self.logging_params.update({'threshold': self.threshold})

    def _decompose_signal(self, input_data):
        fourier_coef = np.fft.rfft(input_data)
        frequencies = np.fft.rfftfreq(
            input_data.size, d=2e-3 / input_data.size)

        if self.threshold > frequencies[-1]:
            median_freq = round(len(frequencies) / 2)
            self.threshold = frequencies[median_freq]
        ind_of_main_freq = np.where(frequencies >= self.threshold)[0][:1]
        ind_of_main_freq = tuple(ind_of_main_freq)

        if self.approximation == 'exact':
            fourier_coef[frequencies != frequencies[ind_of_main_freq]] = 0
        else:
            fourier_coef[frequencies > frequencies[ind_of_main_freq]] = 0
        return np.fft.irfft(fourier_coef).reshape(1, -1)

    def _transform_one_sample(self, series: np.array):
        return self._get_basis(series)
