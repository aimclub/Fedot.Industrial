from typing import Optional

import pandas as pd
from fedot.core.operations.operation_parameters import OperationParameters
from matplotlib import pyplot as plt

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation
from fedot_ind.core.repository.constanst_repository import SPECTRUM_ESTIMATORS


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
        self.threshold = params.get('threshold', 0.9)
        self.sampling_rate = params.get('sampling_rate', 4096)
        self.output_format = params.get('output_format', 'signal')
        self.approximation = params.get('approximation', 'smooth')
        self.min_rank = params.get('low_rank', 5)

        self.estimator = SPECTRUM_ESTIMATORS[params.get('estimator', 'eigen')]
        self.basis = None
        self.filtred_signal = None

        self.logging_params.update({'threshold': self.threshold})

    def _visualise_spectrum(self, estimator):
        import matplotlib
        matplotlib.use('TkAgg')
        if isinstance(estimator, np.ndarray):
            pd.DataFrame(estimator).T.plot()
            plt.show()
        else:
            estimator.plot(marker='o')
            plt.show()
        return

    def _build_spectrum(self, input_data):
        estimator = self.estimator(input_data, self.min_rank)
        estimator.run()
        return estimator

    def _decompose_signal(self, input_data):
        estimator = self._build_spectrum(input_data)
        # self._visualise_spectrum(estimator)
        psd = estimator.psd
        dominant_freq = np.where(psd >= np.quantile(psd, q=self.threshold))[0]
        if self.approximation == 'exact':
            psd[dominant_freq] = 0
        else:
            psd[~dominant_freq] = 0
        self.filtred_signal = psd if self.output_format == 'spectrum' else np.fft.irfft(psd).reshape(1, -1)
        return self.filtred_signal

    def _transform_one_sample(self, series: np.array):
        return self._get_basis(series)
