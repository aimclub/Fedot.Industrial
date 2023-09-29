from typing import Optional, Tuple
import numpy as np
import pywt
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either
from pymonad.list import ListMonad
from fedot_ind.core.operation.transformation.basis.abstract_basis import BasisDecompositionImplementation


class WaveletBasisImplementation(BasisDecompositionImplementation):
    """Wavelet basis
        Example:
            ts1 = np.random.rand(200)
            ts2 = np.random.rand(200)
            ts = [ts1, ts2]
            bss = WaveletBasisImplementation({'n_components': 2, 'wavelet': 'mexh'})
            basis_multi = bss._transform(ts)
            basis_1d = bss._transform(ts1)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.n_components = params.get('n_components')
        self.wavelet = params.get('wavelet')
        self.basis = None
        self.discrete_wavelets = pywt.wavelist(kind='discrete')
        self.continuous_wavelets = pywt.wavelist(kind='continuous')
        self.scales = [2, 4, 10, 20]

    def _decompose_signal(self, input_data) -> Tuple[np.array, np.array]:
        if self.wavelet in self.discrete_wavelets:
            high_freq, low_freq = pywt.dwt(input_data, self.wavelet, 'smooth')
        else:
            high_freq, low_freq = pywt.cwt(data=input_data,
                                           scales=self.scales,
                                           wavelet=self.wavelet)
            low_freq = high_freq[-1, :]
            high_freq = np.delete(high_freq, (-1), axis=0)
            low_freq = low_freq[np.newaxis, :]
        return high_freq, low_freq

    def _decomposing_level(self) -> int:
        """The level of decomposition of the time series.

        Returns:
            The level of decomposition of the time series.
        """
        return pywt.dwt_max_level(len(self.time_series), self.wavelet)

    def _transform_one_sample(self, series: np.array):
        return self._get_basis(series)

    def _get_1d_basis(self, data) -> np.array:

        decompose = lambda signal: ListMonad(self._decompose_signal(signal))
        threshold = lambda Monoid: ListMonad([Monoid[0][
                                              :self.n_components],
                                              Monoid[1]])

        basis = Either.insert(data).then(decompose).then(threshold).value[0]
        basis = np.concatenate(basis)
        return basis

    def _get_multidim_basis(self, data):
        decompose = lambda multidim_signal: ListMonad(list(map(self._decompose_signal, multidim_signal)))
        select_level = lambda Monoid: [Monoid[0][
                                       :self.n_components, :],
                                       Monoid[1]]
        threshold = lambda decomposed_signal: list(map(select_level, decomposed_signal))

        basis = Either.insert(data).then(decompose).then(threshold).value
        return basis


