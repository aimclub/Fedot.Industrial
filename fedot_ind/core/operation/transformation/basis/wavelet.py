from typing import Optional, Tuple
from numbers import Integral

import dask
import pywt
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from .spectral_runtime import SpectralBasisImplementation
from fedot_ind.core.repository.constanst_repository import CONTINUOUS_WAVELETS, DISCRETE_WAVELETS, WAVELET_SCALES


class WaveletBasisImplementation(SpectralBasisImplementation):
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
        params = params if params is not None else OperationParameters()
        super().__init__(params)
        self.n_components = params.get('n_components', 2)
        self.wavelet = params.get('wavelet', 'mexh')
        self.use_low_freq = params.get('low_freq', False)
        self.scales = params.get('scale', WAVELET_SCALES)
        self.basis = None
        self.discrete_wavelets = DISCRETE_WAVELETS
        self.continuous_wavelets = CONTINUOUS_WAVELETS
        self.return_feature_vector = params.get('compute_heuristic_representation', False)
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, Integral) or self.n_components < 1:
            raise ValueError('n_components must be a positive integer')
        self.n_components = int(self.n_components)
        try:
            declared_wavelet = pywt.DiscreteContinuousWavelet(self.wavelet)
        except (TypeError, ValueError) as error:
            raise ValueError('Unknown wavelet') from error
        self.is_discrete = isinstance(declared_wavelet, pywt.Wavelet)
        if self.return_feature_vector and not self.is_discrete:
            raise ValueError('WaveletPacket features require a discrete wavelet')
        scales = np.asarray(self.scales)
        if scales.ndim != 1 or len(scales) < 2 or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('CWT scales must contain at least two finite positive values')
        if np.any(np.diff(scales) <= 0):
            raise ValueError('CWT scales must be strictly increasing')

    def __repr__(self):
        return 'WaveletBasisImplementation'

    def _compute_heuristic_features(self, input_data):
        wp = pywt.WaveletPacket(data=input_data[None, :], wavelet=self.wavelet,
                                maxlevel=3, axis=1,
                                mode='smooth')

        wpd_approximate_3 = wp['aaa'].data.sum()
        wpd_approximate_2 = wp['aa'].data.sum()
        wpd_approximate_1 = wp['a'].data.sum()
        wpd_detail_3 = wp['ddd'].data.sum()
        wpd_detail_2 = wp['dd'].data.sum()
        wpd_detail_1 = wp['d'].data.sum()
        return np.array([wpd_detail_3, wpd_detail_2, wpd_detail_1]), \
            np.array([wpd_approximate_3, wpd_approximate_2, wpd_approximate_1])

    def _decompose_signal(self, input_data) -> Tuple[np.array, np.array]:
        if self.return_feature_vector:
            return self._compute_heuristic_features(input_data)
        else:
            if self.is_discrete:
                low_freq, high_freq = pywt.dwt(input_data, self.wavelet, 'smooth', axis=-1)

            else:
                high_freq, _ = pywt.cwt(data=input_data,
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
        return pywt.dwt_max_level(np.asarray(self.time_series).shape[-1], self.wavelet)

    @dask.delayed
    def _transform_one_sample(self, series: np.array):
        return self._get_basis(series)

    def _get_1d_basis(self, data) -> np.array:
        data = np.asarray(data)
        if data.ndim == 2 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 1 or not np.isfinite(data).all() or np.iscomplexobj(data):
            raise ValueError('Wavelet basis requires a finite real signal')
        high, low = self._decompose_signal(data)
        high, low = np.atleast_2d(high), np.atleast_2d(low)
        basis = np.concatenate((high[:self.n_components], low), axis=0)
        selected = low if self.use_low_freq else basis
        # FEDOT's real-valued feature transport must not silently discard phase.
        if np.iscomplexobj(selected):
            selected = np.concatenate((selected.real, selected.imag), axis=0)
        return selected[0] if self.use_low_freq and selected.shape[0] == 1 else selected

    def _get_multidim_basis(self, data):
        values = np.asarray(data)
        if values.ndim != 2:
            raise ValueError('Multichannel wavelet input must have shape (channels, time)')
        return np.concatenate([np.atleast_2d(self._get_1d_basis(channel)) for channel in values], axis=0)
