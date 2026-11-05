"""Independent numerical references, including odd lengths and phase preservation."""

import numpy as np
import pywt
import pytest
from hypothesis import given, strategies as st

from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation
from fedot_ind.core.operation.transformation.basis.spectral import reconstruct_filtered_signal


@pytest.mark.parametrize('length', [31, 32, 65, 128])
@pytest.mark.parametrize('case', ['zero', 'constant', 'sine', 'impulse'])
def test_fourier_no_filter_reconstructs_original_length_and_phase(length, case):
    x = np.arange(length)
    signal = {'zero': np.zeros(length), 'constant': np.full(length, 2.),
              'sine': np.sin(2 * np.pi * x * 3 / length + .7),
              'impulse': (x == length // 3).astype(float)}[case]
    power = np.abs(np.fft.rfft(signal)) ** 2
    reconstructed = reconstruct_filtered_signal(signal, power, 0., 'smooth')
    np.testing.assert_allclose(reconstructed[0], signal, atol=1e-12)
    assert reconstructed.shape == (1, length)


def test_fourier_filter_preserves_the_selected_sinusoid_not_the_inverse_power():
    time = np.arange(128)
    slow = 3 * np.cos(2 * np.pi * 4 * time / 128 + .3)
    fast = .2 * np.sin(2 * np.pi * 27 * time / 128)
    signal = slow + fast
    power = np.abs(np.fft.rfft(signal)) ** 2
    filtered = reconstruct_filtered_signal(signal, power, .99, 'smooth')[0]
    np.testing.assert_allclose(filtered, slow, atol=1e-12)


@given(st.lists(st.floats(min_value=-100., max_value=100., allow_nan=False, allow_infinity=False), min_size=2, max_size=40),
       st.floats(min_value=0., max_value=1., allow_nan=False, allow_infinity=False))
def test_complementary_fourier_masks_reconstruct_and_cannot_increase_energy(values, threshold):
    signal = np.array(values)
    before = signal.copy()
    power = np.abs(np.fft.rfft(signal)) ** 2
    smooth = reconstruct_filtered_signal(signal, power, threshold, 'smooth')[0]
    exact = reconstruct_filtered_signal(signal, power, threshold, 'exact')[0]
    np.testing.assert_allclose(smooth + exact, signal, atol=1e-10, rtol=1e-10)
    assert np.sum(smooth ** 2) <= np.sum(signal ** 2) * (1. + 1e-12) + 1e-12
    np.testing.assert_array_equal(signal, before)


def test_fourier_public_decomposition_keeps_phase_and_length(monkeypatch):
    from types import SimpleNamespace
    signal = np.cos(2 * np.pi * np.arange(65) / 13 + .6)
    basis = FourierBasisImplementation({'threshold': 0.})
    monkeypatch.setattr(basis, '_build_spectrum', lambda _: SimpleNamespace(psd=np.ones(33)))
    np.testing.assert_allclose(basis._decompose_signal(signal)[0], signal, atol=1e-12)


@pytest.mark.parametrize('length', [31, 32, 65])
@pytest.mark.parametrize('value', [0., 2.])
def test_fourier_constant_signal_has_a_finite_exact_no_filter_result(length, value):
    signal = np.full(length, value)
    actual = FourierBasisImplementation({'threshold': 0.})._decompose_signal(signal)
    np.testing.assert_allclose(actual[0], signal, atol=1e-12)


def test_power_and_signal_outputs_are_distinct_meanings():
    values = np.sin(np.arange(65) * .3 + .7)
    power = FourierBasisImplementation({'output_format': 'spectrum'})._decompose_signal(values)
    signal = FourierBasisImplementation({'output_format': 'signal'})._decompose_signal(values)
    assert np.all(power >= 0.)
    assert signal.shape == (1, len(values))
    assert signal.min() < 0.


def test_spectrum_retry_does_not_change_the_next_samples_rank(monkeypatch):
    from types import SimpleNamespace
    basis = FourierBasisImplementation({'low_rank': 5})
    orders = []

    def estimator(values, order):
        orders.append(order)
        if order == 5:
            raise ValueError('This sample needs another rank')
        return SimpleNamespace(run=lambda: None, psd=np.ones(17))

    monkeypatch.setattr(basis, 'estimator', estimator)
    basis._build_spectrum(np.arange(32.))
    basis._build_spectrum(np.arange(32.) ** 2)
    assert orders == [5, 2, 5, 2]
    assert basis.min_rank == 5


def test_fourier_then_wavelet_is_a_time_signal_composition_not_a_spectrum_composition():
    signal = np.sin(np.arange(65) * .3 + .7)
    filtered = FourierBasisImplementation({'threshold': 0.})._decompose_signal(signal)
    wavelet = WaveletBasisImplementation({'wavelet': 'mexh', 'scale': [1, 2, 4], 'n_components': 1})
    np.testing.assert_allclose(wavelet._get_basis(filtered), wavelet._get_basis(signal), atol=1e-12)


@pytest.mark.parametrize('wavelet', ['db2', 'sym5'])
def test_discrete_wavelet_detail_and_approximation_match_reference(wavelet):
    values = np.sin(np.arange(65) * .3)
    basis = WaveletBasisImplementation({'wavelet': wavelet})
    detail, approximation = basis._decompose_signal(values)
    expected_a, expected_d = pywt.dwt(values, wavelet, mode='smooth')
    np.testing.assert_allclose(detail, expected_d)
    np.testing.assert_allclose(approximation, expected_a)
    np.testing.assert_allclose(pywt.idwt(approximation, detail, wavelet, mode='smooth')[:65], values, atol=1e-12)
    low = WaveletBasisImplementation({'wavelet': wavelet, 'low_freq': True})._get_1d_basis(values)
    np.testing.assert_allclose(low, expected_a)


def test_cwt_matches_reference_and_keeps_each_channel_separate():
    values = np.stack([np.sin(np.arange(64) * .3), np.cos(np.arange(64) * .6)])
    params = {'wavelet': 'mexh', 'n_components': 2, 'scale': [1, 2, 4, 8]}
    basis = WaveletBasisImplementation(params)
    actual = basis._get_multidim_basis(values)
    expected = []
    for channel in values:
        coefficients, _ = pywt.cwt(channel, [1, 2, 4, 8], 'mexh')
        expected.extend([coefficients[0], coefficients[1], coefficients[-1]])
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(values, np.stack([np.sin(np.arange(64) * .3), np.cos(np.arange(64) * .6)]))


def test_complex_cwt_exposes_real_and_imaginary_channels():
    values = np.sin(np.arange(64) * .3)
    basis = WaveletBasisImplementation({'wavelet': 'cmor1.5-1.0', 'scale': [1, 2, 4], 'n_components': 1})
    # Parameterized wavelet names are supported by PyWavelets, not only its short catalog names.
    actual = basis._get_1d_basis(values)
    coefficients, _ = pywt.cwt(values, [1, 2, 4], 'cmor1.5-1.0')
    expected = coefficients[[0, -1]]
    np.testing.assert_allclose(actual, np.concatenate((expected.real, expected.imag)))
    assert not np.iscomplexobj(actual)


@pytest.mark.parametrize('invalid', [np.array([0., np.nan]), np.array([0., np.inf]), np.array([1j, 2j])])
def test_nonfinite_or_complex_raw_signals_are_rejected(invalid):
    with pytest.raises(ValueError):
        reconstruct_filtered_signal(invalid, np.ones(2), 0., 'smooth')
    with pytest.raises(ValueError):
        WaveletBasisImplementation({})._get_1d_basis(invalid)


@pytest.mark.parametrize('implementation', [FourierBasisImplementation, WaveletBasisImplementation])
def test_corrected_basis_never_reads_the_shape_only_cache(monkeypatch, implementation):
    from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
    data = init_input_data(np.arange(64.).reshape(2, 1, 32), np.array([0, 1]))
    basis = implementation({})
    monkeypatch.setattr(basis.cacher, 'load_data_from_cache', lambda *args, **kwargs: pytest.fail('Unsafe cache read'))
    monkeypatch.setattr(basis, '_transform', lambda input_data: input_data.features.copy())
    result = basis.transform(data)
    np.testing.assert_array_equal(result.predict, data.features)
    with pytest.raises(ValueError, match='shape-only'):
        basis.transform(data, use_cache=True)


@pytest.mark.parametrize('implementation', [FourierBasisImplementation, WaveletBasisImplementation])
def test_explicit_unsafe_cache_parameter_is_not_silently_ignored(implementation):
    basis = implementation({'use_cache': True})
    with pytest.raises(ValueError, match='use_cache=False'):
        basis.transform(None)
