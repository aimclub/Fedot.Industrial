from types import SimpleNamespace

import numpy as np
import pytest

from fedot_ind.core.operation.transformation.basis.fourier import FourierBasisImplementation


@pytest.mark.parametrize('power', [np.full(8, np.nan), np.full(8, np.inf), -np.ones(8)])
def test_invalid_power_is_not_accepted_as_a_successful_spectral_fit(monkeypatch, power):
    basis = FourierBasisImplementation({'low_rank': 5})
    orders = []

    def estimator(signal, order):
        orders.append(order)
        return SimpleNamespace(run=lambda: None, psd=power if order == 5 else np.ones(8))

    monkeypatch.setattr(basis, 'estimator', estimator)
    result = basis._build_spectrum(np.arange(16.))
    assert orders == [5, 2]
    np.testing.assert_array_equal(result.psd, np.ones(8))
    assert basis.min_rank == 5


def test_all_invalid_orders_use_the_existing_periodogram_fallback(monkeypatch):
    basis = FourierBasisImplementation({'low_rank': 5})
    monkeypatch.setattr(basis, 'estimator', lambda *args: SimpleNamespace(
        run=lambda: None, psd=np.full(8, np.nan)))
    values = np.sin(np.arange(32.) * .3)
    result = basis._decompose_signal(values)
    assert result.shape == (1, 32)
    assert np.isfinite(result).all()
