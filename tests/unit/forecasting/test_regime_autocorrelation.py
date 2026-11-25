import numpy as np
import pytest

from industrial_forecasting.regimes.diagnostics import _autocorrelations, analyze_regime_diagnostics


@pytest.mark.parametrize('series', [
    np.arange(8, dtype=float),
    np.full(31, 7.),
    np.tile([0., 1., -2., 3., 1.], 31),
    np.random.default_rng(18).normal(size=8193),
])
def test_regime_autocorrelations_match_direct_lags(series):
    centered = series - np.mean(series)
    energy = float(np.sum(centered ** 2))
    period_stop = max(3, len(series) // 3)
    stop = max(period_stop, 6)
    expected = np.array([
        np.sum(centered[:len(series) - lag] * centered[lag:]) / energy
        if energy > 1e-12 else 0.
        for lag in range(stop)
    ])
    np.testing.assert_allclose(_autocorrelations(centered, energy, stop), expected, rtol=0., atol=1e-12)

    result = analyze_regime_diagnostics(series, window_size=9)
    decay_stop = min(6, len(series) // 2)
    assert result.acf_decay_rate == pytest.approx(np.mean(np.abs(expected[1:decay_stop])), abs=1e-12)
    best = int(np.argmax(expected[2:period_stop])) + 2
    assert result.dominant_period == (best if expected[best] >= .2 else None)


@pytest.mark.parametrize('extra_energy', [0., np.sqrt(.5), np.sqrt(.5) + 1e-8])
def test_regime_period_keeps_the_earliest_tie_and_the_original_threshold(extra_energy):
    series = np.zeros(30)
    series[[1, 3, 14, 17, 27, 28]] = [1., 1., -1., -1., extra_energy, -extra_energy]
    centered = series - series.mean()
    scores = np.correlate(centered, centered, mode='full')[len(series) - 1:] / np.dot(centered, centered)
    assert scores[2] == scores[3] == max(scores[2:10])
    assert analyze_regime_diagnostics(series).dominant_period == (2 if scores[2] >= .2 else None)
