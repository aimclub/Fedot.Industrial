"""Pure numerical contracts for one-dimensional spectral representations."""

import numpy as np


def real_signal(values):
    signal = np.asarray(values)
    if signal.ndim != 1 or len(signal) < 2 or np.iscomplexobj(signal):
        raise ValueError('A basis transform requires a real one-dimensional signal of length >= 2')
    if not np.isfinite(signal).all():
        raise ValueError('A basis transform requires finite values')
    return signal.astype(float, copy=False)


def spectral_mask(power, threshold, approximation):
    if isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite and between zero and one')
    if approximation not in ('smooth', 'exact'):
        raise ValueError('approximation must be smooth or exact')
    values = np.asarray(power, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Power estimates must be a finite nonnegative vector')
    keep = values >= np.quantile(values, threshold)
    return ~keep if approximation == 'exact' else keep


def reconstruct_filtered_signal(signal, power, threshold, approximation, power_frequencies=None):
    """Use estimated power only to select bins; retain original complex phase.

    Spectrum estimators may have another FFT length. Interpolate power on the
    normalized one-sided frequency grid, not the mask or the complex spectrum.
    """
    values = real_signal(signal)
    coefficients = np.fft.rfft(values)
    power = np.asarray(power, dtype=float)
    spectral_mask(power, threshold, approximation)  # Validate before interpolation.
    frequencies = np.fft.rfftfreq(len(values))
    grid = (frequencies if len(power) == len(coefficients) else np.linspace(0., .5, len(power)))
    if power_frequencies is not None:
        grid = np.asarray(power_frequencies, dtype=float)
        if (grid.shape != power.shape or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0)
                or grid[0] != 0 or grid[-1] > .5 + 1e-12):
            raise ValueError('Power frequencies must be increasing normalized one-sided FFT frequencies')
    estimated = np.interp(frequencies, grid, power)
    keep = spectral_mask(estimated, threshold, approximation)
    return np.fft.irfft(np.where(keep, coefficients, 0.), n=len(values)).reshape(1, -1)
