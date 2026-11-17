"""Causal, undecimated Haar decomposition on the original time grid."""

from dataclasses import dataclass

import numpy as np

from .grid import positive_count


@dataclass(frozen=True, eq=False)
class CausalHaarBands:
    """Rows start at `start_index`; columns are approximation, then fine-to-coarse details."""

    values: np.ndarray
    start_index: int


def _finite_series(history) -> np.ndarray:
    values = np.asarray(history)
    if values.ndim != 1 or not values.size or values.dtype.kind not in "iuf":
        raise ValueError("Expected a nonempty, real-valued series.")
    values = values.astype(float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("The series must contain only finite values.")
    return values


def causal_haar(history, levels: int) -> CausalHaarBands:
    """Use only present and past samples; discard the left prefix without complete support."""
    series = _finite_series(history)
    levels = positive_count(levels, "levels")
    if levels >= int(series.size).bit_length():
        raise ValueError("Causal Haar needs at least 2 ** levels observations.")

    approximation, details = series, []
    for level in range(levels):
        lag = 1 << level
        current = approximation[lag:]
        approximation = .5 * current + .5 * approximation[:-lag]
        details.append(current - approximation)

    count = len(approximation)
    values = np.column_stack([approximation, *(detail[-count:] for detail in details)])
    values.setflags(write=False)
    return CausalHaarBands(values, len(series) - count)
