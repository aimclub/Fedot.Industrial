"""Bounded row batching of the NumPy Hurst formula, preserving prefix reductions."""

import json
from importlib.util import find_spec
from pathlib import Path

import numpy as np


_LIMITS = json.loads(Path(__file__).with_name('batch_defaults.json').read_text('utf-8'))


def native_hurst_supported(values):
    return (values.dtype == np.float32 and values.shape[-1] >= _LIMITS['native_min_length']
            and np.__version__ == '1.26.4' and np.getbufsize() == 8192 and find_spec('numba') is not None)


def hurst_rows(rows):
    """Match scalar hurst_exponent(axis=0); never pool observations or change dtype."""
    if native_hurst_supported(rows):
        return _native_rows(rows)
    length = rows.shape[-1]
    times = np.arange(1, length + 1)
    logarithms = np.log(times)[1:]
    design = np.column_stack((logarithms, np.ones(logarithms.size)))
    result = np.empty(len(rows))
    batch_size = max(1, min(_LIMITS['max_rows'], _LIMITS['max_elements'] // length))
    for start in range(0, len(rows), batch_size):
        values = rows[start:start + batch_size]
        cumulative = np.cumsum(values, axis=-1)
        averages = cumulative / times
        deviation = np.zeros(values.shape)
        ranges = np.zeros(values.shape)
        for index in range(length):
            # Keep the scalar two-pass std, rather than an unstable E[x^2]-E[x]^2.
            deviation[:, index] = np.std(values[:, :index + 1], axis=-1)
            centered = cumulative[:, :index + 1] - times[:index + 1] * averages[:, index, None]
            ranges[:, index] = np.ptp(centered, axis=-1)
        responses = np.log(ranges / deviation)[:, 1:]
        for offset, response in enumerate(responses):
            # A multiple-RHS solve can round differently from the scalar reference.
            result[start + offset] = np.linalg.lstsq(design, response, rcond=None)[0][0]
    return result


def _native_rows(rows):
    from .native_statistics import prefixes
    from .stat_features import hurst_exponent

    times = np.arange(1, rows.shape[-1] + 1)
    logarithms = np.log(times)[1:]
    design = np.column_stack((logarithms, np.ones(logarithms.size)))
    result = np.empty(len(rows))
    for index, row in enumerate(rows):
        cumulative = np.cumsum(row)
        averages = cumulative / times
        if not np.isfinite(cumulative).all():
            result[index] = hurst_exponent(row, 0)
            continue
        deviations, ranges = prefixes(row, cumulative, averages)
        response = np.log(ranges / deviations)[1:]
        result[index] = np.linalg.lstsq(design, response, rcond=None)[0][0]
    return result
