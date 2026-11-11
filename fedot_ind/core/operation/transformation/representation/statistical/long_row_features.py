"""Equivalent scalar statistics for audited finite float32 rows."""

import numpy as np


def benford_correlation(row):
    # Formatting identical values once keeps NumPy's scientific notation rules;
    # integer weighted counts reproduce the original ordered digit histogram.
    values, counts = np.unique(np.abs(np.nan_to_num(row)), return_counts=True)
    digits = np.array([int(str(np.format_float_scientific(value))[:1]) for value in values])
    distribution = np.array([counts[digits == digit].sum() / len(row) for digit in range(1, 10)])
    benford = np.array([np.log10(1 + 1 / digit) for digit in range(1, 10)])
    return np.corrcoef(benford, distribution)[0, 1]


def hjorth_complexity(row):
    from .native_statistics import hjorth_m4

    difference = np.diff(row)
    second = np.sum(np.power(difference, 2)) / len(difference)
    power = np.sum(np.power(row, 2)) / len(row)
    return np.sqrt((hjorth_m4(row) * power) / (second * second))
