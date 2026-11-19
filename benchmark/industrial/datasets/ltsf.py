"""Local LTSF CSV loading and causal forecast windows for DATA profiles."""

import csv
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalog import CatalogRelease
from ..protocols.schema import ProtocolError, integer, unique_strings


@dataclass(frozen=True)
class ForecastInputs:
    timestamps: tuple[str, ...]
    history: np.ndarray
    observed: np.ndarray
    future_timestamps: tuple[str, ...]
    known_future: np.ndarray


@dataclass(frozen=True)
class LTSFWindow:
    inputs: ForecastInputs
    truth: np.ndarray


@dataclass(frozen=True)
class LTSFTable:
    dates: tuple[str, ...]
    target: np.ndarray
    observed: np.ndarray
    known: np.ndarray

    def window(self, origin: int, horizon: int, *, history_start: int = 0):
        """origin is the last observed index; model inputs do not contain truth."""
        integer(origin, 'origin', minimum=0)
        integer(horizon, 'horizon')
        integer(history_start, 'history_start', minimum=0)
        if not history_start <= origin < origin + horizon < len(self.dates):
            raise ProtocolError('Window is outside the observed table')
        past = slice(history_start, origin + 1)
        ahead = slice(origin + 1, origin + 1 + horizon)
        return LTSFWindow(ForecastInputs(self.dates[past], _snapshot(self.target[past]), _snapshot(self.observed[past]),
                                         self.dates[ahead], _snapshot(self.known[ahead])), _snapshot(self.target[ahead]))


def _snapshot(values):
    # A view would retain the whole table, including future rows, through .base.
    result = values.copy()
    result.setflags(write=False)
    return result


def load_ltsf_csv(release: CatalogRelease, root, *, time='date', targets, observed=(), known_future=()):
    """Use declared columns only; preserve profile order and never fill missing values."""
    if not targets:
        raise ProtocolError('LTSF CSV requires at least one target column')
    columns = (time, *targets, *observed, *known_future)
    unique_strings(columns, 'LTSF columns')
    path = release.files(root, ('series',))['series']
    with path.open(encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle), ())
        unique_strings(header, 'LTSF CSV header')
        if not set(columns) <= set(header):
            raise ProtocolError('LTSF CSV lacks declared columns or records')
        handle.seek(0)
        table = pd.read_csv(handle)
    if not set(columns) <= set(table.columns) or table.empty:
        raise ProtocolError('LTSF CSV lacks declared columns or records')
    if pd.api.types.is_numeric_dtype(table[time]):
        raise ProtocolError('LTSF timestamps need explicit dates, not numeric epochs')
    stamps = pd.to_datetime(table[time], utc=True, errors='raise')
    if stamps.isna().any() or not stamps.is_monotonic_increasing or not stamps.is_unique:
        raise ProtocolError('LTSF timestamps must be unique and increasing')

    def values(names):
        result = table.loc[:, list(names)].to_numpy(dtype=float, copy=True)
        if np.isinf(result).any():
            raise ProtocolError('LTSF CSV contains infinite input values')
        result.setflags(write=False)
        return result

    return LTSFTable(tuple(stamp.isoformat() for stamp in stamps), values(targets), values(observed), values(known_future))
