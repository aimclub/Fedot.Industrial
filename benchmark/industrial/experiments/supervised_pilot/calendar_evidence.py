"""Match known TRAIN objects against a dated price release, not benchmark TEST."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd

from ...protocols.assets import read_checked_asset
from ...protocols.schema import ProtocolError


@dataclass(frozen=True)
class CalendarMatch:
    row: int
    windows: tuple[tuple[int, int], ...]


def recover_calendar(train_features, dates, prices):
    train, dates, prices = map(np.asarray, (train_features, dates, prices))
    if (train.ndim != 3 or train.shape[1:] != (1, 20) or not np.isfinite(train).all()
            or dates.shape != prices.shape or dates.ndim != 1 or len(dates) < 20
            or not np.isfinite(prices).all() or not np.all(np.diff(dates) > 0)):
        raise ProtocolError('Expected TRAIN windows and strictly increasing finite daily price history')
    lookup = {tuple(row[0]): [] for row in train}
    for begin, window in enumerate(np.lib.stride_tricks.sliding_window_view(prices, 20)):
        key = tuple(window)
        if key in lookup:
            lookup[key].append((int(dates[begin]), int(dates[begin + 19])))
    return tuple(CalendarMatch(index, tuple(lookup[tuple(row[0])])) for index, row in enumerate(train))


def require_unique_calendar(matches):
    if not matches or tuple(item.row for item in matches) != tuple(range(len(matches))):
        raise ProtocolError('Calendar recovery did not preserve source row keys')
    if any(len(item.windows) != 1 for item in matches):
        raise ProtocolError('Every TRAIN object must match one dated window exactly')
    bounds = tuple(item.windows[0] for item in matches)
    ordered = sorted(bounds)
    if any(left[1] >= right[0] for left, right in zip(ordered, ordered[1:])):
        raise ProtocolError('Matched TRAIN windows overlap')
    return tuple(begin for begin, _ in bounds)


def read_eia_calendar(root, snapshot):
    _, raw = read_checked_asset(root, snapshot)
    data = pd.read_excel(BytesIO(raw), sheet_name='Data 1', skiprows=2)
    columns = ['Date', 'Henry Hub Natural Gas Spot Price (Dollars per Million Btu)']
    if list(data.columns) != columns:
        raise ProtocolError('Unexpected Henry Hub historical release schema')
    published = data.dropna().sort_values('Date')
    dates = published['Date'].to_numpy(dtype='datetime64[D]').astype(np.int64)
    return dates, published[columns[1]].to_numpy(dtype=float)
