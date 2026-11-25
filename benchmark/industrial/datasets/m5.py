"""M5 local inputs and the competition's twelve-level weighted RMSSE."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd

from ..protocols.schema import ProtocolError, integer


IDENTIFIERS = ('item_id', 'dept_id', 'cat_id', 'store_id', 'state_id')
HIERARCHY = ((), ('state_id',), ('store_id',), ('cat_id',), ('dept_id',),
             ('state_id', 'cat_id'), ('state_id', 'dept_id'), ('store_id', 'cat_id'),
             ('store_id', 'dept_id'), ('item_id',), ('item_id', 'state_id'), ('item_id', 'store_id'))
HORIZON = 28
FIRST_DAY = '2011-01-29'


def _columns(table, required, role):
    if not set(required) <= set(table.columns) or table.empty:
        raise ProtocolError(f'M5 {role} lacks required columns or records')


def _prices(ids, calendar, prices, days):
    weeks = calendar.loc[list(days), 'wm_yr_wk'].to_numpy()
    index = pd.MultiIndex.from_arrays([
        np.repeat(ids['store_id'].to_numpy(), len(days)),
        np.repeat(ids['item_id'].to_numpy(), len(days)),
        np.tile(weeks, len(ids)),
    ], names=['store_id', 'item_id', 'wm_yr_wk'])
    return prices.reindex(index).to_numpy(dtype=float).reshape(len(ids), len(days))


def _calendar_days(calendar):
    if 'd' in calendar and 'date' not in calendar:
        return calendar
    _columns(calendar, ('date',), 'calendar without day IDs')
    dates = pd.to_datetime(calendar['date'], format='%Y-%m-%d', errors='raise')
    offsets = (dates - pd.Timestamp(FIRST_DAY)).dt.days
    if offsets.isna().any() or set(offsets) != set(range(len(calendar))):
        raise ProtocolError('M5 calendar without day IDs must cover consecutive dates from d_1')
    days = 'd_' + (offsets + 1).astype(str)
    if 'd' in calendar and not calendar['d'].equals(days):
        raise ProtocolError('M5 day IDs do not match calendar dates')
    return calendar.assign(d=days)


def _separate_truth(table, ids, days):
    _columns(table, (*IDENTIFIERS, *days), 'targets')
    if table.duplicated(['item_id', 'store_id']).any():
        raise ProtocolError('M5 target sales contain duplicate item/store keys')
    actual = table.set_index(list(IDENTIFIERS))
    expected = pd.MultiIndex.from_frame(ids)
    if len(actual) != len(expected) or not expected.isin(actual.index).all():
        raise ProtocolError('M5 target sales do not match the declared training series')
    return actual.loc[expected, list(days)].to_numpy(dtype=float)


@dataclass(frozen=True)
class M5Window:
    row_ids: tuple[tuple[str, ...], ...]
    history: np.ndarray
    truth: np.ndarray | None
    last_dollar_sales: np.ndarray
    future_calendar: pd.DataFrame
    future_prices: np.ndarray | None
    horizon: tuple[str, ...]

    @property
    def identifiers(self):
        return pd.DataFrame.from_records(self.row_ids, columns=IDENTIFIERS)

    @property
    def series_keys(self):
        return tuple((item, store) for item, _, _, store, _ in self.row_ids)


def load_m5_window(release, root, cutoff_day, *, future_prices_known=False):
    """Load local competition tables, with optional separate targets and known future prices."""
    integer(cutoff_day, 'cutoff_day', minimum=HORIZON)
    if type(future_prices_known) is not bool:
        raise ProtocolError('future_prices_known must be explicit boolean')

    def read_table(role):
        return pd.read_csv(BytesIO(release.read_bytes(root, role)))

    sales, calendar, prices = (read_table(role) for role in ('sales', 'calendar', 'prices'))
    days = [f'd_{day}' for day in range(1, cutoff_day + 1)]
    future = tuple(f'd_{day}' for day in range(cutoff_day + 1, cutoff_day + HORIZON + 1))
    _columns(sales, (*IDENTIFIERS, *days), 'sales')
    calendar = _calendar_days(calendar)
    _columns(calendar, ('d', 'wm_yr_wk'), 'calendar')
    price_keys = ('store_id', 'item_id', 'wm_yr_wk')
    _columns(prices, (*price_keys, 'sell_price'), 'prices')
    ids = sales.loc[:, list(IDENTIFIERS)].copy()
    if (ids.isna().any().any() or ids.duplicated(['item_id', 'store_id']).any()
            or calendar['d'].duplicated().any() or prices.duplicated(list(price_keys)).any()):
        raise ProtocolError('M5 sales, calendar and price keys must be unique and defined')
    calendar = calendar.set_index('d')
    prices = prices.set_index(list(price_keys))['sell_price']
    if not set((*days[-HORIZON:], *future)) <= set(calendar.index):
        raise ProtocolError('M5 calendar does not cover the weighting and forecast windows')
    history = sales.loc[:, days].to_numpy(dtype=float)
    old_prices = _prices(ids, calendar, prices, days[-HORIZON:])
    last_sales = history[:, -HORIZON:]
    if (not np.isfinite(history).all() or np.any(history < 0)
            or np.any((~np.isfinite(old_prices) | (old_prices <= 0)) & (last_sales > 0))):
        raise ProtocolError('M5 needs finite nonnegative sales and positive prices for actual sales')
    dollar_sales = (last_sales * np.where(last_sales > 0, old_prices, 0.0)).sum(axis=1)
    future_prices = _prices(ids, calendar, prices, future) if future_prices_known else None
    if future_prices is not None and (np.isinf(future_prices).any() or np.any(future_prices <= 0)):
        raise ProtocolError('M5 known future prices must be positive or missing')
    available = set(future).intersection(sales.columns)
    if available and available != set(future):
        raise ProtocolError('M5 only accepts a complete forecast target window')
    if 'targets' in dict(release.assets):
        if available:
            raise ProtocolError('Declare either embedded or separate M5 forecast targets, not both')
        truth = _separate_truth(read_table('targets'), ids, future)
    else:
        truth = sales.loc[:, list(future)].to_numpy(dtype=float) if available else None
    row_ids = tuple(ids.itertuples(index=False, name=None))
    return M5Window(row_ids, history, truth, dollar_sales, calendar.loc[list(future)], future_prices, future)


def _aggregate(values, groups, count):
    result = np.zeros((count, *values.shape[1:]), dtype=float)
    np.add.at(result, groups, values)
    return result


def score_wrmsse(window: M5Window, forecast, *, target_days, series_keys):
    """Compute scores in groups; never invent missing scale or dollar weights."""
    protocol = 'm5_wrmsse@1'

    def not_available(reason):
        return {'protocol': protocol, 'status': 'not_available', 'reason': reason, 'wrmsse': None, 'levels': {}}

    if tuple(target_days) != window.horizon:
        raise ProtocolError('WRMSSE needs published truth on the exact forecast days')
    if tuple(series_keys) != window.series_keys:
        raise ProtocolError('WRMSSE needs forecasts in declared item/store order')
    if window.truth is None:
        return not_available('missing_truth')
    forecast = np.asarray(forecast, dtype=float)
    if (forecast.shape != window.truth.shape or not np.isfinite(forecast).all()
            or not np.isfinite(window.truth).all() or np.any(window.truth < 0)):
        raise ProtocolError('M5 forecasts and truth must be finite and aligned')
    weights = window.last_dollar_sales
    weight_total = weights.sum()
    if not np.isfinite(weight_total) or weight_total <= 0:
        return not_available('missing_positive_weights')
    identifiers = window.identifiers
    scores = {}
    for level in HIERARCHY:
        groups = (identifiers.groupby(list(level), sort=False).ngroup().to_numpy()
                  if level else np.zeros(len(identifiers), dtype=int))
        count = int(groups.max()) + 1
        weight = _aggregate(weights, groups, count)
        history = _aggregate(window.history, groups, count)
        residual = _aggregate(window.truth - forecast, groups, count)
        level_score = 0.0
        for index in np.flatnonzero(weight):
            train = history[index]
            observed = np.flatnonzero(train)
            increments = np.diff(train[observed[0]:]) if len(observed) else np.array([])
            scale = np.mean(increments ** 2) if len(increments) else 0.0
            if scale <= 0:
                return not_available(f'zero_scale: {"/".join(level) or "total"}')
            level_score += (weight[index] / weight_total) * np.sqrt(np.mean(residual[index] ** 2) / scale)
        scores['/'.join(level) or 'total'] = float(level_score)
    return {'protocol': protocol, 'status': 'success', 'wrmsse': float(np.mean(tuple(scores.values()))),
            'levels': scores}
