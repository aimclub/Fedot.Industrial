"""Select versioned forecasting inputs without accessing test or assessment values during search."""

import csv
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from io import StringIO
from pathlib import PurePosixPath

import numpy as np

from ...evaluation.forecasting_protocols import reference_mase, reference_smape, vector
from ...protocols.assets import LocalAsset, local_root, read_checked_asset
from ...protocols.json_io import parse_protocol_json
from ...protocols.schema import ProtocolError


VERSION = 'tsf_extension@1'


@dataclass(frozen=True)
class Collection:
    id: str
    filename: str
    horizon: int
    lag: int
    sha256: str
    source: str
    layout: str = 'rows'
    control_lags: tuple[int, ...] = ()

    @property
    def min_history(self):
        return 7 * self.horizon + 48 + self.horizon


@dataclass(frozen=True)
class TrainRelease:
    asset: LocalAsset
    origin: str
    horizon: int


def load_profile():
    raw = files(__package__).joinpath('extension.json').read_text(encoding='utf-8')
    profile = parse_protocol_json(raw)
    if profile['version'] != VERSION:
        raise ProtocolError('Unsupported forecasting extension manifest')
    return tuple(Collection(**item) for item in profile['collections']), tuple(profile['seeds'])


def _tourism_columns(rows):
    columns = tuple(zip(*rows, strict=True))
    result = {}
    for column in columns:
        series_id, count, year, month, *values = column
        if series_id in result or not series_id or not count.isdecimal() or not year.isdecimal() or not month.isdecimal():
            raise ProtocolError('Invalid authors\' Tourism TRAIN column header')
        n = int(count)
        if n < 1 or not 1 <= int(month) <= 12 or any(value.strip() for value in values[n:]):
            raise ProtocolError('Tourism TRAIN observations do not match the header')
        result[series_id] = vector(values[:n], series_id)
    return result


def load_train_csv(root, collection, release: TrainRelease):
    if (PurePosixPath(release.asset.path).name != collection.filename or release.horizon != collection.horizon
            or not release.origin or not release.origin.strip()):
        raise ProtocolError(f'Declare a separate, verified TRAIN release for {collection.id}')
    root = local_root(root)
    path = root / release.asset.path
    if path.resolve() != path:
        raise ProtocolError('Declared TRAIN release must not be redirected to a different file')
    _, payload = read_checked_asset(root, release.asset)
    rows = csv.reader(StringIO(payload.decode('utf-8-sig'), newline=''))
    if collection.layout == 'tourism_columns':
        return _tourism_columns(rows)
    header = next(rows, ())
    if not header or len(header) < 2 or header[0] not in ('V1', 'series_id'):
        raise ProtocolError('Expected an official or normalized TRAIN table with series identifiers')
    result = {}
    for row in rows:
        if not row or not row[0] or row[0] in result or len(row) > len(header):
            raise ProtocolError('Missing or duplicate TRAIN series identifier')
        fields = row[1:]
        while fields and not fields[-1].strip():
            fields.pop()
        if any(not value.strip() for value in fields):
            raise ProtocolError('Interior missing values are outside the extension protocol')
        result[row[0]] = vector(fields, f'{collection.id}/{row[0]}')
    return result


def identity(series):
    return sha256(np.asarray(vector(series, 'TRAIN series'), dtype='<f8').tobytes()).hexdigest()


def select_series(collection, rows, excluded_ids=(), excluded_fingerprints=()):
    blocked_ids, blocked_values = set(excluded_ids), set(excluded_fingerprints)
    candidates, exclusion = [], {}
    for series_id, series in rows.items():
        values = vector(series, series_id)
        signature = identity(values)
        if series_id in blocked_ids or signature in blocked_values:
            exclusion[series_id] = 'previous_series'
        elif len(values) < collection.min_history:
            exclusion[series_id] = 'insufficient_history'
        else:
            candidates.append((series_id, len(values), signature))
    candidates.sort(key=lambda row: (row[1], row[0]))
    unique = []
    for series_id, length, signature in candidates:
        if signature in blocked_values:
            exclusion[series_id] = 'duplicate_series'
        else:
            unique.append((series_id, length, signature))
            blocked_values.add(signature)
    pivot = len(unique) // 2
    lower, upper = unique[:pivot], unique[pivot:]
    if min(len(lower), len(upper)) < 2:
        raise ProtocolError(f'Insufficient distinct, eligible series for {collection.id}')

    def priority(row):
        return sha256(f'{VERSION}/{collection.id}/{row[0]}'.encode()).hexdigest(), row[0]

    chosen = tuple(row[0] for half in (lower, upper) for row in sorted(half, key=priority)[:2])
    return chosen, exclusion


def origins(length, collection):
    if length < collection.min_history:
        raise ProtocolError('Series cannot support the five non-overlapping window horizons')
    h = collection.horizon
    starts = tuple(length - i * h for i in range(5, 0, -1))
    return starts[:2], starts[2:]


def setup_search(series, collection):
    train = vector(series, 'TRAIN series')
    search, assessment = origins(len(train), collection)
    prefix = train[:assessment[0]].copy()
    return prefix, search


def score_window(history, actual, predicted, collection):
    history, actual, predicted = (vector(value, field) for value, field in zip(
        (history, actual, predicted), ('available history', 'actual', 'forecast'), strict=True))
    if len(actual) != collection.horizon or actual.shape != predicted.shape:
        raise ProtocolError('Forecast and actual must match the locked horizon')
    if len(history) <= collection.lag:
        raise ProtocolError('Insufficient history for the declared MASE lag')
    nonconstant = np.any(history[collection.lag:] != history[:-collection.lag])
    return dict(
        mase=reference_mase(history, actual, predicted, collection.lag) if nonconstant else None,
        mase_status='available' if nonconstant else 'zero_scale',
        rmse=float(np.sqrt(np.mean((actual - predicted) ** 2))),
        smape=reference_smape(actual, predicted),
    )
