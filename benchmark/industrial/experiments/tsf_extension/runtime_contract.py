"""Bind forecast work to the original admitted executor and one sealed case."""

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np

from ...protocols.assets import LocalAsset, read_checked_asset
from ...protocols.json_io import parse_protocol_json, read_protocol_json
from ...protocols.schema import ProtocolError
from .planning import Collection, VERSION, origins


ACCEPTED_PILOT = 'd6417fdd1df113966868c988a33d8c7f503c917e7416f7c44fead912535c57fd'


class Role(str, Enum):
    DEVELOPMENT = 'development'
    ASSESSMENT = 'assessment'


@dataclass(frozen=True)
class WindowRequest:
    series_id: str
    horizon: int
    lag: int
    seed: int
    role: Role
    origins: tuple[int, ...]
    values: np.ndarray


def verify_pilot_executor(receipt_path, inventory):
    receipt_path = Path(receipt_path)
    _, payload = read_checked_asset(receipt_path.parent, LocalAsset(receipt_path.name, ACCEPTED_PILOT))
    receipt = parse_protocol_json(payload.decode('utf-8'))
    if receipt.get('official_test_used') is not False or receipt['code'] != inventory:
        raise ProtocolError('Executor does not match the accepted B source inventory')
    return {'previous_pilot_sha256': ACCEPTED_PILOT, 'verified_files': len(inventory)}


def read_manifest(bundle):
    lock = read_protocol_json(Path(bundle) / 'lock.json')
    _, payload = read_checked_asset(bundle, LocalAsset('manifest.json', lock['manifest_sha256']))
    record = parse_protocol_json(payload.decode('utf-8'))
    if (record.get('protocol') != VERSION or record.get('official_test_used') is not False
            or record.get('searches') != 24 or len(record['rows']) != 12):
        raise ProtocolError('A complete extension TRAIN seal is required')
    return record, sha256(payload).hexdigest()


def load_window_request(directory, case, *, seed, role):
    if not isinstance(role, Role) or type(seed) is not int or seed not in case['seeds']:
        raise ProtocolError('Unknown extension role or seed')
    if Path(directory).name != role.value:
        raise ProtocolError('A worker can only open its declared data role')
    name = case['id']
    if not isinstance(name, str) or not name.isalnum():
        raise ProtocolError('Invalid sealed series key')
    group = Collection(**case['collection'])
    field = 'development_sha256' if role is Role.DEVELOPMENT else 'assessment_sha256'
    selected = tuple(case['search'] if role is Role.DEVELOPMENT else case['assessment'])
    _, data = read_checked_asset(directory, LocalAsset(f'{name}.npy', case[field]))
    values = np.load(BytesIO(data), allow_pickle=False)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ProtocolError('Invalid frozen forecasting array')
    if role is Role.DEVELOPMENT:
        full_length = len(values) + 3 * group.horizon
        starts, late = origins(full_length, group)
        if selected != starts or len(values) != late[0]:
            raise ProtocolError('Development array contains late or missing observations')
    else:
        _, late = origins(len(values), group)
        if selected != late:
            raise ProtocolError('Assessment array does not match the frozen evaluation windows')
    return WindowRequest(name, group.horizon, group.lag, seed, role, selected, values)
