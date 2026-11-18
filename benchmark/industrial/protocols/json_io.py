"""Read versioned inputs without silently overwriting duplicate fields."""

import json
from pathlib import Path

from .schema import ProtocolError


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ProtocolError('JSON input contains duplicate object keys')
    return result


def _invalid_constant(value):
    raise ProtocolError(f'Non-finite JSON constant: {value}')


def read_protocol_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
