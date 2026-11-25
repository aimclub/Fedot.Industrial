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


def parse_protocol_json(text):
    try:
        return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except RecursionError as error:
        raise ProtocolError('JSON input exceeds the supported nesting depth') from error


def read_protocol_json(path):
    return parse_protocol_json(Path(path).read_text(encoding='utf-8'))
