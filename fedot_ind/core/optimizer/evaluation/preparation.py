"""Explicit preparation boundary for configuration-addressed evaluations."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Callable


def canonical_json(value: object) -> str:
    """Encode finite JSON values without repr fallbacks or key coercion."""
    def validate(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError('Configuration keys must be strings')
            for child in item.values():
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif isinstance(item, float):
            if not isfinite(item):
                raise ValueError('Configuration numbers must be finite')
        elif item is not None and type(item) not in (str, int, bool):
            raise ValueError(f'Unsupported configuration value: {type(item).__name__}')

    validate(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=True)


@dataclass(frozen=True)
class PreparedEvaluationGraph:
    """A coordinator-owned runtime graph and its immutable configuration snapshot."""

    graph: object
    configuration_json: str

    def __post_init__(self):
        if canonical_json(json.loads(self.configuration_json)) != self.configuration_json:
            raise ValueError('Prepared configuration must use canonical JSON')

    @property
    def configuration_id(self) -> str:
        return 'industrial-evaluation@1:' + sha256(self.configuration_json.encode('utf-8')).hexdigest()


GraphPreparation = Callable[[object], PreparedEvaluationGraph]
