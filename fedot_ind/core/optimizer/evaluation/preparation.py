"""Explicit preparation boundary for configuration-addressed evaluations."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Callable


def canonical_json(value: object) -> str:
    """Encode finite JSON values without repr fallbacks or key coercion."""
    def validate(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError('Configuration keys must be strings')
            for child in item.values():
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        elif type(item) is float:
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


def pipeline_configuration(pipeline, task):
    """Ordered-root identity preserves shared nodes and runtime parent order."""
    order, identifiers, parents = [], {}, {}
    pending = [pipeline.root_node]
    while pending:
        node = pending.pop()
        key = id(node)
        if key in identifiers:
            continue
        identifiers[key] = len(order)
        order.append(node)
        incoming = node._nodes_from_with_fixed_order() if node.nodes_from else []
        parents[key] = incoming
        pending.extend(reversed(incoming))
    if len(order) != len(pipeline.nodes):
        raise ValueError('Prepared pipeline must be one connected rooted graph')
    return {'schema_version': 'industrial_study_candidate@1', 'task': task,
            'nodes': [{'operation': node.operation.operation_type, 'parameters': node.parameters,
                       'parents': [identifiers[id(parent)] for parent in parents[id(node)]]} for node in order]}
