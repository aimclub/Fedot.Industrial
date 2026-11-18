"""Resolve a ROAST suite and Industrial profiles into stable identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import TYPE_CHECKING
import json

from ..core import TaskType
from .schema import (
    PROTOCOL_VERSION,
    DatasetRelease,
    EvaluationProtocol,
    MembershipSplit,
    ObservationUnit,
    ProtocolError,
    fields,
    integer,
    mapping,
)

if TYPE_CHECKING:
    from roast.core.config import BenchmarkSuiteConfig


class MissingRoastError(RuntimeError):
    """The optional ROAST configuration types are not installed."""


def _reject_discarded_fields(raw, resolved, path):
    if isinstance(raw, Mapping) and isinstance(resolved, Mapping):
        unknown = raw.keys() - resolved.keys()
        if unknown:
            raise ProtocolError(f'{path}: unknown fields {sorted(unknown)}')
        for name, value in raw.items():
            _reject_discarded_fields(value, resolved[name], f'{path}.{name}')
    elif isinstance(raw, (list, tuple)) and isinstance(resolved, list):
        for index, (value, canonical) in enumerate(zip(raw, resolved, strict=True)):
            _reject_discarded_fields(value, canonical, f'{path}[{index}]')


def _read_suite(value):
    try:
        from roast.core.config import BenchmarkSuiteConfig
        from roast.core.schema import SchemaError
    except ModuleNotFoundError as error:
        if error.name != 'roast':
            raise
        raise MissingRoastError('Install the benchmark-protocol extra to resolve ROAST suites') from error

    raw = mapping(value, 'suite')
    try:
        suite = BenchmarkSuiteConfig.from_dict(raw)
    except SchemaError as error:
        raise ProtocolError(f'suite: {error}') from error
    _reject_discarded_fields(raw, suite.to_dict(), 'suite')
    budget = fields(suite.run.options.get('evaluation_budget'), ('max_evaluations', 'timeout_seconds'),
                    'suite.run.options.evaluation_budget')
    for name, amount in budget.items():
        integer(amount, f'suite.run.options.evaluation_budget.{name}')
    if suite.run.primary_metric is None:
        raise ProtocolError('suite.run.primary_metric must select a configured metric')
    return suite


def _digest(value):
    canonical = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(',', ':'))
    return sha256(canonical.encode('ascii')).hexdigest()


@dataclass(frozen=True)
class DatasetProfile:
    dataset_id: str
    release: DatasetRelease
    protocol: EvaluationProtocol

    def to_dict(self) -> dict:
        protocol = self.protocol
        split = protocol.split
        return {
            'release': asdict(self.release),
            'protocol': {
                'name': protocol.name,
                'version': protocol.version,
                'mode': protocol.mode,
                'inputs': asdict(protocol.inputs),
                'split': split.to_dict() if isinstance(split, MembershipSplit) else asdict(split),
                'preprocessing': tuple(asdict(step) for step in protocol.preprocessing),
                'metrics': {name: asdict(rule) for name, rule in protocol.metrics},
            },
        }


@dataclass(frozen=True)
class ResolvedProtocolPlan:
    suite: BenchmarkSuiteConfig
    profiles: tuple[DatasetProfile, ...]
    plan_id: str
    comparison_key: str

    def to_dict(self) -> dict:
        return {**_payload(self.suite, self.profiles), 'plan_id': self.plan_id, 'comparison_key': self.comparison_key}


def _payload(suite, profiles):
    return {'version': PROTOCOL_VERSION, 'suite': suite.to_dict(),
            'profiles': {profile.dataset_id: profile.to_dict() for profile in profiles}}


def _comparison_payload(suite, profiles, by_dataset):
    suite_record = suite.to_dict()
    return {
        'version': PROTOCOL_VERSION,
        'task_kind': suite.task_kind,
        'task_options': suite_record['task_options'],
        'primary_metric': suite.run.primary_metric,
        'metrics': {metric.metric_id: metric.to_dict() for metric in suite.metrics},
        'profiles': {
            profile.dataset_id: {
                'release': profile.release.comparison_identity(),
                'dataset': by_dataset[profile.dataset_id].to_dict(),
                'protocol': profile.to_dict()['protocol'],
            } for profile in profiles
        },
    }


def resolve_protocol(payload: Mapping) -> ResolvedProtocolPlan:
    """Validate authored or resolved JSON without fetching data or loading any model."""
    data = fields(payload, ('version', 'suite', 'profiles'), 'manifest', ('plan_id', 'comparison_key'))
    if data['version'] != PROTOCOL_VERSION:
        raise ProtocolError(f"Unsupported protocol version: {data['version']!r}")
    if ('plan_id' in data) != ('comparison_key' in data):
        raise ProtocolError('Resolved manifests require both plan_id and comparison_key')
    suite = _read_suite(data['suite'])
    try:
        task = TaskType(suite.task_kind)
    except ValueError as error:
        raise ProtocolError(f'Unsupported Industrial task kind: {suite.task_kind!r}') from error
    raw_profiles = mapping(data['profiles'], 'profiles')
    by_dataset = {dataset.dataset_id: dataset for dataset in suite.datasets}
    if raw_profiles.keys() != by_dataset.keys():
        raise ProtocolError('profiles must cover exactly the configured ROAST dataset IDs')
    metrics = {metric.metric_id for metric in suite.metrics}
    profiles = []
    for dataset_id in sorted(raw_profiles):
        path = f'profiles.{dataset_id}'
        profile = fields(raw_profiles[dataset_id], ('release', 'protocol'), path)
        release = DatasetRelease.parse(profile['release'], f'{path}.release')
        if task is TaskType.FORECASTING and release.unit is not ObservationUnit.SERIES:
            raise ProtocolError(f'{path}.release.unit: forecasting requires series')
        if task is not TaskType.FORECASTING and release.unit is ObservationUnit.EVENT:
            raise ProtocolError(f'{path}.release.unit: event protocols are not supported for this task')
        protocol = EvaluationProtocol.parse(profile['protocol'], task.value, metrics, f'{path}.protocol')
        profiles.append(DatasetProfile(dataset_id, release, protocol))
    profiles = tuple(profiles)
    plan_id = _digest(_payload(suite, profiles))
    comparison_key = _digest(_comparison_payload(suite, profiles, by_dataset))
    if 'plan_id' in data and (plan_id != data['plan_id'] or comparison_key != data['comparison_key']):
        raise ProtocolError('Resolved manifest identities do not match its content')
    return ResolvedProtocolPlan(suite, profiles, plan_id, comparison_key)
