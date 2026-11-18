"""Industrial-only semantics of a benchmark profile, independent of execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import PureWindowsPath
import re


PROTOCOL_VERSION = 'benchmark_industrial_protocol@1'


class ProtocolError(ValueError):
    """The declared protocol is unsupported or internally inconsistent."""


class EvaluationMode(str, Enum):
    OFFICIAL = 'official'
    TRAIN_ONLY = 'train_only'
    SMOKE = 'smoke'
    RESEARCH = 'research'


class Partition(str, Enum):
    VALIDATION = 'validation'
    TEST = 'test'


class Access(str, Enum):
    PUBLIC = 'public'
    RESTRICTED = 'restricted'
    LOCAL = 'local'


class ObservationUnit(str, Enum):
    SERIES = 'series'
    RECORD = 'record'
    EVENT = 'event'


class MetricDirection(str, Enum):
    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'


def mapping(value, path):
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ProtocolError(f'{path} must be an object with string keys')
    return value


def fields(value, required, path, optional=()):
    value = mapping(value, path)
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ProtocolError(f'{path}: missing {sorted(missing)}, unknown {sorted(unknown)}')
    return value


def string(value, path):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProtocolError(f'{path} must be a non-empty, trimmed string')
    return value


def integer(value, path, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ProtocolError(f'{path} must be an integer >= {minimum}')
    return value


def array(value, path):
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f'{path} must be an array')
    return value


def unique_strings(value, path):
    items = tuple(string(item, path) for item in array(value, path))
    if len(items) != len(set(items)):
        raise ProtocolError(f'{path} contains duplicates')
    return tuple(sorted(items))


def choice(value, enum_type, path):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f'{path} must be one of {[item.value for item in enum_type]}') from error


@dataclass(frozen=True)
class DatasetRelease:
    version: str
    source: str
    artifact: str
    sha256: str
    unit: ObservationUnit
    access: Access
    license: str | None

    @classmethod
    def parse(cls, value, path):
        data = fields(value, ('version', 'source', 'artifact', 'sha256', 'unit', 'access', 'license'), path)
        artifact = string(data['artifact'], f'{path}.artifact')
        parts = artifact.split('/')
        if any(not part or part != part.rstrip(' .') or PureWindowsPath(part).is_reserved() for part in parts) or re.search(
            r'[\\:*?"<>|\x00-\x1f]', artifact
        ):
            raise ProtocolError(f'{path}.artifact must be a portable relative file path')
        digest = string(data['sha256'], f'{path}.sha256').lower()
        if re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ProtocolError(f'{path}.sha256 must contain 64 hexadecimal characters')
        license_id = data['license']
        if license_id is not None:
            license_id = string(license_id, f'{path}.license')
        return cls(string(data['version'], f'{path}.version'), string(data['source'], f'{path}.source'),
                   artifact, digest, choice(data['unit'], ObservationUnit, f'{path}.unit'),
                   choice(data['access'], Access, f'{path}.access'), license_id)

    def comparison_identity(self):
        return {'version': self.version, 'source': self.source, 'sha256': self.sha256, 'unit': self.unit}


def check_partitions(mode, validation, test, path):
    if mode is EvaluationMode.OFFICIAL and not test:
        raise ProtocolError(f'{path}: official evaluation requires a test partition')
    if mode in (EvaluationMode.TRAIN_ONLY, EvaluationMode.SMOKE) and (test or not validation):
        raise ProtocolError(f'{path}: {mode.value} requires validation and forbids test')
    if not (validation or test):
        raise ProtocolError(f'{path}: no evaluation partition declared')


@dataclass(frozen=True)
class MembershipSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    groups: tuple[tuple[str, str], ...]
    kind: str = 'membership'

    @classmethod
    def parse(cls, value, mode, path):
        data = fields(value, ('kind', 'train', 'validation', 'test'), path, ('groups',))
        train, validation, test = (
            unique_strings(data[name], f'{path}.{name}') for name in ('train', 'validation', 'test')
        )
        items = {*train, *validation, *test}
        if not train or len(items) != len(train) + len(validation) + len(test):
            raise ProtocolError(f'{path}: train must not be empty and partitions must be disjoint')
        check_partitions(mode, validation, test, path)
        groups = mapping(data.get('groups', {}), f'{path}.groups')
        if groups:
            if groups.keys() != items:
                raise ProtocolError(f'{path}.groups must cover exactly the declared items')
            by_group = {}
            for name, partition in (('train', train), ('validation', validation), ('test', test)):
                for item in partition:
                    group = string(groups[item], f'{path}.groups.{item}')
                    if by_group.setdefault(group, name) != name:
                        raise ProtocolError(f'{path}: group {group!r} appears in multiple partitions')
        return cls(train, validation, test, tuple(sorted(groups.items())))

    def to_dict(self):
        return {'kind': self.kind, 'train': list(self.train), 'validation': list(self.validation),
                'test': list(self.test), 'groups': dict(self.groups)}


def coordinate(value, path):
    if type(value) is int:
        return integer(value, path, minimum=0)
    if isinstance(value, str):
        try:
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                return date.fromisoformat(value).isoformat()
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})', value):
                parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return parsed.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        except (ValueError, OverflowError) as error:
            raise ProtocolError(f'{path} is not a valid date or timestamp') from error
    raise ProtocolError(f'{path} must be a non-negative index, ISO date or timezone-aware ISO timestamp')


def coordinate_kind(value):
    if type(value) is int:
        return 'index'
    return 'date' if len(value) == 10 else 'timestamp'


@dataclass(frozen=True)
class ForecastCase:
    item_id: str
    series_id: str
    partition: Partition
    origin: int | str
    fit_end: int | str
    target_end: int | str
    horizon: int

    @classmethod
    def parse(cls, value, path):
        data = fields(value, ('item_id', 'series_id', 'partition', 'origin', 'fit_end', 'target_end', 'horizon'), path)
        origin = coordinate(data['origin'], f'{path}.origin')
        fit_end = coordinate(data['fit_end'], f'{path}.fit_end')
        target_end = coordinate(data['target_end'], f'{path}.target_end')
        if len({coordinate_kind(value) for value in (origin, fit_end, target_end)}) != 1:
            raise ProtocolError(f'{path}: forecast boundaries must use the same coordinate type')
        if not fit_end <= origin < target_end:
            raise ProtocolError(f'{path}: expected fit_end <= origin < target_end')
        horizon = integer(data['horizon'], f'{path}.horizon')
        if type(origin) is int and target_end - origin != horizon:
            raise ProtocolError(f'{path}: index target_end must equal origin + horizon')
        return cls(string(data['item_id'], f'{path}.item_id'), string(data['series_id'], f'{path}.series_id'),
                   choice(data['partition'], Partition, f'{path}.partition'), origin, fit_end, target_end, horizon)


@dataclass(frozen=True)
class TemporalSplit:
    frequency: str
    cases: tuple[ForecastCase, ...]
    kind: str = 'temporal'

    @classmethod
    def parse(cls, value, mode, path):
        data = fields(value, ('kind', 'frequency', 'cases'), path)
        cases = tuple(ForecastCase.parse(item, f'{path}.cases[{i}]')
                      for i, item in enumerate(array(data['cases'], f'{path}.cases')))
        validation = tuple(case for case in cases if case.partition is Partition.VALIDATION)
        test = tuple(case for case in cases if case.partition is Partition.TEST)
        check_partitions(mode, validation, test, path)
        if len({coordinate_kind(case.origin) for case in cases}) != 1:
            raise ProtocolError(f'{path}: forecast cases must use one coordinate type')
        if len({case.item_id for case in cases}) != len(cases):
            raise ProtocolError(f'{path}: duplicate forecast item id')
        if len({(case.series_id, case.origin, case.horizon) for case in cases}) != len(cases):
            raise ProtocolError(f'{path}: duplicate forecast window')
        last_validation = {}
        first_test = {}
        for case in validation:
            previous = last_validation.get(case.series_id, case.target_end)
            last_validation[case.series_id] = max(previous, case.target_end)
        for case in test:
            previous = first_test.get(case.series_id, case.origin)
            first_test[case.series_id] = min(previous, case.origin)
        if any(last_validation[key] > first_test[key] for key in last_validation.keys() & first_test.keys()):
            raise ProtocolError(f'{path}: validation targets must not overlap test targets for each series')
        return cls(string(data['frequency'], f'{path}.frequency'), tuple(sorted(cases, key=lambda case: case.item_id)))


@dataclass(frozen=True)
class InputPolicy:
    target: str
    observed: tuple[str, ...]
    known_future: tuple[str, ...]

    @classmethod
    def parse(cls, value, path):
        data = fields(value, ('target', 'observed', 'known_future'), path)
        target = string(data['target'], f'{path}.target')
        observed = unique_strings(data['observed'], f'{path}.observed')
        future = unique_strings(data['known_future'], f'{path}.known_future')
        if set(observed).intersection(future) or target in future:
            raise ProtocolError(f'{path}: target and observed-only inputs cannot be known-future inputs')
        return cls(target, observed, future)


@dataclass(frozen=True)
class PreprocessingStep:
    name: str
    version: str
    fit_on: str

    @classmethod
    def parse(cls, value, path):
        data = fields(value, ('name', 'version', 'fit_on'), path)
        if data['fit_on'] not in ('train', 'none'):
            raise ProtocolError(f'{path}.fit_on must be train or none')
        return cls(string(data['name'], f'{path}.name'), string(data['version'], f'{path}.version'), data['fit_on'])


@dataclass(frozen=True)
class MetricRule:
    version: str
    direction: MetricDirection
    aggregation: str

    @classmethod
    def parse(cls, value, path):
        data = fields(value, ('version', 'direction', 'aggregation'), path)
        return cls(string(data['version'], f'{path}.version'),
                   choice(data['direction'], MetricDirection, f'{path}.direction'),
                   string(data['aggregation'], f'{path}.aggregation'))


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str
    version: str
    mode: EvaluationMode
    inputs: InputPolicy
    split: MembershipSplit | TemporalSplit
    preprocessing: tuple[PreprocessingStep, ...]
    metrics: tuple[tuple[str, MetricRule], ...]

    @classmethod
    def parse(cls, value, task, metric_ids, path):
        data = fields(value, ('name', 'version', 'mode', 'inputs', 'split', 'preprocessing', 'metrics'), path)
        mode = choice(data['mode'], EvaluationMode, f'{path}.mode')
        definition = mapping(data['split'], f'{path}.split')
        inputs = InputPolicy.parse(data['inputs'], f'{path}.inputs')
        if task == 'forecasting':
            if definition.get('kind') != 'temporal':
                raise ProtocolError(f'{path}.split: forecasting requires a temporal split')
            split = TemporalSplit.parse(definition, mode, f'{path}.split')
        else:
            if definition.get('kind') != 'membership' or inputs.known_future or inputs.target in inputs.observed:
                raise ProtocolError(f'{path}: supervised records require a membership split and no target covariates')
            split = MembershipSplit.parse(definition, mode, f'{path}.split')
        rules = mapping(data['metrics'], f'{path}.metrics')
        if rules.keys() != metric_ids:
            raise ProtocolError(f'{path}.metrics must define exactly the configured ROAST metric IDs')
        preprocessing = tuple(PreprocessingStep.parse(item, f'{path}.preprocessing[{i}]')
                              for i, item in enumerate(array(data['preprocessing'], f'{path}.preprocessing')))
        return cls(string(data['name'], f'{path}.name'), string(data['version'], f'{path}.version'), mode, inputs, split,
                   preprocessing, tuple((name, MetricRule.parse(rules[name], f'{path}.metrics.{name}'))
                                        for name in sorted(rules)))
