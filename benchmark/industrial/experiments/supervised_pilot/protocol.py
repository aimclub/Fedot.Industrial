"""Closed, typed protocol for the first classification pilot."""

from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import PurePosixPath
import re

from ...protocols.assets import LocalAsset
from ...protocols.json_io import parse_protocol_json
from ...protocols.schema import ProtocolError, array, fields, integer, string


TSC_VERSION = 'tsc_evolution_pilot@1'


@dataclass(frozen=True)
class TrainSource:
    name: str
    shape: tuple[int, int, int]
    classes: int
    development_size: int
    asset: LocalAsset
    split_sha256: str

    def __post_init__(self):
        if not re.fullmatch('[A-Za-z][A-Za-z0-9]*', self.name):
            raise ProtocolError('Invalid dataset identifier')
        path = PurePosixPath(self.asset.path)
        if (path.suffix not in {'.ts', '.tsv'} or path.stem != f'{self.name}_TRAIN'
                or any(part.casefold() == 'test' for part in path.parts[:-1])):
            raise ProtocolError('Source must be a separately declared TRAIN file')
        if not re.fullmatch('[a-f0-9]{64}', self.split_sha256):
            raise ProtocolError('Invalid split SHA-256')
        if not self.classes <= self.development_size < self.shape[0]:
            raise ProtocolError('Expected development and assessment sizes are inconsistent')


@dataclass(frozen=True)
class SplitProfile:
    sklearn: str
    split_seed: int
    epoch_seed: int
    folds: int
    holdout: float


@dataclass(frozen=True)
class SearchBudget:
    seeds: tuple[int, ...]
    population: int
    generations: int
    elites: int
    tuning_generations: tuple[int, ...]
    candidate_seconds: int
    search_seconds: int
    process_seconds: int

    def __post_init__(self):
        phases = self.tuning_generations
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds) or self.elites >= self.population
                or phases != tuple(sorted(set(phases))) or any(i > self.generations for i in phases)
                or not self.candidate_seconds < self.search_seconds < self.process_seconds):
            raise ProtocolError('Inconsistent search budget')


@dataclass(frozen=True)
class PilotProtocol:
    version: str
    profile: SplitProfile
    search: SearchBudget
    datasets: tuple[TrainSource, ...]

    def tasks(self):
        return tuple((source.name, seed) for source in self.datasets for seed in self.search.seeds)

    def as_payload(self):
        payload = asdict(self)
        for row in payload['datasets']:
            row.update(row.pop('asset'))
        return payload


def _ints(value, path, *, minimum=1):
    return tuple(integer(item, path, minimum=minimum) for item in array(value, path))


def _source(value):
    data = fields(value, ('name', 'shape', 'classes', 'development_size', 'path', 'sha256', 'split_sha256'),
                  'dataset')
    shape = _ints(data['shape'], 'shape')
    if len(shape) != 3:
        raise ProtocolError('Expected panel shape (objects, channels, time)')
    return TrainSource(
        string(data['name'], 'name'), shape, integer(data['classes'], 'classes', minimum=2),
        integer(data['development_size'], 'development_size'), LocalAsset(data['path'], data['sha256']),
        string(data['split_sha256'], 'split_sha256'),
    )


def parse_tsc_protocol(value):
    data = fields(value, ('version', 'profile', 'search', 'datasets'), 'pilot')
    if data['version'] != TSC_VERSION:
        raise ProtocolError('Unsupported TSC pilot version')
    profile = fields(data['profile'], tuple(SplitProfile.__dataclass_fields__), 'profile')
    if type(profile['holdout']) is not float or not 0 < profile['holdout'] < 1:
        raise ProtocolError('holdout must be a fraction in (0, 1)')
    split_profile = SplitProfile(
        string(profile['sklearn'], 'sklearn'), integer(profile['split_seed'], 'split_seed', minimum=0),
        integer(profile['epoch_seed'], 'epoch_seed', minimum=0), integer(profile['folds'], 'folds', minimum=2),
        profile['holdout'],
    )
    search = fields(data['search'], tuple(SearchBudget.__dataclass_fields__), 'search')
    search = SearchBudget(**{
        name: (_ints(item, name, minimum=0 if name == 'seeds' else 1)
               if name in {'seeds', 'tuning_generations'} else integer(item, name))
        for name, item in search.items()
    })
    datasets = tuple(_source(row) for row in array(data['datasets'], 'datasets'))
    expected = ('ECG200', 'GunPoint', 'ArrowHead', 'BasicMotions')
    if tuple(row.name for row in datasets) != expected:
        raise ProtocolError('TSC pilot requires its ordered dataset catalog')
    return PilotProtocol(TSC_VERSION, split_profile, search, datasets)


def tsc_protocol():
    source = files(__package__).joinpath('tsc.json').read_text(encoding='utf-8')
    return parse_tsc_protocol(parse_protocol_json(source))
