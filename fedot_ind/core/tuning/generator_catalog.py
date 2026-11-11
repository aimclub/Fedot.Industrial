"""A finite, immutable generator catalog shared by tuning and evolution."""
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from math import isfinite


def scalar_identity(value):
    if type(value) not in (str, bool, int, float, type(None)) or (type(value) is float and not isfinite(value)):
        raise ValueError('Generator choices must be finite JSON scalars')
    return type(value), value


@dataclass(frozen=True)
class GeneratorParameter:
    name: str
    default: object
    values: tuple
    active_when: tuple

    def active(self, values):
        return all(scalar_identity(values[key]) in {scalar_identity(x) for x in choices}
                   for key, choices in self.active_when)


@dataclass(frozen=True)
class GeneratorSpace:
    operation: str
    parameters: tuple[GeneratorParameter, ...]

    @property
    def defaults(self):
        return {p.name: p.default for p in self.parameters}

    def normalize(self, values):
        result = {**self.defaults, **values}
        for p in self.parameters:
            if scalar_identity(result[p.name]) not in {scalar_identity(x) for x in p.values}:
                raise ValueError(f'{self.operation}.{p.name} is outside the generator catalog')
        for p in self.parameters:
            if not p.active(result):
                result[p.name] = p.default
        return result


def parse_generator_catalog(payload):
    if (type(payload) is not dict or set(payload) != {'schema_version', 'operations', 'aliases'}
            or payload['schema_version'] != 'industrial_generator_spaces@1'):
        raise ValueError('Unsupported generator catalog')
    if type(payload['operations']) is not dict or type(payload['aliases']) is not dict:
        raise ValueError('Generator operations and aliases must be objects')
    spaces = []
    for name, entries in payload['operations'].items():
        if not isinstance(name, str) or not name or type(entries) is not dict or not entries:
            raise ValueError('Generator operations require nonempty named parameter maps')
        domains = {}
        for key, row in entries.items():
            if (not isinstance(key, str) or not key or type(row) is not dict
                    or set(row) != {'default', 'values', 'active_when'}
                    or type(row['values']) is not list or not 1 <= len(row['values']) <= 32
                    or type(row['active_when']) is not dict):
                raise ValueError('Invalid generator parameter shape')
            choices = tuple(row['values'])
            identities = {scalar_identity(v) for v in choices}
            if len(identities) != len(choices) or scalar_identity(row['default']) not in identities:
                raise ValueError('Generator choices must be distinct and include the default')
            domains[key] = identities
        parameters = []
        for key, row in entries.items():
            conditions = []
            for dependency, allowed in row['active_when'].items():
                if dependency == key or dependency not in entries or type(allowed) is not list or not allowed:
                    raise ValueError('Invalid generator condition')
                if any(scalar_identity(v) not in domains[dependency] for v in allowed):
                    raise ValueError('Condition is outside its dependency domain')
                conditions.append((dependency, tuple(allowed)))
            parameters.append(GeneratorParameter(key, row['default'], tuple(row['values']), tuple(conditions)))
        ordered, ready = [], set()
        while parameters:
            batch = [p for p in parameters if all(key in ready for key, _ in p.active_when)]
            if not batch:
                raise ValueError('Cyclic generator dependencies')
            ordered.extend(batch)
            ready.update(p.name for p in batch)
            parameters = [p for p in parameters if p not in batch]
        spaces.append(GeneratorSpace(name, tuple(ordered)))
    names = {s.operation: s for s in spaces}
    for alias, canonical in payload['aliases'].items():
        if (not isinstance(alias, str) or not alias or alias in names
                or not isinstance(canonical, str) or canonical not in names):
            raise ValueError('Invalid or duplicate generator alias')
        spaces.append(GeneratorSpace(alias, names[canonical].parameters))
    return tuple(spaces)


@lru_cache(maxsize=1)
def generator_catalog():
    return parse_generator_catalog(json.loads(files(__package__).joinpath('generator_spaces.json').read_text('utf-8')))


def generator_space(operation):
    return next((space for space in generator_catalog() if space.operation == operation), None)


def parameter_is_active(operation, parameter, values):
    space = generator_space(operation)
    if space is None:
        return True
    config = space.normalize(values)
    spec = next((p for p in space.parameters if p.name == parameter), None)
    return spec is not None and spec.active(config)


def generator_hyperopt_space(hp):
    return {space.operation: {p.name: {'hyperopt-dist': hp.choice, 'sampling-scope': [list(p.values)]}
                              for p in space.parameters} for space in generator_catalog()}
