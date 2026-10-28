"""Small validated, JSON-backed experiment boundary."""
from dataclasses import dataclass
import json
from pathlib import Path


TASKS = ('classification', 'regression', 'ts_forecasting')
BACKENDS = ('sequential', 'process', 'dask')


@dataclass(frozen=True)
class SearchConfig:
    task: str
    backend: str
    generations: int = 15
    workers: int = 4
    budget_seconds: float = 1100
    seed: int = 42
    lifecycle: str = 'batch'
    protocol: str = 'generations'
    pilot: bool = False
    native_threads: int = 1

    def __post_init__(self):
        if self.task not in TASKS or self.backend not in BACKENDS:
            raise ValueError('Unknown task or backend')
        for name, minimum, maximum in (('generations', 1 if self.pilot else 15, 1000),
                                       ('workers', 1, 8), ('seed', 0, 2**32 - 1), ('native_threads', 1, 8)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
        if type(self.budget_seconds) not in (int, float) or not 1 <= self.budget_seconds <= 1100:
            raise ValueError('Search budget must be finite and between 1 and 1100 seconds')
        if self.lifecycle not in ('batch', 'search') or self.protocol not in ('generations', 'time'):
            raise ValueError('Unknown lifecycle or protocol')


def load_defaults():
    value = json.loads(Path(__file__).with_name('defaults.json').read_text('utf-8'))
    if value['schema_version'] != 'industrial_evolution_parallelism@1':
        raise ValueError('Unsupported experiment settings')
    return value
