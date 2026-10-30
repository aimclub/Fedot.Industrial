"""Pure job selection; smoke budgets do not overwrite frozen quality jobs."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Trial:
    index: int
    scenario: str
    task: str
    arm: str
    seed: int
    seconds: int
    generations: int
    process_seconds: int
    smoke: bool

    @property
    def key(self):
        return f'{self.index:03d}_{self.scenario}_{self.arm}_{self.seed}'


def launch_settings():
    value = json.loads(Path(__file__).with_name('launch.json').read_text('utf-8'))
    if value['schema_version'] != 'industrial_quality_launcher@1':
        raise ValueError('Unsupported launcher settings')
    return value


def trial_plan(lock, *, smoke):
    settings = launch_settings()
    if smoke:
        scenarios = list(lock['datasets'])
        arms, seed = lock['protocol']['arms'], lock['protocol']['seeds'][0]
        jobs = [next(row for row in lock['jobs'] if row['scenario'] == name
                     and row['arm'] == arms[index % len(arms)] and row['seed'] == seed)
                for index, name in enumerate(scenarios)]
    else:
        jobs = lock['jobs']
    return tuple(Trial(index, row['scenario'], row['task'], row['arm'], row['seed'],
                       settings['smoke_seconds'] if smoke else row['search_seconds'],
                       settings['smoke_generations'] if smoke else row['minimum_generations'],
                       settings['smoke_process_seconds'] if smoke else row['full_process_seconds'], smoke)
                 for index, row in enumerate(jobs))
