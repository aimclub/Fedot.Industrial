from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from benchmark.industrial.experiments.cyclic_mutations.execution import trials, supervise
from benchmark.industrial.experiments.cyclic_mutations.profile import validate_training_bounds, parameter_profile


def test_new_matrix_is_paired_and_admission_has_both_arms_per_task():
    names = dict(ECG200='classification', Tecator='regression', Airline='ts_forecasting')
    rows = [dict(scenario=n, task=task, arm=arm, seed=seed, search_seconds=600,
                 minimum_generations=15, full_process_seconds=1200)
            for n, task in names.items() for seed in (42, 123, 2026, 777, 2027) for arm in ('fixed', 'cyclic')]
    lock = dict(jobs=rows, admission_scenarios=list(names), protocol=dict(seeds=[42, 123, 2026, 777, 2027]))
    full, gate = trials(lock, False), trials(lock, True)
    assert len(full) == 30 and len(gate) == 6
    assert len({(r.task, r.arm) for r in gate}) == 6
    assert all(r.generations == 15 and r.seconds == 600 and r.smoke for r in gate)
    assert all(not r.smoke for r in full)


def test_missing_gate_blocks_full_before_creation(tmp_path):
    with patch('benchmark.industrial.experiments.cyclic_mutations.execution.checked_lock', return_value={}):
        with pytest.raises(RuntimeError, match='requires'):
            supervise(tmp_path / 'absent', tmp_path, 'seal', False)
    assert not (tmp_path / 'absent').exists()


def test_hankel_bounds_use_actual_train_and_reject_clamping():
    task = SimpleNamespace(task_type=SimpleNamespace(name='ts_forecasting'),
                           task_params=SimpleNamespace(forecast_length=12))
    data = SimpleNamespace(refit=SimpleNamespace(task=task), folds=[(
        SimpleNamespace(task=task, features=np.arange(40).reshape(1, -1)), None)])
    with pytest.raises(ValueError, match='clamping'):
        validate_training_bounds(data, parameter_profile())
    data.folds[0][0].features = np.arange(100).reshape(1, -1)
    validate_training_bounds(data, parameter_profile())
