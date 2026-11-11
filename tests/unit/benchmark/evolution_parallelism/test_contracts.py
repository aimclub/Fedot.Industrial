from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from benchmark.industrial.experiments.evolution_parallelism.config import SearchConfig
from benchmark.industrial.experiments.evolution_parallelism.data import load_data
from benchmark.industrial.experiments.evolution_parallelism.operation_timing import operation_trace


@pytest.mark.parametrize('field,value', [('generations', 14), ('generations', True), ('workers', 0),
                                         ('workers', 9), ('budget_seconds', float('nan')), ('budget_seconds', 1201)])
def test_search_config_enforces_explicit_bounds(field, value):
    with pytest.raises(ValueError):
        replace(SearchConfig('classification', 'sequential'), **{field: value})


def test_preview_needs_no_ml_runtime_and_writes_nothing(tmp_path):
    code = '''
import importlib.abc,sys
class NoRuntime(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'fedot','golem','torch','dask','distributed','sktime'}: raise AssertionError(fullname)
sys.meta_path.insert(0,NoRuntime())
from benchmark.industrial.experiments.evolution_parallelism.__main__ import main
main()
'''
    result = subprocess.run([sys.executable, '-c', code, '--output-dir', str(tmp_path / 'unused')],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('task', ['classification', 'regression', 'ts_forecasting'])
def test_pilot_has_no_test_container_and_finite_disjoint_training_folds(task):
    data = load_data(task, 42, include_test=False)
    assert data.test is None and not data.source['test_used_for_search']
    assert len(data.folds) == 3
    for train, validation in data.folds:
        assert np.isfinite(np.asarray(train.features)).all()
        assert train is not validation
    if task != 'ts_forecasting':
        for fold in data.source['folds']:
            assert set(fold['train_indices']).isdisjoint(fold['validation_indices'])


def test_operation_instrumentation_restores_methods_on_error(tmp_path):
    from fedot.core.operations.operation import Operation
    before = (Operation.fit, Operation.predict, Operation.predict_for_fit)
    with pytest.raises(RuntimeError):
        with operation_trace(tmp_path, {'evaluation_id': 'one'}):
            assert Operation.fit is not before[0]
            raise RuntimeError('body')
    assert (Operation.fit, Operation.predict, Operation.predict_for_fit) == before


def test_all_legacy_feature_operations_explicitly_disable_disk_cache():
    from benchmark.industrial.experiments.evolution_parallelism.config import load_defaults
    from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest
    from fedot.extensions.parameter_rules import resolve_extension_params
    parameters = load_defaults()['operation_parameters']
    specs = {spec.name: spec for spec in build_industrial_extension_manifest().transforms}
    for name in ('quantile_extractor', 'fourier_basis', 'wavelet_basis'):
        assert parameters[name]['use_cache'] is False
        assert resolve_extension_params(specs[name], parameters[name]).is_right()


def test_stagnation_timeout_cannot_preempt_the_demo_search_budget():
    from benchmark.industrial.experiments.evolution_parallelism.config import load_defaults
    from benchmark.industrial.experiments.evolution_parallelism.search import build_environment
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    settings = load_defaults()
    requirements, _ = build_environment(settings, Task(TaskTypesEnum.classification), 4)
    assert requirements.early_stopping_timeout * 60 == settings['budget_seconds']
