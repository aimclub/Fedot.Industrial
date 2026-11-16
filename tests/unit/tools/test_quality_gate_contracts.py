from pathlib import Path
import shlex

from coverage import Coverage
import yaml


ROOT = Path(__file__).resolve().parents[3]


def test_failure_handlers_and_api_facade_are_not_blanket_excluded():
    coverage = Coverage(config_file=str(ROOT / 'pyproject.toml'))
    exclusions = coverage.get_option('report:exclude_also')
    assert 'except Exception' not in exclusions
    assert 'class ApiTemplate' not in exclusions


def test_ci_measures_library_neural_and_benchmark_coverage():
    path = ROOT / '.github/workflows/poetry_unit_test.yml'
    workflow = yaml.safe_load(path.read_text(encoding='utf-8'))
    steps = workflow['jobs']['test']['steps']
    commands = (shlex.split(step['run'], comments=True) for step in steps if 'run' in step)
    pytest_args, = (command[3:] for command in commands if command[:3] == ['python', '-m', 'pytest'])
    assert {'--cov=fedot_ind', '--cov=fedot_ind.neural', '--cov=benchmark.industrial'} <= set(pytest_args)
