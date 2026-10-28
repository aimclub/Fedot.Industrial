from pathlib import Path

from coverage import Coverage


ROOT = Path(__file__).resolve().parents[3]


def test_failure_handlers_and_api_facade_are_not_blanket_excluded():
    coverage = Coverage(config_file=str(ROOT / 'pyproject.toml'))
    exclusions = coverage.get_option('report:exclude_also')
    assert 'except Exception' not in exclusions
    assert 'class ApiTemplate' not in exclusions


def test_ci_measures_benchmark_runtime_as_well_as_library():
    workflow = (ROOT / '.github/workflows/poetry_unit_test.yml').read_text(encoding='utf-8')
    assert '--cov=fedot_ind --cov=benchmark.industrial' in workflow
