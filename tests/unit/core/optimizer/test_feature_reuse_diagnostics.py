"""The cost study is explicit, bounded and rejects any output difference."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.industrial.experiments.mutation_planning.population_study import feature_reuse_diagnostics as diagnostic


def _defaults():
    return json.loads(Path(diagnostic.__file__).with_name('feature_reuse_defaults.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('field,value', [('paired_repeats', 2), ('predict_repeats', True),
                                         ('max_bytes', -1), ('max_entries', 0), ('seed', -1),
                                         ('candidates', ['composite']), ('datasets', ['ECG200', 'ECG200']),
                                         ('schema_version', 'unknown')])
def test_diagnostic_config_rejects_changed_or_unbounded_contract(field, value):
    raw = _defaults()
    raw[field] = value
    with pytest.raises(ValueError):
        diagnostic.parse_config(raw)


def test_dry_run_reads_no_dataset_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, '_train_arrays', lambda *args: pytest.fail('TRAIN opened in preview'))
    assert diagnostic.run(tmp_path / 'missing_source', tmp_path / 'output')['input'] == 'original_TRAIN_only'
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('difference', ['value', 'dtype', 'shape', 'field'])
def test_exact_comparison_cannot_hide_differences(difference):
    a = {'features': np.ones((2, 3))}
    b = deepcopy(a)
    if difference == 'value':
        b['features'][0, 0] = np.nextafter(1., 2.)
    elif difference == 'dtype':
        b['features'] = b['features'].astype(np.float32)
    elif difference == 'shape':
        b['features'] = b['features'].reshape(3, 2)
    else:
        b['extra'] = np.ones(2)
    with pytest.raises(ValueError):
        diagnostic.compare_arrays(a, b)
    diagnostic.compare_arrays(a, deepcopy(a))
