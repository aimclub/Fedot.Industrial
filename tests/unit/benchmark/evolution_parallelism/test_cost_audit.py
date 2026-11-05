import pytest

from benchmark.industrial.experiments.cyclic_mutations.cost_audit import summarize_operations


def row(operation, seconds):
    return dict(operation=operation, exclusive_seconds=seconds, evaluation_id='e1', configuration_id='c1',
                configuration={'nodes': []})


def test_exclusive_costs_conserve_work_without_double_counting_nested_calls():
    operations, candidates = summarize_operations([row('generator', 3.), row('generator', 2.), row('ridge', .5)])
    assert operations[0]['worker_seconds'] == 5.
    assert sum(r['fraction_of_traced_work'] for r in operations) == pytest.approx(1.)
    assert candidates['e1']['operation_seconds'] == {'generator': 5., 'ridge': .5}


@pytest.mark.parametrize('seconds', [-1, float('nan'), float('inf')])
def test_invalid_costs_are_not_silently_accepted(seconds):
    with pytest.raises(ValueError):
        summarize_operations([row('generator', seconds)])
