import json
import sys

import pytest

from benchmark.industrial.experiments.staged_mutations.config import build_config, load_protocol


def test_future_comparison_arms_share_grids_actions_budget_and_family_slots():
    a, b = build_config('fixed'), build_config('staged')
    assert len(a.mutation_schedule.stages) == 1 and len(b.mutation_schedule.stages) == 3
    assert a.mutation_schedule.parameters == b.mutation_schedule.parameters
    assert a.evaluation_execution == b.evaluation_execution
    assert a.mutation_planning == b.mutation_planning
    assert a.mutation_schedule.max_families == b.mutation_schedule.max_families
    assert {s.family_fraction for s in (*a.mutation_schedule.stages, *b.mutation_schedule.stages)} == {.5}
    assert load_protocol()['training_enabled'] is False
    with pytest.raises(ValueError):
        build_config('unknown')


def test_preview_does_not_train_or_create_output(tmp_path, monkeypatch, capsys):
    from fedot.core.operations.operation import Operation
    from benchmark.industrial.experiments.staged_mutations.__main__ import main
    calls = []
    monkeypatch.setattr(Operation, 'fit', lambda *args, **kwargs: calls.append(args))
    output = tmp_path / 'unused'
    monkeypatch.setattr(sys, 'argv', ['staged_mutations', '--output-dir', str(output)])
    main()
    record = json.loads(capsys.readouterr().out)
    assert record['protocol']['training_enabled'] is False
    assert not output.exists() and calls == []
