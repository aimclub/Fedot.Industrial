from types import SimpleNamespace
from unittest.mock import patch

from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator

from benchmark.industrial.experiments.cyclic_mutations.history import SearchHistoryJournal, linked_history, atomic_text


def sample():
    first = Individual(OptGraph(OptNode('ridge')), fitness=SingleObjFitness(2.))
    child = Individual(OptGraph(OptNode('rf')), ParentOperator('mutation', 'single_change', first),
                       fitness=SingleObjFitness(1.))
    history = OptHistory(Objective({'loss': lambda g: 0.}).get_info())
    history.add_to_history([first], 'initial')
    # Archive snapshots intentionally omit lineage; the journal must recover it.
    history.add_to_history([Individual(child.graph, uid=child.uid, fitness=child.fitness)], 'final_choices')
    return first, child, history


def test_linked_history_recovers_archive_winner_lineage_without_changing_native_generations():
    first, child, history = sample()
    result = linked_history(history, {child.uid: child})
    assert not history.generations[-1][0].parents
    assert result.generations[-1][0].parents[0] is result.generations[0][0]
    assert result.generations_count == history.generations_count
    restored = OptHistory.load(result.save())
    assert restored.generations[-1][0].parents[0].uid == first.uid
    assert tuple(restored.generations[-1][0].parent_operator.operators) == ('single_change',)


def test_native_visuals_render_from_saved_history_without_fit(tmp_path):
    from benchmark.industrial.experiments.cyclic_mutations.history import render_native_history
    first, child, history = sample()
    for i in range(3):
        history.add_to_history([first, child], f'generation_{i}')
    path = tmp_path / 'history.json'
    atomic_text(path, linked_history(history, {child.uid: child}).save())
    with patch('fedot.core.operations.operation.Operation.fit', side_effect=AssertionError('Fit forbidden')):
        records = render_native_history(path, tmp_path / 'plots', animations=False)
    assert all(r['status'] == 'succeeded' for r in records), records


def test_lineage_prefers_completed_parent_over_unevaluated_reference():
    first, child, history = sample()
    stale = Individual(first.graph, uid=first.uid)
    from dataclasses import replace
    child = replace(child, parent_operator=ParentOperator('mutation', 'single_change', stale))
    result = linked_history(history, {first.uid: first, child.uid: child})
    assert result.generations[-1][0].parents[0].fitness.valid
    assert result.generations[-1][0].parents[0].fitness.value == 2.


def test_checkpoints_reload_without_training_and_survive_later_write_failure(tmp_path):
    first, child, history = sample()
    owner = SimpleNamespace(staged_search=None, history=history, final_individuals=(child,))
    journal = SearchHistoryJournal(tmp_path, 42)
    journal.owner = owner
    journal.created(child)
    journal.checkpoint(owner)
    path = tmp_path / 'history/opt_history.json'
    original = path.read_bytes()
    with patch('pathlib.Path.replace', side_effect=OSError('interrupted')):
        try:
            atomic_text(path, 'partial')
        except OSError:
            pass
    assert path.read_bytes() == original
    restored = OptHistory.load(path)
    assert restored.generations[-1][0].parents[0].uid == first.uid
    assert not list(path.parent.glob('*.tmp'))
