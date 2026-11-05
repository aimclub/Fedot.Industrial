"""Durable native history, including candidates discarded before survivor selection."""
from copy import copy, deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def linked_history(history, candidates):
    """Restore UID ownership without fabricating generations for archive-only candidates."""
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    result = OptHistory(history.objective)
    for original in history.generations:
        group = copy(original)
        group.data = list(original)
        group.metadata = deepcopy(original.metadata)
        result.generations.append(group)
    known = dict(candidates)
    pending = [item for group in result.generations for item in group]
    visited = set()
    while pending:
        item = pending.pop()
        if item.uid in visited:
            continue
        visited.add(item.uid)
        old = known.get(item.uid)
        if old is None:
            known[item.uid] = item
        else:
            known[item.uid] = replace(item, parent_operator=old.parent_operator or item.parent_operator)
        pending.extend(item.parents)
    resolved, visiting = {}, set()

    def resolve(item):
        if item.uid in resolved:
            return resolved[item.uid]
        if item.uid in visiting:
            raise ValueError('History contains a parent cycle')
        visiting.add(item.uid)
        original = known.get(item.uid, item)
        parent = original.parent_operator
        if parent is not None:
            parent = replace(parent, parent_individuals=tuple(resolve(p) for p in parent.parent_individuals))
        clone = replace(original, graph=deepcopy(original.graph),
                        metadata=deepcopy(original.metadata), parent_operator=parent)
        resolved[item.uid] = clone
        visiting.remove(item.uid)
        return clone

    for group in result.generations:
        group.data[:] = [resolve(item) for item in group]
    result.archive_history = [[resolve(item) for item in group] for group in result.archive_history]
    return result


class SearchHistoryJournal:
    def __init__(self, root, seed):
        self.root, self.seed = Path(root) / 'history', seed
        (self.root / 'candidates').mkdir(parents=True)
        self.candidates = {}
        self.write_seconds = 0.
        self.owner = None

    def phase(self):
        controller = getattr(self.owner, 'staged_search', None)
        return controller.stage.name if controller is not None else 'unphased'

    def attach(self, optimizer):
        from fedot_ind.core.optimizer.randomness import SeededOperator
        self.owner = optimizer
        mutation = optimizer.mutation
        while isinstance(mutation, SeededOperator):
            mutation = mutation.operator
        if mutation.candidate_observer is not None:
            raise ValueError('Candidate history observer already installed')
        mutation.candidate_observer = self.created

    def created(self, candidate):
        start = perf_counter()
        candidate.metadata['search_phase'] = self.phase()
        candidate.metadata['search_seed'] = self.seed
        # Only optimisation graphs are stored, never a fitted domain pipeline or dataset.
        self.candidates[candidate.uid] = candidate
        atomic_text(self.root / 'candidates' / f'{candidate.uid}.json', candidate.save())
        self.write_seconds += perf_counter() - start

    def completed(self, results):
        from golem.core.optimisers.opt_history_objects.individual import Individual
        start = perf_counter()
        for result in results:
            old = self.candidates.get(result.uid_of_individual)
            candidate = Individual(graph=deepcopy(result.graph), uid=result.uid_of_individual,
                                   parent_operator=None if old is None else old.parent_operator,
                                   metadata={**({} if old is None else old.metadata), **result.metadata},
                                   fitness=result.fitness)
            self.candidates[candidate.uid] = candidate
            atomic_text(self.root / 'candidates' / f'{candidate.uid}.json', candidate.save())
        self.write_seconds += perf_counter() - start

    def checkpoint(self, owner):
        start = perf_counter()
        history = linked_history(owner.history, self.candidates)
        history.generations[-1].metadata.update(dict(search_phase=self.phase(), search_seed=self.seed))
        atomic_text(self.root / 'opt_history.json', history.save())
        self.write_seconds += perf_counter() - start
        atomic_text(self.root / 'manifest.json', json.dumps(dict(
            schema_version='industrial_native_history@1', seed=self.seed,
            generations=history.generations_count, candidates=len(self.candidates),
            history_write_seconds=self.write_seconds, complete=bool(owner.final_individuals),
            checkpoint='opt_history.json', candidates_directory='candidates',
            raw_data_included=False), indent=2, allow_nan=False) + '\n')


def render_native_history(history_path, output_dir, *, animations=True):
    """Offline native rendering; no fit calls and no custom substitute charts."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.visualisation.pipeline_specific_visuals import PipelineHistoryVisualizer, PipelineVisualizer
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    history = OptHistory.load(history_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    with industrial_extension_scope():
        visualizer = PipelineHistoryVisualizer(history)
        plots = [('fitness_line', 'png', {}), ('fitness_box', 'png', {}),
                 ('operations_kde', 'png', {'use_tags': False}), ('diversity_line', 'png', {'show': False})]
        if animations:
            plots.extend([('operations_animated_bar', 'gif', {'use_tags': False}),
                          ('genealogical_path', 'gif', {'show': False})])
        for name, extension, options in plots:
            target = root / f'{name}.{extension}'
            try:
                getattr(visualizer, name)(save_path=str(target), **options)
                if not target.is_file():
                    raise RuntimeError('Native renderer did not create its artifact')
                records.append(dict(plot=name, status='succeeded', path=target.name))
            except (ValueError, KeyError, IndexError, TypeError, RuntimeError, ZeroDivisionError) as error:
                records.append(dict(plot=name, status='failed', error_type=type(error).__name__, message=str(error)))
            finally:
                plt.close('all')
        try:
            PipelineVisualizer(history.generations[-1][0].graph).visualise(save_path=str(root / 'best_graph.png'))
            records.append(dict(plot='best_graph', status='succeeded', path='best_graph.png'))
        finally:
            plt.close('all')
    atomic_text(root / 'render_manifest.json', json.dumps(records, indent=2) + '\n')
    return records
