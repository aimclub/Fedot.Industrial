"""Independent control prediction after a complete, hash-bound search."""

from io import BytesIO
import json

import numpy as np

from .data import LocalAsset, Panel, _snapshot, canonical_bytes, digest, read_development
from .execution import _batch, read_record, write_record
from .runtime import Job, ValidationFold, evaluate
from .space import initial, parse_graph
from .training import Recipe


def assess(data, output, record, recipe_decision, budget):
    import os
    from time import time
    from fedot_ind.core.optimizer.evaluation.dask_config import DaskExecutionConfig
    from fedot_ind.core.optimizer.evaluation.dask_executor import DaskEvaluationSession

    result = read_record(output / 'search.json')
    if (result['status'] != 'complete' or result['development_sha256'] != record['development']
            or result['recipes_sha256'] != digest(canonical_bytes(recipe_decision))):
        raise ValueError('Assessment requires a complete search on the same prepared inputs')
    winner = parse_graph(result['winner']['graph'])
    if result['winner']['graph_sha256'] != winner.key:
        raise ValueError('The chosen graph changed after search')
    dev, description = read_development(data / 'development', record['development'])
    _, raw = _snapshot(data / 'assessment', LocalAsset('receipt.json', record['assessment']))
    receipt = json.loads(raw)
    if (receipt['role'] != 'assessment' or receipt['dataset'] != dev.dataset
            or receipt['train_sha256'] != description['train_sha256']
            or receipt['split_sha256'] != description['split_sha256']):
        raise ValueError('The assessment receipt does not match development')
    _, raw = _snapshot(data / 'assessment', LocalAsset('panel.npz', receipt['panel_sha256']))
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        external = Panel(dev.dataset, archive['rows'], archive['features'], archive['target'])
    if tuple(external.rows) != tuple(receipt['assessment']) or np.intersect1d(external.rows, dev.rows).size:
        raise ValueError('Assessment object keys do not match the seal or overlap development')
    controls = (('quantile_logreg', 0), ('rqa_logreg', 4), ('cnn2d', 6), ('inception', 7), ('resnet', 8))
    graphs = (('winner', winner), *((label, initial()[index]) for label, index in controls))
    fold = (ValidationFold(dev, external, tuple(description['final_epoch_validation'])),)
    recipes = {family: Recipe(**params) for family, params in recipe_decision['recipes'].items()}
    jobs = tuple((label, Job(graph, fold, tuple(description['classes']), recipes, 'cuda:0')) for label, graph in graphs)
    target = output / 'assessment.json'
    if target.exists():
        raise FileExistsError(target)
    deadline = float(os.environ.get('INDUSTRIAL_SEARCH_DEADLINE_UNIX', time() + budget.search_seconds))
    attempts = []
    with DaskEvaluationSession(2, DaskExecutionConfig()) as session:
        rows = _batch(session, jobs, budget, result['seed'], deadline, evaluate, attempts)
    from dataclasses import asdict
    write_record(target, dict(results=[asdict(row) for row in rows], attempts=[asdict(item) for item in attempts],
                              official_test_used=False, assessment_sha256=record['assessment'], graph_sha256=winner.key))
    return 0 if all(row.status.value == 'succeeded' for row in rows) else 1
