"""Pure identity admission, complete-score ranking and paired summaries."""

from dataclasses import dataclass
from itertools import combinations
import json
import math
from statistics import mean, stdev

import numpy as np
from scipy.stats import rankdata

from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json


@dataclass(frozen=True)
class FrozenCandidate:
    name: str
    configuration_id: str
    configuration_json: str
    label: str

    @property
    def configuration(self):
        return json.loads(self.configuration_json)


def admit_candidates(catalogue, expected):
    chosen = [row for row in catalogue if row['candidate'] in expected]
    if len(chosen) != len(expected) or {row['candidate'] for row in chosen} != set(expected):
        raise ValueError('Require each frozen candidate exactly once')
    result = []
    for row in chosen:
        raw = row['configuration']
        configuration_json = canonical_json(raw)
        identity = PreparedEvaluationGraph(None, configuration_json).configuration_id
        if identity != row['configuration_id'] or identity != expected[row['candidate']]:
            raise ValueError('Canonical candidate identity differs from the approved audit')
        if (set(raw) != {'schema_version', 'task', 'nodes'}
                or raw['schema_version'] != 'industrial_study_candidate@1' or raw['task'] != 'regression'
                or not isinstance(raw['nodes'], list) or not raw['nodes']):
            raise ValueError('Require a regression study configuration')
        nodes = raw['nodes']
        for node in nodes:
            if (not isinstance(node, dict) or set(node) != {'operation', 'parameters', 'parents'}
                    or not isinstance(node['operation'], str) or not isinstance(node['parameters'], dict)
                    or not isinstance(node['parents'], list)
                    or any(not isinstance(index, int) or not 0 <= index < len(nodes) for index in node['parents'])
                    or len(set(node['parents'])) != len(node['parents'])):
                raise ValueError('Invalid frozen node shape')
        visiting, visited = set(), set()

        def visit(index):
            if index in visiting:
                raise ValueError('Frozen graph must be acyclic')
            if index not in visited:
                visiting.add(index)
                for parent in nodes[index]['parents']:
                    visit(parent)
                visiting.remove(index)
                visited.add(index)
        visit(0)
        if len(visited) != len(nodes):
            raise ValueError('Frozen graph must be connected to root zero')
        result.append(FrozenCandidate(row['candidate'], identity, configuration_json, row['label']))
    return tuple(sorted(result, key=lambda choice: choice.name))


def rank_scores(scores):
    if (not isinstance(scores, dict) or len(scores) < 2
            or any(type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
                   for loss in scores.values())):
        raise ValueError('Ranks require at least two finite nonnegative losses')
    names = sorted(scores)
    average = rankdata([scores[name] for name in names], method='average')
    ordered = sorted(names, key=lambda name: (scores[name], name))
    return [{'candidate': name, 'loss': scores[name], 'decision_rank': ordinal + 1,
             'average_rank': float(average[names.index(name)])} for ordinal, name in enumerate(ordered)]


def choose_inner(records, names, folds):
    expected = {(fold, name) for fold in range(folds) for name in names}
    keys = [(row['inner_fold'], row['candidate']) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Require every unique inner fold/candidate score')
    by_key = {(row['inner_fold'], row['candidate']): row['loss'] for row in records}
    # Validate each fold before averaging; bool, NaN and incomplete evidence fail closed.
    for fold in range(folds):
        rank_scores({name: by_key[fold, name] for name in names})
    scores = {name: math.fsum(by_key[fold, name] for fold in range(folds)) / folds for name in names}
    ranking = rank_scores(scores)
    return {'candidate': ranking[0]['candidate'], 'mean_inner_loss': ranking[0]['loss'],
            'margin_to_second': ranking[1]['loss'] - ranking[0]['loss'], 'ranking': ranking}


def rank_correlation(first, second):
    if set(first) != set(second):
        raise ValueError('Rank correlations require the same candidate set')
    a = {row['candidate']: row['average_rank'] for row in rank_scores(first)}
    b = {row['candidate']: row['average_rank'] for row in rank_scores(second)}
    x, y = np.asarray([a[k] for k in sorted(a)]), np.asarray([b[k] for k in sorted(b)])
    return None if np.ptp(x) == 0 or np.ptp(y) == 0 else float(np.corrcoef(x, y)[0, 1])


def summarize(inner, outer, selections, names):
    folds = sorted(selections)
    outer_keys = [(row['outer_fold'], row['candidate']) for row in outer]
    inner_keys = [(row['outer_fold'], row['inner_fold'], row['candidate']) for row in inner]
    if (folds != list(range(5)) or len(outer_keys) != 25 or len(set(outer_keys)) != 25
            or set(outer_keys) != {(fold, name) for fold in folds for name in names}
            or len(inner_keys) != 75 or len(set(inner_keys)) != 75
            or set(inner_keys) != {(fold, split, name) for fold in folds for split in range(3) for name in names}):
        raise ValueError('Require the complete approved 75-inner/25-outer matrix')
    ranking_rows, comparisons, paired = [], [], []
    inner_means, outer_scores = {}, {}
    for fold in folds:
        selection = choose_inner([row for row in inner if row['outer_fold'] == fold], names, 3)
        if selection != selections[fold]:
            raise ValueError('Locked selection differs from the inner evidence')
        inner_means[fold] = {row['candidate']: row['loss'] for row in selection['ranking']}
        outer_scores[fold] = {row['candidate']: row['loss'] for row in outer if row['outer_fold'] == fold}
        for stage, scores in (('inner_mean', inner_means[fold]), ('outer', outer_scores[fold])):
            ranking_rows.extend({'outer_fold': fold, 'stage': stage, **row} for row in rank_scores(scores))
        comparisons.append({'kind': 'inner_mean_vs_outer', 'first_fold': fold, 'second_fold': fold,
                            'spearman': rank_correlation(inner_means[fold], outer_scores[fold])})
        for split in range(3):
            scores = {row['candidate']: row['loss'] for row in inner
                      if row['outer_fold'] == fold and row['inner_fold'] == split}
            comparisons.append({'kind': f'inner_{split}_vs_outer', 'first_fold': fold, 'second_fold': fold,
                                'spearman': rank_correlation(scores, outer_scores[fold])})
    for stage, scores in (('inner_mean', inner_means), ('outer', outer_scores)):
        for first, second in combinations(folds, 2):
            comparisons.append({'kind': f'{stage}_between_folds', 'first_fold': first, 'second_fold': second,
                                'spearman': rank_correlation(scores[first], scores[second])})
    for first, second in combinations(names, 2):
        deltas = [outer_scores[fold][first] - outer_scores[fold][second] for fold in folds]
        paired.append({'first': first, 'second': second, 'mean_first_minus_second': mean(deltas),
                       'sd_difference': stdev(deltas), 'first_better_folds': sum(x < 0 for x in deltas),
                       'ties': sum(x == 0 for x in deltas), 'differences': deltas})
    methods = []
    for name in names:
        losses = [outer_scores[fold][name] for fold in folds]
        ranks = [row['average_rank'] for row in ranking_rows if row['candidate'] == name and row['stage'] == 'outer']
        rows = [row for row in outer if row['candidate'] == name]
        squared = [(y - p) ** 2 for row in rows for y, p in zip(row['target'], row['prediction'])]
        methods.append({'candidate': name, 'mean_outer_loss': mean(losses), 'sd_outer_loss': stdev(losses),
                        'worst_outer_loss': max(losses), 'oof_rmse_original_units': math.sqrt(mean(squared)),
                        'mean_outer_rank': mean(ranks), 'sd_outer_rank': stdev(ranks),
                        'inner_selected_folds': sum(selections[f]['candidate'] == name for f in folds),
                        'outer_best_folds_including_ties': sum(outer_scores[f][name] == min(outer_scores[f].values())
                                                               for f in folds)})
    selected = [outer_scores[f][selections[f]['candidate']] for f in folds]
    return {'methods': methods, 'rankings': ranking_rows, 'correlations': comparisons, 'paired': paired,
            'selected': {'mean_outer_loss': mean(selected), 'sd_outer_loss': stdev(selected),
                         'worst_outer_loss': max(selected),
                         'outer_best_folds_including_ties': sum(selected[f] == min(outer_scores[f].values())
                                                                for f in folds),
                         'mean_gap_to_outer_best': mean(selected[f] - min(outer_scores[f].values()) for f in folds)},
            'official_test_used': False, 'full_study_accepted': False,
            'interpretation': 'conditional_fixed_pool_TRAIN_ranking_not_independent_search_evaluation'}
