"""Typed split plans and selection using inner losses only; no training or I/O."""

from dataclasses import asdict, dataclass
import math
import re
from statistics import mean, stdev

import numpy as np
from sklearn.model_selection import KFold

from ..ridge_diagnostics.analysis import RidgeStudy


@dataclass(frozen=True)
class NestedConfig:
    outer_folds: int
    inner_folds: int
    outer_seed: int
    inner_seed: int
    feature_seed: int
    population_protocol_sha256: str
    ridge_grid_sha256: str
    tie_break: str


@dataclass(frozen=True)
class Split:
    training: tuple[int, ...]
    validation: tuple[int, ...]


@dataclass(frozen=True)
class OuterFold:
    ordinal: int
    split: Split
    inner: tuple[Split, ...]


@dataclass(frozen=True)
class Candidate:
    structure: str
    scaling: str
    alpha: float


@dataclass(frozen=True)
class InnerScore:
    fold: int
    candidate: Candidate
    loss: float


@dataclass(frozen=True)
class Selection:
    candidate: Candidate
    mean_inner_loss: float
    margin_to_second: float
    scores: tuple[dict, ...]


def parse_config(raw):
    if (type(raw) is not dict or set(raw) != set(NestedConfig.__dataclass_fields__) | {'schema_version'}
            or raw['schema_version'] != 'industrial_nested_train_validation@1'):
        raise ValueError('Nested TRAIN fields/schema must match the contract')
    for key in ('outer_folds', 'inner_folds'):
        if type(raw[key]) is not int or not 2 <= raw[key] <= 10:
            raise ValueError(f'{key} must be an integer between 2 and 10')
    for key in ('outer_seed', 'inner_seed', 'feature_seed'):
        if type(raw[key]) is not int or not 0 <= raw[key] < 2 ** 32:
            raise ValueError(f'{key} must be a uint32 seed')
    for key in ('population_protocol_sha256', 'ridge_grid_sha256'):
        if type(raw[key]) is not str or re.fullmatch('[0-9a-f]{64}', raw[key]) is None:
            raise ValueError(f'{key} must be a SHA-256 digest')
    if raw['tie_break'] != 'exact_mean_then_structure_order_scaling_order_descending_alpha':
        raise ValueError('Unsupported tie-breaking rule')
    return NestedConfig(**{name: raw[name] for name in NestedConfig.__dataclass_fields__})


def _split(training, validation):
    return Split(tuple(int(index) for index in training), tuple(int(index) for index in validation))


def make_plan(sample_count, config):
    if type(sample_count) is not int or sample_count < config.outer_folds:
        raise ValueError('Require enough original TRAIN samples for the outer folds')
    plan = []
    for ordinal, (training, validation) in enumerate(KFold(
            config.outer_folds, shuffle=True, random_state=config.outer_seed).split(np.arange(sample_count))):
        if len(training) - math.ceil(len(training) / config.inner_folds) < 2:
            raise ValueError('Require at least two training samples in every inner fold')
        inner = tuple(_split(tr, val) for tr, val in KFold(
            config.inner_folds, shuffle=True, random_state=config.inner_seed).split(training))
        plan.append(OuterFold(ordinal, _split(training, validation), inner))
    return tuple(plan)


def candidates(grid: RidgeStudy):
    return tuple(Candidate(structure, scaling, float(alpha)) for structure in grid.structures
                 for scaling in grid.scalings for alpha in grid.alphas)


def select_candidate(records: tuple[InnerScore, ...], grid, inner_folds):
    """No outer labels, predictions or scores are accepted by this interface."""
    choices = candidates(grid)
    expected = {(fold, choice) for fold in range(inner_folds) for choice in choices}
    keys = [(row.fold, row.candidate) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Selection requires every unique inner fold/candidate score')
    values = {}
    for row in records:
        if type(row.loss) not in (int, float) or not math.isfinite(row.loss) or row.loss < 0:
            raise ValueError('Inner losses must be finite nonnegative numbers')
        values[row.fold, row.candidate] = row.loss
    means = {choice: math.fsum(values[fold, choice] for fold in range(inner_folds)) / inner_folds
             for choice in choices}
    ordered = sorted(choices, key=lambda choice: (means[choice], grid.structures.index(choice.structure),
                                                  grid.scalings.index(choice.scaling), -choice.alpha))
    return Selection(ordered[0], means[ordered[0]], means[ordered[1]] - means[ordered[0]],
                     tuple({**asdict(choice), 'mean_inner_loss': means[choice]} for choice in ordered))


def summarize_outer(records, plan):
    methods = ('selected', 'median', 'simple_alpha1', 'composite_alpha1', 'q_pdl')
    expected = {(fold.ordinal, method) for fold in plan for method in methods}
    keys = [(row['outer_fold'], row['method']) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Require all outer folds and all fixed comparisons exactly once')
    target_by_index, output = {}, []
    selected = {row['outer_fold']: row['outer_loss'] for row in records if row['method'] == 'selected'}
    planned = {fold.ordinal: fold for fold in plan}
    for row in records:
        fold = planned[row['outer_fold']]
        if tuple(row['validation_indices']) != fold.split.validation:
            raise ValueError('Outer row indices differ from the frozen plan')
        if tuple(row['training_indices']) != fold.split.training:
            raise ValueError('Outer training indices differ from the frozen plan')
        y, prediction = np.asarray(row['target']), np.asarray(row['prediction'])
        scale = row['normalization_scale']
        if (y.shape != (len(fold.split.validation),) or prediction.shape != y.shape
                or not np.isfinite(y).all() or not np.isfinite(prediction).all()
                or type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0):
            raise ValueError('Require finite aligned outer predictions and a positive training scale')
        if (type(row['outer_loss']) not in (int, float) or not math.isfinite(row['outer_loss'])
                or row['outer_loss'] < 0):
            raise ValueError('Outer losses must be finite nonnegative numbers')
        try:
            with np.errstate(over='raise', invalid='raise'):
                loss = float(np.sqrt(np.mean((y - prediction) ** 2)) / scale)
        except FloatingPointError as error:
            raise ValueError('Outer squared errors overflowed') from error
        if not math.isclose(loss, row['outer_loss'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Outer loss does not replay from stored predictions')
        for index, target in zip(row['validation_indices'], row['target']):
            if index in target_by_index and target_by_index[index] != target:
                raise ValueError('Conflicting outer targets across comparison methods')
            target_by_index[index] = target
    for row in records:
        scale = float(np.std([target_by_index[index] for index in row['training_indices']]))
        if not math.isclose(scale, row['normalization_scale'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Outer normalization must use its training targets only')
    for method in methods:
        group = sorted((row for row in records if row['method'] == method), key=lambda row: row['outer_fold'])
        losses = [row['outer_loss'] for row in group]
        deltas = [selected[row['outer_fold']] - row['outer_loss'] for row in group]
        squared = [(y - p) ** 2 for row in group for y, p in zip(row['target'], row['prediction'])]
        output.append({'method': method, 'outer_folds': len(group), 'objects': len(squared),
                       'mean_outer_loss': mean(losses), 'sd_outer_loss': stdev(losses),
                       'worst_outer_loss': max(losses), 'oof_rmse_original_units': math.sqrt(mean(squared)),
                       'mean_selected_minus_method': mean(deltas),
                       'selected_better_folds': sum(delta < 0 for delta in deltas)})
    return {'methods': output, 'official_test_used': False, 'full_study_accepted': False,
            'interpretation': 'nested_TRAIN_procedure_evaluation_not_new_external_evidence'}
