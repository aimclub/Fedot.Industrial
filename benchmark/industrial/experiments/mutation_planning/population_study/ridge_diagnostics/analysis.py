"""Finite TRAIN matrices, training-owned scaling, and descriptive split summaries."""

from collections import defaultdict
from dataclasses import dataclass
import math
from statistics import mean, stdev

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RidgeStudy:
    dataset: str
    primary_seeds: tuple[int, ...]
    additional_seeds: tuple[int, ...]
    validation_fraction: float
    structures: tuple[str, ...]
    scalings: tuple[str, ...]
    alphas: tuple[float, ...]
    reference_alpha: float
    replay_atol: float
    replay_rtol: float


def parse_config(raw):
    fields = set(RidgeStudy.__dataclass_fields__) | {'schema_version'}
    if (not isinstance(raw, dict) or set(raw) != fields
            or raw['schema_version'] != 'industrial_train_ridge_diagnostics@1'
            or raw['dataset'] != 'NaturalGasPricesSentiment'):
        raise ValueError('Require the approved NaturalGas TRAIN diagnostic contract')
    seeds = []
    for key in ('primary_seeds', 'additional_seeds'):
        group = raw[key]
        if (not isinstance(group, list) or len(group) < 3
                or any(not isinstance(seed, int) or not 0 <= seed < 2 ** 32 for seed in group)):
            raise ValueError('Require at least three uint32 seeds per group')
        seeds.extend(group)
    if len(set(seeds)) != len(seeds):
        raise ValueError('TRAIN split seeds must be unique across groups')
    for key, allowed in (('structures', {'simple', 'composite'}), ('scalings', {'none', 'standard'})):
        if (not isinstance(raw[key], list) or len(raw[key]) != len(allowed)
                or any(not isinstance(value, str) for value in raw[key]) or set(raw[key]) != allowed):
            raise ValueError(f'{key} must cover the two approved choices once')
    values = raw['alphas']
    if (not isinstance(values, list) or not values
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError('alphas must be unique finite positive numbers')
    for key in ('reference_alpha', 'validation_fraction', 'replay_atol', 'replay_rtol'):
        value = raw[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if raw['reference_alpha'] not in values or not 0 < raw['validation_fraction'] < 1:
        raise ValueError('Require a reference in the grid and a proper holdout fraction')
    return RidgeStudy(**{name: tuple(raw[name]) if name.endswith('seeds') or name in
                         ('structures', 'scalings', 'alphas') else raw[name]
                         for name in RidgeStudy.__dataclass_fields__})


def finite_matrix(values):
    matrix = np.asarray(values)
    if (matrix.ndim != 2 or not all(matrix.shape) or matrix.dtype.kind != 'f'
            or not np.isfinite(matrix).all()):
        raise ValueError('Require a nonempty finite floating-point feature matrix')
    return matrix.copy()


def matrix_profile(values, alpha):
    matrix = finite_matrix(values).astype(np.float64)
    centered = matrix - matrix.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    squared = singular ** 2
    smallest = squared[-1] if matrix.shape[1] <= matrix.shape[0] else 0.
    std = matrix.std(axis=0)
    positive = std[std > 0]
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    return {'shape': list(matrix.shape), 'dtype': str(np.asarray(values).dtype),
            'centered_rank': int((singular > tolerance).sum()), 'constant_columns': int((std == 0).sum()),
            'std_min_positive': float(positive.min()) if len(positive) else None,
            'std_max': float(std.max()), 'max_abs': float(np.abs(matrix).max()),
            'regularized_gram_condition': float((squared[0] + alpha) / (smallest + alpha)),
            'effective_degrees_of_freedom': float((squared / (squared + alpha)).sum())}


def fit_head(training, target, validation, *, scaling, alpha):
    """Never fit a transformer on validation; preserve the native feature dtype."""
    training, validation = finite_matrix(training), finite_matrix(validation)
    target = np.asarray(target)
    if (target.ndim != 1 or len(target) != len(training) or not np.isfinite(target).all()
            or training.shape[1] != validation.shape[1]):
        raise ValueError('Require finite aligned targets and matching feature columns')
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    scaler = None
    if scaling == 'standard':
        scaler = StandardScaler().fit(training)
        training, validation = scaler.transform(training), scaler.transform(validation)
    elif scaling != 'none':
        raise ValueError('Unknown scaling policy')
    model = Ridge(alpha=alpha).fit(training, target)
    train_prediction, prediction = model.predict(training), model.predict(validation)
    if not np.isfinite(train_prediction).all() or not np.isfinite(prediction).all():
        raise ValueError('Ridge returned nonfinite predictions')
    # Original-basis coefficients allow comparisons between split-specific scalers.
    coefficient = model.coef_.astype(np.float64)
    original_coefficient = coefficient if scaler is None else coefficient / scaler.scale_
    original_intercept = float(model.intercept_ if scaler is None else
                               model.intercept_ - scaler.mean_ @ original_coefficient)
    return {'train_prediction': train_prediction.tolist(), 'validation_prediction': prediction.tolist(),
            'model_parameters': model.get_params(deep=False),
            'coefficient': coefficient.tolist(), 'original_coefficient': original_coefficient.tolist(),
            'original_intercept': original_intercept, 'coefficient_l2': float(np.linalg.norm(coefficient)),
            'original_coefficient_l2': float(np.linalg.norm(original_coefficient)),
            'matrix': matrix_profile(training, alpha), 'validation_max_abs': float(np.abs(validation).max()),
            'scaler': None if scaler is None else {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                                                   'samples_seen': int(scaler.n_samples_seen_)}}


def summarize(records, config):
    """Descriptive repeated holdouts, not independent trials or nested CV."""
    expected = {(seed, structure, scaling, float(alpha))
                for seed in config.primary_seeds + config.additional_seeds
                for structure in config.structures for scaling in config.scalings for alpha in config.alphas}
    keys = [(row['seed'], row['structure'], row['scaling'], float(row['alpha'])) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Summary requires the complete unique preregistered grid')
    groups, splits = defaultdict(list), defaultdict(list)
    for row in records:
        if any(not math.isfinite(row[key]) or row[key] < 0 for key in ('validation_loss', 'train_loss')):
            raise ValueError('Losses must be finite and nonnegative')
        groups[row['structure'], row['scaling'], row['alpha']].append(row)
        splits[row['seed']].append(row)
    rows = []
    for (structure, scaling, alpha), group in sorted(groups.items()):
        values = [row['validation_loss'] for row in group]
        baseline = {row['seed']: row['validation_loss'] for row in groups[
            structure, 'none', config.reference_alpha]}
        coefficients = np.asarray([row['original_coefficient'] for row in group])
        center = coefficients.mean(axis=0)
        rows.append({'structure': structure, 'scaling': scaling, 'alpha': alpha, 'splits': len(group),
                     'mean_loss': mean(values), 'sd_loss': stdev(values), 'worst_loss': max(values),
                     'primary_mean': mean(row['validation_loss'] for row in group
                                          if row['seed'] in config.primary_seeds),
                     'additional_mean': mean(row['validation_loss'] for row in group
                                             if row['seed'] in config.additional_seeds),
                     'mean_train_loss': mean(row['train_loss'] for row in group),
                     'mean_paired_delta_to_unscaled_alpha1': mean(
                         row['validation_loss'] - baseline[row['seed']] for row in group),
                     'better_than_reference_splits': sum(
                         row['validation_loss'] < baseline[row['seed']] for row in group),
                     'coefficient_original_basis_rms_deviation': float(np.sqrt(np.mean(
                         np.sum((coefficients - center) ** 2, axis=1))))})
    selections = []
    for seed, candidates in sorted(splits.items()):
        ordered = sorted(candidates, key=lambda row: (row['validation_loss'], row['structure'],
                                                      row['scaling'], row['alpha']))
        winner = ordered[0]
        selections.append({'seed': seed, 'structure': winner['structure'], 'scaling': winner['scaling'],
                           'alpha': winner['alpha'], 'loss': winner['validation_loss'],
                           'margin_to_second': ordered[1]['validation_loss'] - winner['validation_loss']})
    return {'configurations': rows, 'per_split_selection': selections,
            'interpretation': 'overlapping_TRAIN_holdouts_descriptive_not_unbiased_generalization',
            'test_used': False, 'full_study_accepted': False}
