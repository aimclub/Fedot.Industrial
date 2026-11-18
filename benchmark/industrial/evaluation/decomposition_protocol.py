"""Synthetic diagnostic cases with independent seeds and labelled components."""

from dataclasses import dataclass

import numpy as np

from .forecasting_protocols import vector
from ..protocols.schema import ProtocolError, fields, integer


VERSION = 'synthetic_decomposition@1'


@dataclass(frozen=True)
class SyntheticSpec:
    length: int
    horizon: int
    period: int
    train_seed: int
    evaluation_seed: int
    noise: float = 0.1

    @classmethod
    def parse(cls, payload):
        data = fields(payload, ('version', 'length', 'horizon', 'period', 'train_seed', 'evaluation_seed'),
                      'synthetic profile', ('noise',))
        if data['version'] != VERSION:
            raise ProtocolError('Unsupported synthetic profile version')
        return cls(**{name: value for name, value in data.items() if name != 'version'})

    def __post_init__(self):
        integer(self.period, 'period', minimum=2)
        integer(self.length, 'length', minimum=2 * self.period)
        integer(self.horizon, 'horizon')
        integer(self.train_seed, 'train_seed', minimum=0)
        integer(self.evaluation_seed, 'evaluation_seed', minimum=0)
        if (self.train_seed == self.evaluation_seed or type(self.noise) not in (int, float)
                or not np.isfinite(self.noise) or self.noise < 0
                or self.horizon >= self.length - self.period):
            raise ProtocolError('Synthetic splits require different seeds, nonnegative noise and sufficient history')


def synthetic_case(spec: SyntheticSpec, partition):
    seeds = {'train': spec.train_seed, 'evaluation': spec.evaluation_seed}
    if partition not in seeds:
        raise ProtocolError('Synthetic partition must be train or evaluation')
    rng = np.random.default_rng(seeds[partition])
    grid = np.arange(spec.length)
    trend = rng.uniform(0.05, 0.2) * grid
    seasonal = np.sin(2 * np.pi * grid / spec.period + rng.uniform(0, 2 * np.pi))
    regime = np.where(grid >= spec.length // 2, rng.uniform(1, 3), 0.0)
    noise = rng.normal(0, spec.noise, spec.length)
    return {'version': VERSION, 'partition': partition, 'seed': seeds[partition],
            'origin': spec.length - spec.horizon - 1, 'period': spec.period,
            'values': trend + seasonal + regime + noise,
            'components': {'trend': trend, 'seasonal': seasonal, 'regime': regime, 'noise': noise}}


def component_errors(expected, reconstructed):
    """Use named components; mismatches are errors, not silently permuted."""
    if not expected or expected.keys() != reconstructed.keys():
        raise ProtocolError('Reconstruction must include every declared component by name')
    scores = {}
    for name, truth in expected.items():
        truth, estimate = vector(truth, name), vector(reconstructed[name], f'predicted {name}')
        if truth.shape != estimate.shape:
            raise ProtocolError(f'Unaligned component {name}')
        truth_spectrum = np.fft.rfft(truth - truth.mean())
        estimate_spectrum = np.fft.rfft(estimate - estimate.mean())
        power = np.abs(truth_spectrum)
        strongest = int(np.argmax(power))
        actual, predicted = truth_spectrum[strongest], estimate_spectrum[strongest]
        phase_error = (float(abs(np.angle(predicted * np.conj(actual))))
                       if abs(actual) > 1e-12 and abs(predicted) > 1e-12 else None)
        scores[name] = {'rmse': float(np.sqrt(np.mean((truth - estimate) ** 2))),
                        'dominant_phase_error_radians': phase_error,
                        'spectral_error': float(np.linalg.norm(power - np.abs(estimate_spectrum)) / len(truth))}
    return scores


def seasonal_control(case):
    history = case['values'][:case['origin'] + 1]
    truth = case['values'][case['origin'] + 1:]
    prediction = np.resize(history[-case['period']:], len(truth))
    return {'model': 'seasonal_repeat', 'mae': float(np.abs(truth - prediction).mean())}


def decomposition_control(values, period):
    """A small classical additive reference, not a new model family."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    integer(period, 'period', minimum=2)
    result = seasonal_decompose(vector(values, 'values'), model='additive', period=period, extrapolate_trend=period - 1)
    return {'trend': result.trend, 'seasonal': result.seasonal, 'residual': result.resid}
