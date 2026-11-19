"""No-fit forecast evaluation with explicit source overlap and available inputs."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..datasets.ltsf import LTSFWindow, load_ltsf_csv
from ..protocols.schema import ProtocolError, string, unique_strings


@dataclass(frozen=True)
class TrainingSources:
    model_release: str
    known_datasets: tuple[str, ...]
    complete: bool

    def __post_init__(self):
        string(self.model_release, 'model release')
        object.__setattr__(self, 'known_datasets', unique_strings(self.known_datasets, 'known training datasets'))
        if type(self.complete) is not bool:
            raise ProtocolError('Training source completeness must be explicit')

    def overlap(self, aliases):
        if set(aliases) & set(self.known_datasets):
            return 'known_overlap'
        return 'no_declared_overlap' if self.complete else 'unknown'


def evaluate_zero_shot(predict, window: LTSFWindow, *, training: TrainingSources, dataset_aliases,
                       input_layout='regular', dates_public, frequency):
    """Predict once; the callable receives neither future targets nor past-only covariates."""
    aliases = unique_strings(dataset_aliases, 'dataset aliases')
    if not aliases or type(dates_public) is not bool:
        raise ProtocolError('Declare dataset identity and whether evaluation dates are public')
    if input_layout != 'regular':
        return {'status': 'not_available', 'reason': f'Unsupported input layout: {input_layout}'}
    frequency = string(frequency, 'sampling frequency')
    truth = window.truth
    if not np.isfinite(truth).all() or not np.isfinite(window.inputs.history[-1]).all():
        return {'status': 'not_available', 'reason': 'Missing observed target or forecast truth'}
    naive_mae = float(np.abs(truth - window.inputs.history[-1]).mean())
    started = perf_counter()
    forecast = np.asarray(predict(window.inputs), dtype=float)
    elapsed = perf_counter() - started
    if forecast.shape != truth.shape or not np.isfinite(forecast).all():
        raise ProtocolError('Forecast must match the declared horizon and target dimensions')
    return {'status': 'success', 'mode': 'zero_shot', 'model_release': training.model_release,
            'training_sources_complete': training.complete, 'overlap': training.overlap(aliases),
            'datasets': aliases, 'dates_public': dates_public,
            'origin': window.inputs.timestamps[-1] if dates_public else None,
            'horizon': len(truth), 'frequency': frequency, 'prediction_seconds': elapsed,
            'mae': float(np.abs(truth - forecast).mean()),
            'last_value_mae': naive_mae}


def _check_frequency(table, frequency):
    import pandas as pd

    stamps = pd.DatetimeIndex(table.dates)
    expected = pd.date_range(stamps[0], periods=len(stamps), freq=frequency)
    if not stamps.equals(expected):
        raise ProtocolError('Regular transfer data do not match the declared frequency')
    return stamps


def load_transfer_window(release, root, *, origin_date, horizon, frequency, targets, time='date',
                         observed=(), known_future=()):
    """Read one explicitly declared timestamped split from a checked, licensed release."""
    import pandas as pd

    table = load_ltsf_csv(release, root, time=time, targets=targets, observed=observed, known_future=known_future)
    origin = pd.Timestamp(origin_date)
    if origin.tzinfo is None:
        raise ProtocolError('Transfer origin must include its time zone')
    stamps = _check_frequency(table, frequency)
    try:
        index = int(stamps.get_loc(origin))
    except KeyError as error:
        raise ProtocolError('Declared transfer origin is not present') from error
    return table.window(index, horizon)


def load_time_windows(release, root, *, target, frequency, test_length, horizon):
    """Replicate TIME's end-aligned, non-overlapping official test windows."""
    from ..protocols.schema import integer

    integer(horizon, 'TIME horizon')
    integer(test_length, 'TIME test length', minimum=horizon)
    table = load_ltsf_csv(release, root, time='timestamp', targets=(target,))
    _check_frequency(table, frequency)
    last_training_index = len(table.dates) - test_length - 1
    if last_training_index < 0:
        raise ProtocolError('TIME needs a non-empty history preceding its complete test tail')
    return tuple(table.window(last_training_index + index * horizon, horizon)
                 for index in range(test_length // horizon))
