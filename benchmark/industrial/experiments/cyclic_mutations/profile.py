"""Validate a bounded subset, never a second independent parameter catalog."""
import json
from pathlib import Path

from fedot_ind.core.optimizer.search_policy.contracts import ParameterGrid
from fedot_ind.core.optimizer.search_policy import load_staged_search_defaults
from fedot_ind.core.tuning.generator_catalog import generator_catalog, scalar_identity


def parse_parameter_profile(payload):
    if (type(payload) is not dict or set(payload) != {'schema_version', 'generators'}
            or payload['schema_version'] != 'industrial_bd_parameters@1' or type(payload['generators']) is not dict):
        raise ValueError('Invalid B/D parameter profile')
    catalog = {s.operation: s for s in generator_catalog()}
    if set(payload['generators']) != set(catalog):
        raise ValueError('B/D profile must cover every declared generator')
    grids = []
    for operation, entries in payload['generators'].items():
        specs = {p.name: p for p in catalog[operation].parameters}
        if type(entries) is not dict or set(entries) != set(specs):
            raise ValueError('Incomplete generator parameter profile')
        for name, values in entries.items():
            if type(values) is not list:
                raise ValueError('Profile choices must be arrays')
            grid = ParameterGrid(operation, name, tuple(values))
            if any(scalar_identity(v) not in {scalar_identity(x) for x in specs[name].values} for v in values):
                raise ValueError('Profile choices exceed the common catalog')
            grids.append(grid)
    if payload['generators']['fourier_basis']['output_format'] != ['signal']:
        raise ValueError('Representation changes are not ordinary parameter mutations')
    return (*load_staged_search_defaults().parameters, *grids)


def parameter_profile():
    return parse_parameter_profile(json.loads(Path(__file__).with_name('parameters.json').read_text('utf-8')))


def validate_training_bounds(data, grids):
    """Reject a profile that would be silently clamped on an inner TRAIN fold."""
    windows = next(g.values for g in grids if (g.operation, g.parameter) == ('hankelisation', 'window_size'))
    strides = next(g.values for g in grids if (g.operation, g.parameter) == ('hankelisation', 'stride'))
    if data.refit.task.task_type.value != 'ts_forecasting':
        return
    for train, _ in data.folds:
        horizon = train.task.task_params.forecast_length
        if max(windows) > len(train.features) - horizon or max(strides) > min(windows) // 2:
            raise ValueError('Hankel profile would require silent clamping on inner TRAIN')
