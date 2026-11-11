"""Optional temporal contract, enabled only by the active FEDOT installation."""

import numpy as np
from fedot.extensions import runtime_contracts

from .contracts import IndustrialExtensionContractError, IndustrialExtensionErrorCode


def uses_temporal_contract(declaration):
    return 'ts_to_table' in declaration.tags


def supports_temporal_contract():
    return all(isinstance(getattr(runtime_contracts, name, None), type)
               for name in ('TemporalTransformInput', 'TemporalTransformOutput'))


def require_temporal_contract():
    if not supports_temporal_contract():
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.FEDOT_CONTRACT_UNAVAILABLE,
            'The installed FEDOT does not provide the temporal transform contract; this operation is disabled.',
            context={'required': ['TemporalTransformInput', 'TemporalTransformOutput']},
        )
    return runtime_contracts.TemporalTransformInput, runtime_contracts.TemporalTransformOutput


def fit_temporal_operation(runtime, features, invoke):
    require_temporal_contract()
    horizon = runtime.params.get('forecast_length')
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Temporal fitting requires an explicit positive forecast_length')
    return invoke(runtime._method('fit_temporal'), (features, horizon), {}, runtime.declaration.name)


def transform_temporal_operation(runtime, request, invoke):
    input_type, output_type = require_temporal_contract()
    if not isinstance(request, input_type):
        raise TypeError('Temporal transforms require TemporalTransformInput')
    if runtime.params.get('forecast_length', request.horizon) != request.horizon:
        raise ValueError('Temporal transform horizon differs from its fitted horizon')
    values = request.features
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    values = np.array(values, copy=True)
    if values.ndim != 1:
        raise ValueError('Temporal transforms require one univariate history')
    output = invoke(runtime._method('transform_temporal_values'),
                    (values, np.array(request.idx, copy=True), request.horizon, request.fit_stage),
                    {}, runtime.declaration.name)
    return output_type(output.features, output.target, output.idx)
