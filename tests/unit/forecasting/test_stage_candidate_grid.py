from industrial_forecasting.tuning.execution import _build_stage_candidate_grid
from industrial_forecasting.tuning.plan import ForecastingStageSearchSpace


def test_stage_candidates_deduplicate_nested_parameters_in_order():
    current = {'lagged_params': {'alpha': 1., 'layers': [8, 4]}}
    alternative = {'alpha': .1, 'layers': [16, 8]}
    space = ForecastingStageSearchSpace(
        model_name='hybrid_ensemble_forecaster', canonical_model_name='hybrid_ensemble_forecaster',
        family='hybrid_ensemble', stage='ensemble',
        parameter_space={'lagged_params': {'sampling-scope': [[
            alternative, {'layers': [8, 4], 'alpha': 1.}, {'layers': [16, 8], 'alpha': .1},
        ]]}},
    )

    result = _build_stage_candidate_grid(space, current_parameters=current)

    assert result == ({}, current, {'lagged_params': alternative})
    assert result[1]['lagged_params'] is current['lagged_params']
