from fedot_ind.core.models.ts_forecasting.forecast_tuning import stage_tuning_execution as _impl
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_execution import *  # noqa: F401,F403

tqdm = _impl.tqdm


def run_sequential_stage_tuning(*args, **kwargs):
    _impl.tqdm = globals().get('tqdm', _impl.tqdm)
    return _impl.run_sequential_stage_tuning(*args, **kwargs)
