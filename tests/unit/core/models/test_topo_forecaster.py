from fedot_ind.core.models.ts_forecasting.lagged_model.topo_forecaster import (
    TopologicalAR,
    TopologicalRidgeForecaster,
)
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data import InputData
import numpy as np
import pytest

pytest.importorskip('fedot')
pytest.importorskip('torch')


def _build_ts_input(horizon: int = 6):
    series = np.linspace(1.0, 72.0, num=72)
    return InputData(
        idx=np.arange(len(series)),
        features=series.reshape(-1, 1),
        target=series,
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon)),
        data_type=DataTypesEnum.ts,
    )


def test_topological_ridge_forecaster_fits_and_predicts_with_runtime_contract(monkeypatch):
    monkeypatch.setattr(
        'fedot_ind.core.models.ts_forecasting.lagged_model.topo_forecaster._extract_topological_window_features',
        lambda window, patch_len, stride: np.asarray([
            float(np.mean(window)),
            float(np.std(window)),
            float(np.max(window)),
        ]),
    )

    model = TopologicalRidgeForecaster(
        forecast_horizon=6,
        window_size=12,
        patch_len=4,
        stride=1,
        alpha=1.0,
        device='cpu',
    )

    model.fit(np.linspace(1.0, 80.0, num=80))
    prediction = model.predict()

    assert prediction.shape == (6,)
    assert model.get_diagnostics()['model_family'] == 'lagged_linear'
    assert model.get_diagnostics()['trajectory_transform']['representation'] == 'topological_hankel'


def test_topological_ar_exposes_stage_tuning_contract(monkeypatch):
    monkeypatch.setattr(
        'fedot_ind.core.models.ts_forecasting.lagged_model.topo_forecaster._extract_topological_window_features',
        lambda window, patch_len, stride: np.asarray([
            float(np.mean(window)),
            float(np.std(window)),
            float(np.max(window)),
        ]),
    )

    implementation = TopologicalAR(
        OperationParameters(window_size=12, patch_len=4, stride=1, alpha=1.0, channel_model='ridge')
    )
    implementation.fit(_build_ts_input())

    plan = implementation.get_stage_tuning_plan()
    spaces = implementation.get_stage_search_spaces()
    execution = implementation.get_stage_tuning_execution({'forecast_head': {'alpha': 0.5}})

    assert plan['canonical_model_name'] == 'topo_forecaster'
    assert spaces[0]['stage'] == 'trajectory_transform'
    assert execution['final_parameters']['alpha'] == 0.5
