"""Real OKHS-FDMD extension execution, including FEDOT's task-horizon transport."""
from dataclasses import replace

import fedot
import numpy as np
import pytest
import torch

from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.extensions.contracts import ExtensionContractError
from fedot_ind.integration.fedot.extensions import industrial_extension_scope

pytest.importorskip('fedot.core.data.split.temporal', reason='Requires the EVO-04 temporal compatibility patch')


@pytest.mark.parametrize('horizon', [1, 8])
def test_okhs_fdmd_extension_fits_and_predicts_actual_task_horizon(horizon):
    history = np.sin(np.arange(88) / 6.) + np.arange(88) * .01
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(horizon))
    data = fedot.create_data(history, target=history, task=task,
                             ts_forecast_horizon=horizon, without_target=True)
    original = data.features.clone()
    with industrial_extension_scope():
        pipeline = PipelineBuilder().add_node('okhs_fdmd_forecaster').build()
        fitted = pipeline.fit(data)
        model = pipeline.root_node.fitted_operation.implementation
        assert model.model_.forecast_horizon == horizon
        assert 'forecast_length' not in model.spec_.params
        inner = model.model_.inner_model_.model
        assert len(inner.fdmd_.eigenvalues_) == inner.fdmd_.modes_.shape[0]
        assert fitted.predict.shape == (1, horizon)
        prediction_data = replace(data, target=torch.full((1, horizon), 1e6))
        output = pipeline.predict(prediction_data)
        assert output.predict.shape == (horizon,)
        assert torch.isfinite(output.predict).all()
        unlabelled = replace(data, target=None)
        torch.testing.assert_close(pipeline.predict(unlabelled).predict, output.predict)
        assert model.get_diagnostics()['model_name'] == 'okhs_fdmd_forecaster'
    torch.testing.assert_close(data.features, original)


def test_okhs_fdmd_extension_rejects_horizon_conflict_before_model_fit():
    history = np.sin(np.arange(64) / 5.)
    data = fedot.create_data(history, target=history,
                             task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(8)),
                             ts_forecast_horizon=8, without_target=True)
    with industrial_extension_scope():
        pipeline = PipelineBuilder().add_node('okhs_fdmd_forecaster', params={'forecast_length': 3}).build()
        with pytest.raises(ValueError, match='agree with the task horizon'):
            pipeline.fit(data)
        assert pipeline.root_node.fitted_operation is None


def test_okhs_fdmd_does_not_hide_absence_of_stable_selected_modes():
    history = np.sin(np.arange(88) / 6.) + np.arange(88) * .01
    data = fedot.create_data(history, target=history,
                             task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(8)),
                             ts_forecast_horizon=8, without_target=True)
    with industrial_extension_scope():
        pipeline = PipelineBuilder().add_node('okhs_fdmd_forecaster', params={
            'mode_selection_policy': 'fixed', 'n_modes': 3,
        }).build()
        with pytest.raises(ExtensionContractError) as error:
            pipeline.fit(data)
        cause = error.value
        while cause.__cause__ is not None:
            cause = cause.__cause__
        assert isinstance(cause, ValueError)
        assert 'No stable modes remain' in str(cause)
