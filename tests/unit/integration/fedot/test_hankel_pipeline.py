"""Regression checks for the explicit EVO-04 single-series extension boundary."""
from dataclasses import replace

import fedot
import numpy as np
import pytest
import torch

from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.extensions import get_registered_extensions
from fedot.extensions.contracts import ExtensionContractError
from fedot_ind.core.models.ts_forecasting.lagged_strategy.lagged_forecaster import LaggedAR
from fedot_ind.integration.fedot.compatibility import InputData, DataTypesEnum
from fedot_ind.integration.fedot.extensions import industrial_extension_scope

try:
    from fedot.core.data.split.temporal import temporal_tensor_holdout
except ModuleNotFoundError:
    pytest.skip('Requires the explicit EVO-04 temporal compatibility patch',
                allow_module_level=True)

pytest.importorskip('fedot.core.operations.evaluation.temporal_extensions',
                    reason='Requires the explicit EVO-04 temporal compatibility patch')


def _data(horizon, length=80):
    series = np.sin(np.arange(length) / 5) + np.arange(length) * .01
    return fedot.create_data(series, target=series,
                             task=Task(TaskTypesEnum.ts_forecasting,
                                       TsForecastingParams(horizon)),
                             ts_forecast_horizon=horizon, without_target=True)


@pytest.mark.parametrize('horizon, stride', [(1, 1), (8, 1), (8, 3)])
def test_hankel_ridge_uses_supervised_windows_and_never_holdout_targets(horizon, stride):
    data = _data(horizon)
    snapshot = data.features.clone()
    train, test = temporal_tensor_holdout(data)
    with industrial_extension_scope():
        pipeline = PipelineBuilder().add_node('hankelisation', params={
            'window_size': 10, 'stride': stride,
        }).add_node('ridge').build()
        fitted = pipeline.fit(train)
        history = train.features.numpy().reshape(-1)
        starts = range(0, len(history) - 10 - horizon + 1, stride)
        expected_x = np.array([history[i:i + 10] for i in starts])
        expected_y = np.array(
            [history[i + 10:i + 10 + horizon] for i in starts])
        np.testing.assert_allclose(fitted.features.numpy(), expected_x)
        np.testing.assert_allclose(fitted.target.numpy(), expected_y)
        result = pipeline.predict(test)
        assert result.predict.shape == (horizon,)
        assert torch.isfinite(result.predict).all()
        altered = replace(test, target=torch.full_like(test.target, 1e6))
        torch.testing.assert_close(
            pipeline.predict(altered).predict, result.predict)
        # Independent sklearn reference verifies the numerical window/target alignment.
        from sklearn.linear_model import Ridge
        reference = Ridge(alpha=1.).fit(expected_x, expected_y)
        # sklearn versions differ in retaining the singleton target axis.
        coefficients = pipeline.root_node.fitted_operation.coef_
        expected_shapes = (
            (10,), (1, 10)) if horizon == 1 else ((horizon, 10),)
        assert coefficients.shape in expected_shapes
        np.testing.assert_allclose(coefficients.reshape(horizon, 10), reference.coef_.reshape(horizon, 10),
                                   atol=1e-5)
        np.testing.assert_allclose(result.predict.numpy(), reference.predict(history[-10:][None, :]).reshape(-1),
                                   atol=1e-5)
        changed_horizon = replace(test, task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(horizon + 1)),
                                  ts_forecast_horizon=horizon + 1)
        with pytest.raises(ExtensionContractError) as error:
            pipeline.predict(changed_horizon)
        assert error.value.code == 'extension_execution_failed'
        assert 'fitted horizon' in str(error.value.__cause__)
    torch.testing.assert_close(data.features, snapshot)


def test_lagged_ar_tuning_scores_are_real_and_predict_owns_registration(monkeypatch):
    from golem.core.tuning.simultaneous import SimultaneousTuner

    scores = []
    original = SimultaneousTuner.get_metric_value

    def recording_metric(self, *args, **kwargs):
        metric = original(self, *args, **kwargs)
        scores.extend(np.asarray(metric).reshape(-1).tolist())
        return metric

    monkeypatch.setattr(SimultaneousTuner,
                        'get_metric_value', recording_metric)
    series = np.sin(np.arange(80) / 5)
    data = InputData(features=series, target=series.copy(), idx=np.arange(80),
                     task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(8)), data_type=DataTypesEnum.ts)
    before = tuple(get_registered_extensions())
    model = LaggedAR({'window_size': 10}).fit(data)
    assert tuple(get_registered_extensions()) == before
    # Initial, ten trials, final; a finite penalty is not a successful score.
    assert len(scores) >= 12
    assert all(np.isfinite(score) and abs(score) < 1e6 for score in scores)
    assert model.predict(data).predict.shape == (8,)
    assert tuple(get_registered_extensions()) == before
    with industrial_extension_scope():
        nested = tuple(get_registered_extensions())
        assert model.predict(data).predict.shape == (8,)
        assert tuple(get_registered_extensions()) == nested
    with pytest.raises(ValueError):
        model.predict(
            replace(data, features=series[:1], target=None, idx=np.array([80])))
    assert tuple(get_registered_extensions()) == before


def test_lagged_forecaster_extension_uses_full_history_for_window_percent():
    data = _data(8)
    with industrial_extension_scope():
        pipeline = PipelineBuilder().add_node(
            'lagged_forecaster', params={'window_size': 10}).build()
        pipeline.fit(data)
        model = pipeline.root_node.fitted_operation.implementation
        assert model.resolved_window_size_ == 8
        assert pipeline.predict(data).predict.shape == (8,)
