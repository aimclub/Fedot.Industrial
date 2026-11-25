import numpy as np
import pytest
import torch

from industrial_forecasting.neural.runtime import (
    ForecastingSplitKind,
    ForecastingSplitSpec,
    MLPForecastingHead,
    TensorDevicePolicy,
)
from industrial_forecasting.neural.hybrid import _safe_forecast
from industrial_forecasting.tuning.series import ForecastingSeriesEvaluator


def test_mlp_normalization_depends_only_on_the_fit_prefix():
    head = MLPForecastingHead(validation_fraction=.5, device_policy=TensorDevicePolicy(device='cpu'))
    prefix = torch.tensor([[0.], [2.]])
    for tail in (4., 400.):
        values = torch.cat((prefix, torch.full_like(prefix, tail)))
        _, _, fit_x, fit_y, validation_x, validation_y = head._prepare_fit_data(values, 2 * values)
        torch.testing.assert_close(head.feature_mean_, torch.tensor([[1.]]))
        torch.testing.assert_close(head.target_mean_, torch.tensor([[2.]]))
        torch.testing.assert_close(fit_x, torch.tensor([[-1.], [1.]]))
        torch.testing.assert_close(fit_y, fit_x)
        torch.testing.assert_close(validation_x, torch.full_like(prefix, tail - 1))
        torch.testing.assert_close(validation_y, validation_x)


@pytest.mark.parametrize('kind', [ForecastingSplitKind.EXPANDING_WINDOW, ForecastingSplitKind.ROLLING_WINDOW])
@pytest.mark.parametrize('step', [-1, 0, .5, True, '1'])
def test_window_splits_reject_invalid_steps_before_building_folds(kind, step):
    with pytest.raises(ValueError, match='step_length must be a positive integer'):
        ForecastingSplitSpec(kind=kind, step_length=step)


@pytest.mark.parametrize('ignore_gap', [False, True])
def test_temporal_evaluator_scores_after_the_gap(ignore_gap):
    requested = []

    class Linear:
        def fit(self, _history):
            return self

        def predict(self, history, *, forecast_horizon):
            requested.append(forecast_horizon)
            steps = 2 if ignore_gap else forecast_horizon
            return history[-1] + np.arange(1, steps + 1)

    def factory(_name, *, forecast_horizon, **_kwargs):
        assert forecast_horizon == 4
        return Linear()

    evaluator = ForecastingSeriesEvaluator(
        time_series=np.arange(20, dtype=float),
        model_name='lagged_ridge_forecaster',
        forecast_horizon=2,
        split_spec=ForecastingSplitSpec(validation_horizon=2, gap=2),
        model_factory=factory,
    )
    fold, = evaluator._iter_over_folds()
    if ignore_gap:
        with pytest.raises(ValueError, match='4 forecast steps including the gap'):
            evaluator._fit_predict_fold(fold)
    else:
        forecast, target, metric, _, _ = evaluator._fit_predict_fold(fold)
        np.testing.assert_array_equal(forecast, [18., 19.])
        np.testing.assert_array_equal(target, forecast)
        assert metric.metric_value == 0.
    assert requested == [4]


def test_temporal_owa_compares_the_seasonal_baseline_on_the_same_gap():
    class Seasonal:
        def fit(self, observed):
            np.testing.assert_array_equal(observed, np.tile([2., 5.], 9))

        def predict(self, observed, *, forecast_horizon):
            return np.resize(observed[-2:], forecast_horizon)

    evaluator = ForecastingSeriesEvaluator(
        time_series=np.r_[np.tile([2., 5.], 9), 999., 8., 4.],
        model_name='lagged_ridge_forecaster',
        forecast_horizon=2, metric_name='owa', seasonal_period=2,
        split_spec=ForecastingSplitSpec(validation_horizon=2, gap=1),
        model_factory=lambda *_args, **_kwargs: Seasonal(),
    )
    result = evaluator.run()
    assert result.forecast == (5., 2.)
    assert result.target == (8., 4.)
    assert result.metric.metric_value == pytest.approx(1.)


def test_temporal_evaluator_requires_an_explicitly_univariate_target():
    with pytest.raises(ValueError, match='single target channel'):
        ForecastingSeriesEvaluator(
            time_series=np.arange(40, dtype=float).reshape(20, 2),
            model_name='mssa_forecaster', forecast_horizon=2,
        )


def test_temporal_evaluator_does_not_flatten_multiple_forecast_channels():
    class Multi:
        def fit(self, _history):
            pass

        def predict(self, _history, *, forecast_horizon):
            return np.ones((forecast_horizon, 2))

    evaluator = ForecastingSeriesEvaluator(
        time_series=np.arange(20, dtype=float),
        model_name='lagged_ridge_forecaster', forecast_horizon=2,
        split_spec=ForecastingSplitSpec(validation_horizon=2, gap=1),
        model_factory=lambda *_args, **_kwargs: Multi(),
    )
    with pytest.raises(ValueError, match='single forecast channel'):
        evaluator.run()


@pytest.mark.parametrize('caller', ['ensemble', 'tuner'])
@pytest.mark.parametrize('legacy', [False, True])
def test_prediction_does_not_retry_a_model_type_error(caller, legacy):
    history = np.arange(20, dtype=float)
    calls = []
    original = TypeError('inside the forecasting algorithm')

    class Native:
        def fit(self, _history):
            return self

        def predict(self, _history, *, forecast_horizon=None):
            calls.append(forecast_horizon)
            if forecast_horizon is not None:
                raise original
            return history[-2:]

    class Legacy(Native):
        def predict(self, observed):
            calls.append('legacy')
            return observed[-1] + np.arange(1, 3)

    model = Legacy() if legacy else Native()

    def forecast():
        if caller == 'ensemble':
            return _safe_forecast(model, history[:-2], 2)[0]
        evaluator = ForecastingSeriesEvaluator(
            model_name='lagged_ridge_forecaster', time_series=history, forecast_horizon=2,
            split_spec=ForecastingSplitSpec(validation_horizon=2),
            model_factory=lambda *_args, **_kwargs: model,
        )
        fold, = evaluator._iter_over_folds()
        return evaluator._fit_predict_fold(fold)[0]

    if legacy:
        np.testing.assert_array_equal(forecast(), [18., 19.])
        assert calls == ['legacy']
    else:
        with pytest.raises(TypeError) as caught:
            forecast()
        assert caught.value is original
        assert calls == [2]
