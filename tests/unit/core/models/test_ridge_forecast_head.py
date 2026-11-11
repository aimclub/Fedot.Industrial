import numpy as np
import pytest
import torch

from fedot_ind.core.models.ts_forecasting.forecasting_runtime import RidgeForecastingHead, TensorDevicePolicy


def reference(x, y, alpha):
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    design = np.vstack([xc, np.sqrt(alpha) * np.eye(x.shape[1])])
    target = np.vstack([yc, np.zeros((x.shape[1], y.shape[1]))])
    weights = np.linalg.lstsq(design, target, rcond=None)[0]
    return (x - x_mean) @ weights + y_mean


@pytest.mark.parametrize('alpha', [0.0, 1.0])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_ridge_matches_independent_augmented_lstsq_at_large_feature_offsets(alpha, dtype):
    rng = np.random.default_rng(42)
    base = rng.normal(size=(120, 2))
    x = np.column_stack([base, base[:, 0], base[:, 0] + base[:, 1]]) + 10000
    y = np.column_stack([2 * base[:, 0] - base[:, 1], base[:, 0] + 3 * base[:, 1]])
    xt = torch.tensor(x, dtype=getattr(torch, dtype))
    yt = torch.tensor(y, dtype=getattr(torch, dtype))
    x_before, y_before = xt.clone(), yt.clone()
    head = RidgeForecastingHead(alpha=alpha, device_policy=TensorDevicePolicy(device='cpu', dtype=dtype))
    actual = head.fit(xt, yt).predict(xt)
    expected = reference(xt.numpy().astype(float), yt.numpy().astype(float), alpha)
    assert actual.dtype == xt.dtype and actual.device == xt.device
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(xt, x_before)
    torch.testing.assert_close(yt, y_before)


@pytest.mark.parametrize('alpha', [0.0, 1.0])
def test_ridge_preserves_prediction_under_feature_and_target_translation(alpha):
    x = np.arange(120, dtype=float).reshape(40, 3) / 8
    y = np.column_stack([x[:, 0] * 2, x[:, 1] * -3])
    policy = TensorDevicePolicy(device='cpu', dtype='float64')
    original = RidgeForecastingHead(alpha=alpha, device_policy=policy).fit(x, y).predict(x)
    changed = RidgeForecastingHead(alpha=alpha, device_policy=policy).fit(x + 2**20, y + 128).predict(x + 2**20)
    torch.testing.assert_close(changed - 128, original, rtol=1e-8, atol=1e-8)


def test_ridge_constant_and_wide_rank_deficient_designs_have_finite_minimum_norm_predictions():
    policy = TensorDevicePolicy(device='cpu')
    for x in (np.ones((8, 4)) * 10000, np.tile(np.arange(4), (12, 1)).T):
        y = np.arange(len(x), dtype=float).reshape(-1, 1)
        head = RidgeForecastingHead(alpha=0.0, device_policy=policy).fit(x, y)
        result = head.predict(x)
        np.testing.assert_allclose(result.numpy(), reference(x, y, 0.0), atol=1e-5)
        assert torch.isfinite(result).all()


@pytest.mark.parametrize('alpha', [-1.0, float('nan'), float('inf'), True])
def test_ridge_rejects_invalid_regularization_before_training(alpha):
    with pytest.raises(ValueError, match='alpha'):
        RidgeForecastingHead(alpha=alpha).fit(np.ones((4, 2)), np.ones((4, 1)))


@pytest.mark.parametrize('x, y', [
    (np.ones((4, 2)), np.ones((3, 1))),
    (np.empty((0, 2)), np.empty((0, 1))),
    (np.full((4, 2), np.nan), np.ones((4, 1))),
    (np.ones((4, 2)), np.full((4, 1), np.inf)),
])
def test_ridge_rejects_nonfinite_empty_or_misaligned_training_data(x, y):
    with pytest.raises(ValueError):
        RidgeForecastingHead().fit(x, y)


def test_legacy_coefficients_remain_readable_without_claiming_new_solver():
    head = RidgeForecastingHead(weights_=torch.tensor([[2.0]]), intercept_=torch.tensor([3.0]),
                                input_dim_=1, output_dim_=1)
    torch.testing.assert_close(head.predict([[4.0]]), torch.tensor([[11.0]]))
    assert head.get_diagnostics()['solver'] == 'legacy_coefficients'
    assert head.get_diagnostics()['solve_dtype'] == 'float32'
