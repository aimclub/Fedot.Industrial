import numpy as np
import pytest

from industrial_forecasting.wire import predict_positional_wire, wire_features


def test_positional_width_and_periodic_control():
    x = np.array([0., 1., 2.])
    periodic = wire_features(x, [1., 3.], 0.)
    local = wire_features(x, [1., 3.], [1., 2.])
    np.testing.assert_array_equal(periodic[0], local[0])
    assert np.linalg.norm(local[-1]) < .1 * np.linalg.norm(periodic[-1])
    for mode in ("wire", "fourier", "none"):
        np.testing.assert_allclose(predict_positional_wire(np.ones(24), 8, frequencies=[1.], widths=2., mode=mode), 1.)


def test_coordinate_models_use_histories_and_forecast_without_updates():
    torch = pytest.importorskip("torch")
    from industrial_forecasting.neural.coordinate_forecast import fit_amortized_wire, predict_segment_wire

    train = np.sin(np.arange(32.) / 8)
    local = predict_segment_wire(train, 7, seed=44, epochs=2)
    assert local.values.shape == (7,) and np.isfinite(local.history_rmse) and local.fit_seconds >= 0
    with pytest.raises(ValueError, match="finite"):
        predict_segment_wire(np.append(train, np.nan), 7)
    model = fit_amortized_wire((train, train + .5), context=8, horizon=3, seed=8, epochs=2)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    prediction = model(torch.tensor(train[-8:], dtype=torch.float32)[None])
    assert prediction.shape == (1, 3) and torch.isfinite(prediction).all()
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
