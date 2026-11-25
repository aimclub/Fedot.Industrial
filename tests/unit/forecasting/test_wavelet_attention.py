import numpy as np
import pytest


@pytest.mark.parametrize("task", ["point", "direction"])
def test_joint_attention_on_aligned_univariate_windows(task):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pywt")
    from industrial_forecasting.neural.wavelet_attention import (
        WaveletAttentionSpec, cwt_view, fit_wavelet_attention,
    )

    history = np.sin(np.arange(32.) / 3)
    spec = WaveletAttentionSpec(window=8, horizon=3, scales=(1., 2., 4.), width=4, task=task)
    windows = np.stack([history[:8], history[8:16]])
    views = cwt_view(windows, spec)
    assert tuple(views[0].shape) == (2, 8, 1)
    assert tuple(views[1].shape) == (2, 2, 3, 8)
    result = fit_wavelet_attention((history,), spec, epochs=1)
    prediction = result.predict(history)
    assert prediction.shape == ((2,) if task == "direction" else (3,))
    assert np.isfinite(prediction).all()
    if task == "direction":
        assert prediction.sum() == pytest.approx(1.)
    result.model(views).sum().backward()
    assert torch.isfinite(result.model.temporal.weight_ih_l0.grad).all()
    assert torch.isfinite(result.model.spectral[0].weight.grad).all()
