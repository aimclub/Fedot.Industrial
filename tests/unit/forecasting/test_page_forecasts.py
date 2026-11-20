import numpy as np
import pytest

from industrial_forecasting.page_models import (
    fit_page_mssa, page_forecast_examples, page_tensor, reconstruct_tensor_page,
)


def test_columns_are_channel_major_and_target_is_not_returned_as_a_feature():
    x = np.arange(14, dtype=float).reshape(7, 2)
    prefix, target = page_forecast_examples(x, 3)
    np.testing.assert_array_equal(prefix, [[0, 2], [6, 8], [1, 3], [7, 9]])
    np.testing.assert_array_equal(target, [4, 10, 5, 11])
    x[2::3] = 999.
    np.testing.assert_array_equal(page_forecast_examples(x, 3)[0], prefix)
    assert page_tensor(x, 3).shape == (2, 3, 2)


def test_mssa_single_shared_head_recovers_rank_one_case():
    train = np.tile([1., 2., 3.], 12)[:, None] * np.array([[1., 2.]])
    reference = fit_page_mssa(train, length=3, rank=1)
    np.testing.assert_allclose(reference.predict_block_end([[4., 7.], [8., 14.]]), [12., 21.])
    permuted = fit_page_mssa(train[:, ::-1], length=3, rank=1)
    np.testing.assert_allclose(permuted.weights, reference.weights)


def test_tensor_reconstruction_does_not_erase_missingness():
    pytest.importorskip("tensorly")
    x = np.tile([1., 2., 3.], 5)[:, None] * np.array([[1., 2.]])
    x[1, 0] = np.nan
    result = reconstruct_tensor_page(x, length=3, rank=1, seed=7)
    assert not result.observed[0, 1, 0] and result.observed.sum() == 29
    assert np.isfinite(result.values).all()
    np.testing.assert_allclose(result.values[result.observed], page_tensor(x, 3)[result.observed], atol=.1)


def test_page_projection_gradients_and_batch_independence():
    torch = pytest.importorskip("torch")
    from industrial_forecasting.neural.page_projection import fit_page_projection

    x = np.sin(np.arange(84.) / 5)
    frozen = fit_page_projection(x, length=4, rank=2, learn_projection=False, epochs=2)
    learned = fit_page_projection(x, length=4, rank=2, learn_projection=True, epochs=2)
    assert not frozen.projection.requires_grad and learned.projection.requires_grad
    assert not torch.equal(frozen.projection, learned.projection)
    changed = x.copy()
    changed[3::4] += 50
    changed_target = fit_page_projection(changed, length=4, rank=2, learn_projection=False, epochs=1)
    torch.testing.assert_close(frozen.projection, changed_target.projection)
    torch.testing.assert_close(frozen.center, changed_target.center)
    torch.testing.assert_close(frozen.scale, changed_target.scale)
    windows = torch.from_numpy(page_forecast_examples(x, 4)[0].astype(np.float32))
    torch.testing.assert_close(learned(windows[:1]), learned(windows)[:1])
