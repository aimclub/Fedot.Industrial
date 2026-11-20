"""Page-matrix forecast reference and masked tensor reconstruction."""

from dataclasses import dataclass

import numpy as np

from .grid import positive_count


def _matrix(series):
    values = np.asarray(series, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not all(values.shape) or np.isinf(values).any():
        raise ValueError("Expected (time, channels) values with NaN as the only missing marker.")
    return values


def page_tensor(series, length):
    """Return (channels, within_block, block), dropping any incomplete suffix."""
    values = _matrix(series)
    length = positive_count(length, "length")
    count = len(values) // length
    if length < 2 or not count:
        raise ValueError("At least one complete block of length >= 2 is required.")
    return values[:count * length].reshape(count, length, -1).transpose(2, 1, 0)


def page_forecast_examples(series, length):
    tensor = page_tensor(series, length)
    columns = tensor.transpose(0, 2, 1).reshape(-1, length)
    return columns[:, :-1].copy(), columns[:, -1].copy()


def _observed_projection(features, rank):
    observed = np.isfinite(features)
    fraction = observed.mean()
    if fraction <= 0:
        raise ValueError("Cannot fit a subspace without observed inputs.")
    rank = positive_count(rank, "rank")
    if rank > min(features.shape):
        raise ValueError("Rank exceeds the Page input dimensions.")
    u, s, vt = np.linalg.svd(np.nan_to_num(features, nan=0.), full_matrices=False)
    return ((u[:, :rank] * s[:rank]) @ vt[:rank] / fraction, vt[:rank].T, fraction)


@dataclass(frozen=True, eq=False)
class PageMSSA:
    weights: np.ndarray
    fraction: float

    def predict_block_end(self, available):
        rows = _matrix(available)
        if len(rows) != len(self.weights):
            raise ValueError("Provide exactly L-1 known rows for a block-end prediction.")
        return np.nan_to_num(rows, nan=0.).T @ self.weights / self.fraction


def fit_page_mssa(series, *, length, rank):
    """Fit one head across channels. Only Page prefixes enter the SVD."""
    features, target = page_forecast_examples(series, length)
    denoised, _, fraction = _observed_projection(features, rank)
    if not np.isfinite(target).any():
        raise ValueError("At least one training target must be observed.")
    weights, *_ = np.linalg.lstsq(denoised, np.nan_to_num(target, nan=0.) / fraction, rcond=None)
    return PageMSSA(weights, fraction)


@dataclass(frozen=True, eq=False)
class TensorPageReconstruction:
    values: np.ndarray
    observed: np.ndarray
    errors: tuple[float, ...]


def reconstruct_tensor_page(series, *, length, rank, iterations=30, seed=0):
    """Masked CP-ALS reconstruction only; does not implement tSSA forecasting."""
    from tensorly.cp_tensor import cp_to_tensor
    from tensorly.decomposition import parafac

    values = page_tensor(series, length)
    observed = np.isfinite(values)
    rank = positive_count(rank, "rank")
    if not observed.any() or not all(observed.any(axis=axis).all() for axis in (0, 1, 2)):
        raise ValueError("Every mode must have observations along its fibres.")
    model, errors = parafac(
        np.nan_to_num(values, nan=0.), rank=rank, n_iter_max=positive_count(iterations, "iterations"),
        init="random", mask=observed, random_state=seed, return_errors=True, l2_reg=1e-8,
    )
    return TensorPageReconstruction(cp_to_tensor(model), observed, tuple(map(float, errors)))
