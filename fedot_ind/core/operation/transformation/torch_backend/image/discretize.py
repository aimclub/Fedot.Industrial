from __future__ import annotations

from typing import Literal
import warnings

import torch

Strategy = Literal["uniform", "quantile", "normal"]


def _validate_kbins_params(n_bins: int, strategy: str) -> None:
    """Validate discretization parameters."""

    if n_bins < 2:
        raise ValueError(f"'n_bins' must be >= 2, got {n_bins}.")
    if strategy not in {"uniform", "quantile", "normal"}:
        raise ValueError(
            f"'strategy' must be one of ['uniform', 'quantile', 'normal'], got {strategy}."
        )


def _linspace_per_row(
    start: torch.Tensor,
    end: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Vectorized per-row linspace."""

    if start.ndim != 1 or end.ndim != 1:
        raise ValueError("start and end must be 1D tensors.")
    if start.shape != end.shape:
        raise ValueError("start and end must have the same shape.")
    if steps < 2:
        raise ValueError("steps must be >= 2.")

    t = torch.linspace(0, 1, steps=steps, device=start.device, dtype=start.dtype)
    return start[:, None] + (end - start)[:, None] * t[None, :]


def _uniform_bins_torch(X: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Compute uniform bin edges per sample."""

    sample_min = X.min(dim=1).values
    sample_max = X.max(dim=1).values
    edges = _linspace_per_row(sample_min, sample_max, steps=n_bins + 1)
    return edges[:, 1:-1]


def _normal_bins_torch(
    n_bins: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute bin edges from standard normal quantiles."""

    probs = torch.linspace(0, 1, steps=n_bins + 1, device=device, dtype=dtype)[1:-1]
    normal = torch.distributions.Normal(
        loc=torch.tensor(0.0, device=device, dtype=dtype),
        scale=torch.tensor(1.0, device=device, dtype=dtype),
    )
    return normal.icdf(probs)


def _pad_rows_with_nan(rows: list[torch.Tensor]) -> torch.Tensor:
    """Pad a list of 1D tensors to a 2D tensor with NaN on the right."""

    if len(rows) == 0:
        raise ValueError("rows must not be empty.")

    max_len = max(row.numel() for row in rows)
    device = rows[0].device
    dtype = rows[0].dtype

    out = torch.full((len(rows), max_len), torch.nan, device=device, dtype=dtype)
    for index, row in enumerate(rows):
        if row.numel() > 0:
            out[index, : row.numel()] = row
    return out


def _quantile_bins_torch(
    X: torch.Tensor,
    n_bins: int,
    *,
    raise_warning: bool = True,
    atol: float = 1e-8,
) -> torch.Tensor:
    """Compute quantile bin edges per sample."""

    probs = torch.linspace(
        0,
        1,
        steps=n_bins + 1,
        device=X.device,
        dtype=X.dtype,
    )[1:-1]
    edges = torch.quantile(X, probs, dim=1).transpose(0, 1)

    if n_bins <= 2:
        return edges

    diffs = torch.diff(edges, dim=1).abs()
    keep = torch.cat(
        [
            diffs > atol,
            torch.ones((X.shape[0], 1), device=X.device, dtype=torch.bool),
        ],
        dim=1,
    )

    if torch.all(keep):
        return edges

    if raise_warning:
        bad_samples = torch.where((~keep).any(dim=1))[0].tolist()
        warnings.warn(
            "Some quantiles are equal. The number of bins will be smaller for "
            f"sample {bad_samples}. Consider decreasing the number of bins "
            "or removing these samples.",
            UserWarning,
        )

    rows = [edges[index][keep[index]] for index in range(edges.shape[0])]
    return _pad_rows_with_nan(rows)


def _digitize_global_bins_torch(X: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Digitize using the same bin edges for all samples."""

    return torch.bucketize(X, bins, right=False)


def _digitize_per_sample_bins_torch(
    X: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """Digitize using per-sample bin edges."""

    return (bins[:, None, :] < X[:, :, None]).sum(dim=-1).long()


def _digitize_torch(X: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Digitize X according to bin edges."""

    if bins.ndim == 1:
        return _digitize_global_bins_torch(X, bins)
    if bins.ndim == 2:
        if bins.shape[0] != X.shape[0]:
            raise ValueError(
                "For per-sample bins, bins.shape[0] must equal X.shape[0]. "
                f"Got bins.shape={tuple(bins.shape)}, X.shape={tuple(X.shape)}"
            )
        return _digitize_per_sample_bins_torch(X, bins)

    raise ValueError(f"bins must be 1D or 2D, got shape={tuple(bins.shape)}")


def _compute_bins_torch(
    X: torch.Tensor,
    n_bins: int,
    strategy: Strategy = "quantile",
    *,
    raise_warning: bool = True,
) -> torch.Tensor:
    """Compute bin edges for discretization."""

    _validate_kbins_params(n_bins, strategy)

    if strategy == "normal":
        return _normal_bins_torch(n_bins, device=X.device, dtype=X.dtype)

    if strategy == "uniform":
        return _uniform_bins_torch(X, n_bins)

    return _quantile_bins_torch(X, n_bins, raise_warning=raise_warning)


def kbins_discretize_torch(
    X: torch.Tensor,
    n_bins: int = 5,
    strategy: Strategy = "quantile",
    *,
    raise_warning: bool = True,
    return_bins: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Discretize X into integer bins."""

    bins = _compute_bins_torch(
        X,
        n_bins=n_bins,
        strategy=strategy,
        raise_warning=raise_warning,
    )
    X_binned = _digitize_torch(X, bins)

    if return_bins:
        return X_binned, bins
    return X_binned
