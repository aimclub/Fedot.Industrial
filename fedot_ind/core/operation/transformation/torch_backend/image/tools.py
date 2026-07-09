import torch
from torch.nn import functional as F
from typing import Any, Literal, Optional, Tuple
import warnings

from fedot_ind.core.kernel_learning import resolve_torch_device


def per_sample_minmax_scale(
    X: torch.Tensor,
    *,
    feature_range: tuple[float, float] = (-1.0, 1.0),
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    min_value = X.amin(dim=dim, keepdim=True)
    max_value = X.amax(dim=dim, keepdim=True)
    scale = (max_value - min_value).clamp_min(eps)

    low, high = feature_range
    X_scaled = (X - min_value) / scale
    X_scaled = X_scaled * (high - low) + low
    return torch.nan_to_num(X_scaled)


def check_input_shape(X: Any) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if not isinstance(X, torch.Tensor):
        try:
            X = torch.as_tensor(X)
        except (TypeError, ValueError):
            raise TypeError(f"X must be a torch.Tensor or convertible to one, got {type(X)}")

    init_shape = X.shape
    if X.ndim == 1:
        X = X.unsqueeze(0)
    elif X.ndim == 3:
        B, C, T = X.shape
        X = X.reshape(B * C, T)
    elif X.ndim > 3:
        raise ValueError(f"X must be 1D, 2D or 3D, got shape={tuple(X.shape)}")

    if not torch.is_floating_point(X):
        X = X.float()

    return X, init_shape


def prepare_series_input(
    X: Any,
    *,
    torch_device: Any = "auto",
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Convert input to float32 tensor on the resolved device."""
    resolved = resolve_torch_device(torch_device)
    X, init_shape = check_input_shape(X)
    return X.to(device=resolved, dtype=torch.float32), init_shape


def convert_to_init_dim(
    X: torch.Tensor, init_shape: tuple[int, ...]
) -> torch.Tensor:
    """Restore batch/channel axes after ``check_input_shape`` flattening.

    ``check_input_shape`` maps inputs to a 2D working tensor ``(N, T)``:

    - ``(T,)`` → ``(1, T)``, ``init_shape=(T,)``
    - ``(B, T)`` → ``(B, T)``, ``init_shape=(B, T)``
    - ``(B, C, T)`` → ``(B*C, T)``, ``init_shape=(B, C, T)``

    Transform outputs are always ``(N, *image_dims)``. This function inverts
    the channel flattening for 3D input only:

    - ``(B*C, …)`` → ``(B, C, …)`` when ``init_shape=(B, C, T)``

    For 1D ``(T,)`` and 2D ``(B, T)`` inputs the tensor is returned unchanged
    (e.g. ``(1, H, W)`` or ``(B, H, W)``), preserving the working batch axis.
    """

    if len(init_shape) == 3:
        batch, n_channels = init_shape[0], init_shape[1]
        expected = batch * n_channels
        if X.shape[0] != expected:
            raise ValueError(
                f"Batch/channel flatten mismatch: input shape {init_shape} "
                f"implies {expected} flat samples, got {X.shape[0]}."
            )
        return X.reshape(batch, n_channels, *X.shape[1:])
    else:
        return X


class PAA:
    """
    A PyTorch-based Piecewise Aggregate Approximation (PAA) transformer for time
    series data.

    This class reduces the dimensionality of time series by dividing them into
    segments and computing the mean value for each segment. It supports both
    non-overlapping and overlapping windows, as well as batch processing and GPU
    acceleration.

    Attributes:
        window_size (int or float): It specifies the number of time steps per
            segment.
        output_size (int or float): It specifies the exact number of segments.
        overlapping (bool): If True, segments will overlap. Defaults to False.
    """

    def __init__(self,
                 window_size: int,
                 output_size: int,
                 overlapping: bool = False):
        self.window_size = window_size
        self.output_size = output_size
        self.overlapping = overlapping

    def segmentation(self, ts_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the start and end indices for each segment of the time series.

        This method divides the time series into segments based on the specified
        `window_size` and `overlapping`. It returns the start and end indices
        for each segment, which can be used to extract the segments.

        Args:
            ts_size (int): Length of the time series.

        Returns:
            tuple: A tuple containing two tensors:
                - start (torch.Tensor): Start indices for each segment.
                - end (torch.Tensor): End indices for each segment.
        """

        quotient, remainder = divmod(ts_size, self.window_size)
        n_segments = quotient if remainder == 0 else quotient + 1

        if not self.overlapping:
            bounds = torch.linspace(0, ts_size, n_segments + 1).to(torch.long)
            start = bounds[:-1]
            end = bounds[1:]
            return start, end
        else:
            n_overlapping = (n_segments * self.window_size) - ts_size
            n_overlaps = n_segments - 1
            overlaps = torch.linspace(0,
                                      n_overlapping,
                                      n_overlaps + 1).to(torch.long)
            bounds = torch.arange(0,
                                  (n_segments + 1) * self.window_size,
                                  self.window_size)
            start = bounds[:-1] - overlaps
            end = bounds[1:] - overlaps
            return start, end

    def _paa(self, X: torch.Tensor,
             start: torch.Tensor,
             end: torch.Tensor) -> torch.Tensor:
        """
        Applies Piecewise Aggregate Approximation to a batch of time series
        using vectorized operations.

        This method computes the mean value for each segment of the time series,
        as defined by the `start` and `end` indices. It uses cumulative sums and
        vectorized operations for efficiency, avoiding explicit loops.

        Args:
            X (torch.Tensor): Input time series tensor of shape (batch,
                                                                  n_timestamps).
            start (torch.Tensor): Start indices for each segment.
            end (torch.Tensor): End indices for each segment.

        Returns:
            torch.Tensor: PAA-transformed tensor of shape (batch, n_segments).
        """
        cumsum = torch.cumsum(X, dim=1)
        cumsum = F.pad(cumsum, (1, 0), value=0)
        segment_sums = cumsum[:, end] - cumsum[:, start]

        segment_lengths = (end - start).unsqueeze(0)

        X_paa = segment_sums / segment_lengths

        return X_paa

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms a batch of time series using Piecewise Aggregate
        Approximation.

        This method applies PAA to each time series in the batch, reducing their
        dimensionality. It uses the `segmentation` method to compute segment
        indices and the `_paa` method to compute the mean values.

        Args:
            x (torch.Tensor): Input time series tensor of shape (batch,
                                                                  n_timestamps).

        Returns:
            torch.Tensor: PAA-transformed tensor of shape (batch, n_segments).
        """
        if self.window_size == 1:
            return x

        start, end, _ = segmentation_torch(
            x.shape[-1],
            self.window_size,
            self.overlapping,
            self.output_size,
            device=x.device,
        )
        return self._paa(x, start, end)


def segmentation_torch(
    ts_size: int,
    window_size: int,
    overlapping: bool = False,
    n_segments: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Torch equivalent of pyts.utils.segmentation
    """

    if device is None:
        device = torch.device("cpu")

    if ts_size < 2:
        raise ValueError("'ts_size' must be >= 2.")
    if window_size < 1:
        raise ValueError("'window_size' must be >= 1.")
    if window_size > ts_size:
        raise ValueError("'window_size' must be <= ts_size.")

    if n_segments is not None:
        if n_segments < 2:
            raise ValueError("'n_segments' must be >= 2.")
        if n_segments > ts_size:
            raise ValueError("'n_segments' must be <= ts_size.")
    else:
        quotient, remainder = divmod(ts_size, window_size)
        n_segments = quotient if remainder == 0 else quotient + 1

    # no overlapping
    if not overlapping:
        bounds = torch.linspace(
            0, ts_size, n_segments + 1, device=device
        ).to(torch.int64)

        start = bounds[:-1]
        end = bounds[1:]
        size = start.numel()
        return start, end, size

    # with overlapping
    else:
        n_overlapping = (n_segments * window_size) - ts_size
        n_overlaps = n_segments - 1

        overlaps = torch.linspace(
            0,
            n_overlapping,
            n_overlaps + 1,
            device=device
        ).to(torch.int64)

        bounds = torch.arange(
            0,
            (n_segments + 1) * window_size,
            window_size,
            device=device
        )

        start = bounds[:-1] - overlaps
        end = bounds[1:] - overlaps
        size = start.numel()
        return start, end, size


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
    """Vectorized per-row linspace.

    Parameters
    ----------
    start : torch.Tensor
        Shape (n_samples,)
    end : torch.Tensor
        Shape (n_samples,)
    steps : int
        Number of points in each row.

    Returns
    -------
    torch.Tensor
        Shape (n_samples, steps)
    """
    if start.ndim != 1 or end.ndim != 1:
        raise ValueError("start and end must be 1D tensors.")
    if start.shape != end.shape:
        raise ValueError("start and end must have the same shape.")
    if steps < 2:
        raise ValueError("steps must be >= 2.")

    t = torch.linspace(
        0, 1, steps=steps, device=start.device, dtype=start.dtype
    )  # (steps,)
    return start[:, None] + (end - start)[:, None] * t[None, :]


def _uniform_bins_torch(X: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Compute uniform bin edges per sample.

    Equivalent to the original `_uniform_bins`, but vectorized in torch.

    Parameters
    ----------
    X : torch.Tensor
        Shape (n_samples, n_timestamps)
    n_bins : int

    Returns
    -------
    torch.Tensor
        Shape (n_samples, n_bins - 1)
    """
    sample_min = X.min(dim=1).values
    sample_max = X.max(dim=1).values
    # Need n_bins + 1 points, then drop first and last -> internal edges
    edges = _linspace_per_row(sample_min, sample_max, steps=n_bins + 1)
    return edges[:, 1:-1]


def _normal_bins_torch(
    n_bins: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute bin edges from standard normal quantiles.

    Returns
    -------
    torch.Tensor
        Shape (n_bins - 1,)
    """
    probs = torch.linspace(0, 1, steps=n_bins + 1, device=device, dtype=dtype)[1:-1]
    normal = torch.distributions.Normal(
        loc=torch.tensor(0.0, device=device, dtype=dtype),
        scale=torch.tensor(1.0, device=device, dtype=dtype),
    )
    return normal.icdf(probs)


def _pad_rows_with_nan(rows: list[torch.Tensor]) -> torch.Tensor:
    """Pad a list of 1D tensors to a 2D tensor with NaN on the right.

    Parameters
    ----------
    rows : list[torch.Tensor]
        List of 1D sorted tensors with variable lengths.

    Returns
    -------
    torch.Tensor
        Shape (len(rows), max_len)
    """
    if len(rows) == 0:
        raise ValueError("rows must not be empty.")

    max_len = max(r.numel() for r in rows)
    device = rows[0].device
    dtype = rows[0].dtype

    out = torch.full((len(rows), max_len), torch.nan, device=device, dtype=dtype)
    for i, row in enumerate(rows):
        if row.numel() > 0:
            out[i, : row.numel()] = row
    return out


def _quantile_bins_torch(
    X: torch.Tensor,
    n_bins: int,
    *,
    raise_warning: bool = True,
    atol: float = 1e-8,
) -> torch.Tensor:
    """Compute quantile bin edges per sample.

    Reproduces pyts behavior:
    - compute internal quantiles
    - remove repeated edges (degenerate bins)
    - if resulting number of edges differs across samples, pad with NaN

    Parameters
    ----------
    X : torch.Tensor
        Shape (n_samples, n_timestamps)
    n_bins : int
    raise_warning : bool, default=True
    atol : float, default=1e-8

    Returns
    -------
    torch.Tensor
        If all rows have same number of valid edges:
            shape (n_samples, n_bins - 1)
        else:
            shape (n_samples, max_valid_edges), padded with NaN
    """
    probs = torch.linspace(
        0, 1, steps=n_bins + 1, device=X.device, dtype=X.dtype
    )[1:-1]  # (n_bins - 1,)

    # torch.quantile over dim=1 returns shape (n_quantiles, n_samples)
    edges = torch.quantile(X, probs, dim=1).transpose(0, 1)  # (n_samples, n_bins - 1)

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

    rows = [edges[i][keep[i]] for i in range(edges.shape[0])]
    return _pad_rows_with_nan(rows)


def _digitize_global_bins_torch(X: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Digitize using the same bin edges for all samples.

    Parameters
    ----------
    X : torch.Tensor
        Shape (n_samples, n_timestamps)
    bins : torch.Tensor
        Shape (n_edges,)

    Returns
    -------
    torch.Tensor
        Shape (n_samples, n_timestamps), dtype=torch.long
    """
    return torch.bucketize(X, bins, right=False)


def _digitize_per_sample_bins_torch(X: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Digitize using per-sample bin edges.

    Supports NaN-padded bin matrices.

    Parameters
    ----------
    X : torch.Tensor
        Shape (n_samples, n_timestamps)
    bins : torch.Tensor
        Shape (n_samples, max_n_edges), sorted in each row, NaN padded on the right.

    Returns
    -------
    torch.Tensor
        Shape (n_samples, n_timestamps), dtype=torch.long

    Notes
    -----
    We mimic np.searchsorted(..., side='left'):
        output = number of edges strictly less than x

    Since padded NaNs satisfy (NaN < x) == False, they are ignored automatically.
    """
    # X:    (N, T)
    # bins: (N, B)
    # compare -> (N, T, B)
    return (bins[:, None, :] < X[:, :, None]).sum(dim=-1).long()


def _digitize_torch(X: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Digitize X according to bin edges.

    Parameters
    ----------
    X : torch.Tensor
        Shape (n_samples, n_timestamps)
    bins : torch.Tensor
        Either:
        - shape (n_edges,) for shared bins
        - shape (n_samples, max_n_edges) for per-sample bins

    Returns
    -------
    torch.Tensor
        Shape (n_samples, n_timestamps), dtype=torch.long
    """
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
    """Compute bin edges for discretization.

    Parameters
    ----------
    X : torch.Tensor
        Shape (n_samples, n_timestamps)
    n_bins : int
    strategy : {'uniform', 'quantile', 'normal'}, default='quantile'
    raise_warning : bool, default=True

    Returns
    -------
    torch.Tensor
        Bin edges. Shape depends on strategy:
        - normal   -> (n_bins - 1,)
        - uniform  -> (n_samples, n_bins - 1)
        - quantile -> (n_samples, k) or NaN-padded (n_samples, max_k)
    """
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
    """Discretize X into integer bins.

    Pure functional replacement for `KBinsDiscretizer.transform`.

    Parameters
    ----------
    X : torch.Tensor
        Shape (n_samples, n_timestamps)
    n_bins : int, default=5
    strategy : {'uniform', 'quantile', 'normal'}, default='quantile'
    raise_warning : bool, default=True
    return_bins : bool, default=False
        If True, return both discretized X and computed bin edges.

    Returns
    -------
    X_binned : torch.Tensor
        Shape (n_samples, n_timestamps), dtype=torch.long
    bins : torch.Tensor, optional
        Computed bin edges.
    """
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
