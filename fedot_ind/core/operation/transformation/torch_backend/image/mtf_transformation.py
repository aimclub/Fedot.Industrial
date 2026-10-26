import math
from typing import Any, Optional

import torch

from fedot_ind.core.operation.transformation.torch_backend.image.discretize import kbins_discretize_torch
from fedot_ind.core.operation.transformation.torch_backend.image.paa import segmentation_torch
from fedot_ind.core.operation.transformation.torch_backend.image.shape_io import (
    convert_to_init_dim,
    prepare_series_input,
)
from fedot_ind.core.operation.transformation.torch_backend.rules import (
    normalize_image_size,
    validate_image_size_fits_series,
    validate_kbins_params,
    validate_min_series_length,
    validate_mtf_output_layout,
)


def _aggregate_transition_field(binned, transition, image_size, overlapping=False):
    """Average the transition field over the requested time windows."""
    n_samples, n_timestamps = binned.shape
    n_bins = transition.shape[-1]
    if image_size == n_timestamps or n_timestamps <= n_bins:
        return _dense_transition_field(binned, transition, image_size, overlapping)
    window_size, remainder = divmod(n_timestamps, image_size)
    if remainder == 0:
        start = torch.arange(image_size, device=binned.device) * window_size
        end = start + window_size
    else:
        start, end, _ = segmentation_torch(
            n_timestamps, window_size + 1, overlapping, image_size, device=binned.device
        )
    # A block mean of P[x_t, x_u] is h_block @ P @ h_block.T.
    # Prefix histograms use O(B*T*K), avoiding the O(B*T*T) field.
    indicators = transition.new_zeros((n_samples, n_timestamps, n_bins))
    indicators.scatter_(2, binned.unsqueeze(2), 1)
    prefix = torch.nn.functional.pad(indicators.cumsum(1), (0, 0, 1, 0))
    histograms = (prefix[:, end] - prefix[:, start]) / (end - start)[None, :, None]
    return torch.bmm(torch.bmm(histograms, transition), histograms.transpose(1, 2))


def _dense_transition_field(binned, transition, image_size, overlapping):
    """Keep the direct lookup when there is no worthwhile temporal reduction."""
    n_samples, n_timestamps = binned.shape
    sample_ids = torch.arange(n_samples, device=binned.device)[:, None, None]
    field = transition[sample_ids, binned[:, :, None], binned[:, None, :]]
    window_size, remainder = divmod(n_timestamps, image_size)
    if remainder == 0:
        return field.reshape(
            n_samples, image_size, window_size, image_size, window_size
        ).mean(dim=(2, 4))
    start, end, _ = segmentation_torch(
        n_timestamps, window_size + 1, overlapping, image_size, device=binned.device
    )
    integral = torch.nn.functional.pad(field.cumsum(1).cumsum(2), (1, 0, 1, 0))
    r0, r1 = start[:, None], end[:, None]
    c0, c1 = start[None, :], end[None, :]
    total = (integral[:, r1, c1] - integral[:, r0, c1]
             - integral[:, r1, c0] + integral[:, r0, c0])
    area = ((r1 - r0) * (c1 - c0)).to(transition.dtype).clamp_min(1)
    return total / area


class MTF:
    """
    Markov Transition Field (MTF) image transformation for time series.

    Config parameters (``params`` dict):
        image_size (int or float, default ``1.``): Side length of the square
            MTF image. Float values are a fraction of ``T`` in ``(0, 1]``; int
            values set the exact side length.
        n_bins (int, default ``8``): Number of discrete bins for quantization.
        strategy (str or BinningStrategy, default ``'quantile'``): Binning
            strategy — one of ``'uniform'``, ``'quantile'``, ``'normal'``.
        overlapping (bool, default ``False``): Use overlapping windows when
            downsampling the MTF image and ``T`` is not evenly divisible.
        flatten (bool, default ``False``): If ``True``, return a 1D vector of
            length ``image_size ** 2`` per sample instead of a square image.
        return_init_dim (bool, default ``True``): If ``True``, restore batch/
            channel axes for 3D input ``(B, C, T)`` → ``(B, C, H, W)``.
            For 1D/2D inputs the output batch layout is left unchanged.
            Cannot be combined with ``flatten=True``.
        torch_device (str, default ``'auto'``): Device to use for the transformation.
    """

    def __init__(self, params: Optional[dict[str, Any]] = None):
        params = params or {}
        self.image_size = normalize_image_size(params.get("image_size", 1.0))
        self.n_bins = params.get("n_bins", 8)
        self.strategy = params.get("strategy", "quantile")
        self.overlapping = params.get("overlapping", False)
        self.return_init_dim = bool(params.get("return_init_dim", True))
        self.flatten = params.get("flatten", False)
        self.torch_device = params.get("torch_device", "auto")

        validate_mtf_output_layout(
            flatten=self.flatten,
            return_init_dim=self.return_init_dim,
        )
        self.strategy = validate_kbins_params(self.n_bins, self.strategy)

    def transform(self, X: Any) -> torch.Tensor:

        X, init_shape = prepare_series_input(X, torch_device=self.torch_device)

        n_samples, n_timestamps = X.shape
        validate_min_series_length(n_timestamps, operation="MTF")
        if isinstance(self.image_size, int):
            validate_image_size_fits_series(self.image_size, n_timestamps)
            image_size = self.image_size
        else:
            image_size = math.ceil(self.image_size * n_timestamps)

        X_binned = kbins_discretize_torch(
            X, n_bins=self.n_bins, strategy=self.strategy
        ).long()

        # Build Markov transition matrix per sample.
        src = X_binned[:, :-1]
        dst = X_binned[:, 1:]
        flat_idx = src * self.n_bins + dst
        X_mtm = torch.zeros(
            (n_samples, self.n_bins * self.n_bins), device=X.device, dtype=X.dtype
        )
        X_mtm.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=X.dtype))
        X_mtm = X_mtm.view(n_samples, self.n_bins, self.n_bins)
        row_sum = X_mtm.sum(dim=2, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        X_mtm = X_mtm / row_sum

        X_amtf = _aggregate_transition_field(
            X_binned, X_mtm, image_size, self.overlapping
        )

        if self.flatten:
            return X_amtf.reshape(n_samples, -1)
        if self.return_init_dim:
            return convert_to_init_dim(X_amtf, init_shape)

        return X_amtf
