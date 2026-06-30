import math
from typing import Any, Optional

import torch

from fedot_ind.core.operation.transformation.torch_backend.image.tools import (
    _validate_kbins_params,
    kbins_discretize_torch,
    segmentation_torch,
    prepare_series_input,
    convert_to_init_dim,
)


class MTF:
    """
    Markov Transition Field (MTF) image transformation for time series.

    Config parameters (``params`` dict):
        image_size (int or float, default ``1.``): Side length of the square
            MTF image. Float values are a fraction of ``T`` in ``(0, 1]``; int
            values set the exact side length.
        n_bins (int, default ``8``): Number of discrete bins for quantization.
        strategy (str, default ``'quantile'``): Binning strategy — one of
            ``'uniform'``, ``'quantile'``, ``'normal'``.
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
        self.image_size = params.get("image_size", 1.0)
        self.n_bins = params.get("n_bins", 8)
        self.strategy = params.get("strategy", "quantile")
        self.overlapping = params.get("overlapping", False)
        self.return_init_dim = bool(params.get("return_init_dim", True))
        self.flatten = params.get("flatten", False)
        self.torch_device = params.get("torch_device", "auto")

        if self.flatten and self.return_init_dim:
            raise ValueError(
                "'flatten' and 'return_init_dim' cannot both be True."
            )
        _validate_kbins_params(self.n_bins, self.strategy)

    def transform(self, X: Any) -> torch.Tensor:

        X, init_shape = prepare_series_input(X, torch_device=self.torch_device)

        n_samples, n_timestamps = X.shape
        if n_timestamps < 2:
            raise ValueError(
                f"Time series length must be >= 2 for MTF, got {n_timestamps}."
            )
        if isinstance(self.image_size, int):
            image_size = self.image_size
            if image_size < 1 or image_size > n_timestamps:
                raise ValueError(
                    "If 'image_size' is an integer, it must be >= 1 and <= n_timestamps."
                )
        elif isinstance(self.image_size, float):
            if self.image_size <= 0.0 or self.image_size > 1.0:
                raise ValueError(
                    "If 'image_size' is a float, it must be > 0 and <= 1."
                )
            image_size = math.ceil(self.image_size * n_timestamps)
        else:
            raise TypeError("'image_size' must be int or float.")

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

        # Expand transition matrix back to MTF image.
        sample_ids = torch.arange(n_samples, device=X.device)[:, None, None]
        X_mtf = X_mtm[sample_ids, X_binned[:, :, None], X_binned[:, None, :]]

        window_size, remainder = divmod(n_timestamps, image_size)
        if remainder == 0:
            X_amtf = X_mtf.reshape(
                n_samples, image_size, window_size, image_size, window_size
            ).mean(dim=(2, 4))
        else:
            window_size += 1
            start, end, _ = segmentation_torch(
                n_timestamps, window_size, self.overlapping, image_size, device=X.device
            )

            # Rectangle means via integral image: O(N * image_size^2).
            integral = X_mtf.cumsum(dim=1).cumsum(dim=2)
            integral = torch.nn.functional.pad(integral, (1, 0, 1, 0), value=0)
            r0, r1 = start[:, None], end[:, None]
            c0, c1 = start[None, :], end[None, :]
            block_sum = (
                integral[:, r1, c1]
                - integral[:, r0, c1]
                - integral[:, r1, c0]
                + integral[:, r0, c0]
            )
            area = ((r1 - r0) * (c1 - c0)).to(X.dtype).clamp_min(1)
            X_amtf = block_sum / area

        if self.flatten:
            return X_amtf.reshape(n_samples, -1)
        if self.return_init_dim:
            return convert_to_init_dim(X_amtf, init_shape)

        return X_amtf
