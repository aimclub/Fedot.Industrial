import torch
import math
from typing import Any, Optional
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.operation.transformation.torch_backend.image_transformation.usefull_transformations import (
    kbins_discretize_torch,
    segmentation_torch,
)


class MTF:

    image_size = 1.
    n_bins = 8
    strategy = 'quantile'
    overlapping = False
    flatten = False

    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or {}
        self.image_size = params.get('image_size', 1.)
        self.n_bins = params.get('n_bins', 8)
        self.strategy = params.get('strategy', 'quantile')
        self.overlapping = params.get('overlapping', False)
        self.flatten = params.get('flatten', False)

    def transform(self, X: Any) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={tuple(X.shape)}")
        if not torch.is_floating_point(X):
            X = X.float()

        n_samples, n_timestamps = X.shape
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
        return X_amtf
