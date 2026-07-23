from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F


def segmentation_torch(
    ts_size: int,
    window_size: int,
    overlapping: bool = False,
    n_segments: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Torch equivalent of ``pyts.utils.segmentation``."""

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

    if not overlapping:
        bounds = torch.linspace(
            0,
            ts_size,
            n_segments + 1,
            device=device,
        ).to(torch.int64)
        start = bounds[:-1]
        end = bounds[1:]
        return start, end, start.numel()

    n_overlapping = (n_segments * window_size) - ts_size
    n_overlaps = n_segments - 1
    overlaps = torch.linspace(
        0,
        n_overlapping,
        n_overlaps + 1,
        device=device,
    ).to(torch.int64)
    bounds = torch.arange(
        0,
        (n_segments + 1) * window_size,
        window_size,
        device=device,
    )
    start = bounds[:-1] - overlaps
    end = bounds[1:] - overlaps
    return start, end, start.numel()


class PAA:
    """Piecewise Aggregate Approximation transformer for time series."""

    def __init__(
        self,
        window_size: int,
        output_size: int,
        overlapping: bool = False,
    ):
        self.window_size = window_size
        self.output_size = output_size
        self.overlapping = overlapping

    def segmentation(self, ts_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return segment bounds for the configured PAA layout."""

        start, end, _ = segmentation_torch(
            ts_size,
            self.window_size,
            self.overlapping,
            self.output_size,
        )
        return start, end

    def _paa(
        self,
        X: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
    ) -> torch.Tensor:
        """Apply vectorized segment means."""

        cumsum = torch.cumsum(X, dim=1)
        cumsum = F.pad(cumsum, (1, 0), value=0)
        segment_sums = cumsum[:, end] - cumsum[:, start]
        segment_lengths = (end - start).unsqueeze(0)
        return segment_sums / segment_lengths

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a batch of time series into PAA segments."""

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
