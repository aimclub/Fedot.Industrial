from typing import Any, Optional
import math
import torch

from fedot_ind.core.operation.transformation.torch_backend.image.paa import PAA
from fedot_ind.core.operation.transformation.torch_backend.image.scaling import (
    per_sample_minmax_scale,
)
from fedot_ind.core.operation.transformation.torch_backend.image.shape_io import (
    convert_to_init_dim,
    prepare_series_input,
)
from fedot_ind.core.operation.transformation.torch_backend.rules import (
    GAFMethod,
    normalize_gaf_method,
    normalize_image_size,
    normalize_optional_window_size,
    validate_gaf_input_range,
    validate_image_size_fits_series,
    validate_min_series_length,
    validate_window_size,
)


class GAF:
    """
    A PyTorch-based Gramian Angular Field (GAF) transformer for time series
    data.

    This class converts time series into Gramian Angular Field (GAF) images,
    which can be used for visualizing and analyzing time series data as images.
    The class supports two types of GAF: Gramian Angular Summation Field (GASF)
    and Gramian Angular Difference Field (GADF). It also supports batch
    processing and GPU acceleration.

    Config parameters (``params`` dict):
        image_size (int or float, default ``1.``): Side length of the square
            GAF image. Float values are treated as a fraction of ``T`` in
            ``(0, 1]``; int values set the exact side length.
        method (str, default ``'summation'``): ``'summation'``/``'s'``/``'gasf'``
            for GASF, ``'difference'``/``'d'``/``'gadf'`` for GADF.
        overlapping (bool, default ``False``): Use overlapping PAA windows when
            ``T`` is not evenly divisible by the target image side.
        window_size (int or None, default ``None``): PAA window size. When set,
            ``image_size`` is derived from ``T`` and ``window_size`` instead
            of taken directly from ``image_size``.
        sample_range (tuple or None, default ``(-1, 1)``): Per-sample min-max
            scaling range before GAF encoding. If ``None`` and
            ``use_per_sample_minmax=True``, ``(-1, 1)`` is used.
        use_per_sample_minmax (bool, default ``True``): If ``True``, apply
            per-sample min-max scaling before GAF encoding. If ``False``, input
            values must already lie in ``[-1, 1]``.
        return_init_dim (bool, default ``True``): If ``True``, restore batch/
            channel axes for 3D input ``(B, C, T)`` → ``(B, C, H, W)``.
            For 1D/2D inputs the output batch layout is left unchanged.
        torch_device (str, default ``'auto'``): Device to use for the transformation.
    """

    def __init__(self, params: Optional[dict[str, Any]] = None):
        params = params or {}
        self.window_size = normalize_optional_window_size(
            params.get("window_size", None)
        )
        self.sample_range = params.get("sample_range", (-1, 1))
        self.use_per_sample_minmax = bool(params.get("use_per_sample_minmax", True))
        self.method = normalize_gaf_method(params.get("method", "summation"))
        self.image_size = normalize_image_size(params.get("image_size", 1.0))
        self.overlapping = params.get("overlapping", False)
        self.return_init_dim = bool(params.get("return_init_dim", True))
        self.torch_device = params.get("torch_device", "auto")

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transforms a batch of time series into GAF images.

        This method first applies Piecewise Aggregate Approximation (PAA) to
        reduce the dimensionality of the time series. It then scales the values
        to the specified range and computes the GAF image using either the GASF
        or Gramian Angular Difference Field (GADF) method.

        Args:
            X (torch.Tensor): Input time series tensor of shape (batch,
                n_timestamps).

        Returns:
            torch.Tensor: GAF-transformed tensor of shape (batch, image_size,
                image_size).
        """
        X, init_shape = prepare_series_input(X, torch_device=self.torch_device)
        n_timestamps = X.shape[1]
        validate_min_series_length(n_timestamps, operation="GAF")
        window_size, paa_output_size = self._resolve_paa_layout(n_timestamps)
        paa = PAA(
            window_size=window_size,
            output_size=paa_output_size,
            overlapping=self.overlapping,
        )
        X_paa = paa.transform(X)
        if self.use_per_sample_minmax:
            feature_range = self.sample_range if self.sample_range is not None else (-1, 1)
            X_cos = per_sample_minmax_scale(X_paa, feature_range=feature_range, dim=1)
        else:
            X_min, X_max = torch.min(X_paa), torch.max(X_paa)
            validate_gaf_input_range(
                float(X_min.item()),
                float(X_max.item()),
            )
            X_cos = X_paa.clamp(-1.0, 1.0)
        X_sin = torch.sqrt(torch.clamp(1 - X_cos**2, min=0, max=1))

        if self.method is GAFMethod.gasf:
            X_new = self._gasf(X_cos, X_sin)
        else:
            X_new = self._gadf(X_cos, X_sin)

        if self.return_init_dim:
            return convert_to_init_dim(X_new, init_shape)

        return X_new

    def _gasf(self, X_cos: torch.Tensor, X_sin: torch.Tensor) -> torch.Tensor:
        """
        Computes the Gramian Angular Summation Field (GASF) for a batch of time
        series.

        GASF encodes temporal correlations using trigonometric summation.

        Args:
            X_cos (torch.Tensor): Cosine-transformed time series tensor of shape
                (batch, n_timestamps).
            X_sin (torch.Tensor): Sine-transformed time series tensor of shape
                (batch, n_timestamps).

        Returns:
            torch.Tensor: GASF image tensor of shape (batch, n_timestamps,
                                                                n_timestamps).
        """
        cos_outer = X_cos.unsqueeze(2) * X_cos.unsqueeze(1)
        sin_outer = X_sin.unsqueeze(2) * X_sin.unsqueeze(1)
        return cos_outer - sin_outer

    def _gadf(self, X_cos: torch.Tensor, X_sin: torch.Tensor) -> torch.Tensor:
        """
        Computes the Gramian Angular Difference Field (GADF) for a batch of time
        series.

        GADF encodes temporal correlations using trigonometric differences.

        Args:
            X_cos (torch.Tensor): Cosine-transformed time series tensor of shape
                (batch, n_timestamps).
            X_sin (torch.Tensor): Sine-transformed time series tensor of shape
                (batch, n_timestamps).

        Returns:
            torch.Tensor: GADF image tensor of shape (batch, n_timestamps,
                                                                n_timestamps).
        """
        sin_cos = X_sin.unsqueeze(2) * X_cos.unsqueeze(1)
        cos_sin = X_cos.unsqueeze(2) * X_sin.unsqueeze(1)
        return sin_cos - cos_sin

    def _resolve_paa_layout(self, n_timestamps: int) -> tuple[int, int]:
        """Compute PAA ``window_size`` and ``output_size`` for an input length."""

        if self.window_size is None:
            if isinstance(self.image_size, int):
                validate_image_size_fits_series(self.image_size, n_timestamps)
                image_size = self.image_size
            else:
                image_size = math.ceil(self.image_size * n_timestamps)

            window_size, remainder = divmod(n_timestamps, image_size)
            if remainder != 0:
                window_size += 1
        else:
            validate_window_size(self.window_size, n_timestamps)
            window_size = self.window_size
            image_size, remainder = divmod(n_timestamps, window_size)
            if remainder != 0:
                image_size += 1

        return window_size, image_size
