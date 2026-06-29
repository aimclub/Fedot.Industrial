from typing import Any, Optional
import math
import torch

from fedot_ind.core.operation.transformation.torch_backend.image.tools import (
    MinMaxScalerTorch,
    PAA,
    prepare_series_input,
    convert_to_init_dim,
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
            scaling range before GAF encoding. If ``None``, input values must
            already lie in ``[-1, 1]``.
        return_init_dim (bool, default ``True``): If ``True``, restore batch/
            channel axes for 3D input ``(B, C, T)`` → ``(B, C, H, W)``.
            For 1D/2D inputs the output batch layout is left unchanged.
        torch_device (str, default ``'auto'``): Device to use for the transformation.
    """
    def __init__(self, params: Optional[dict[str, Any]] = None):
        params = params or {}
        self.window_size = params.get('window_size', None)
        self.sample_range = params.get('sample_range', (-1, 1))
        self.method = params.get('method', 'summation')
        self.image_size = params.get('image_size', 1.)
        self.overlapping = params.get('overlapping', False)
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
        if n_timestamps < 2:
            raise ValueError(
                f"Time series length must be >= 2 for GAF, got {n_timestamps}."
            )
        window_size, paa_output_size = self._check_params(n_timestamps)
        method = self._resolve_method()
        paa = PAA(
            window_size=window_size,
            output_size=paa_output_size,
            overlapping=self.overlapping,
        )
        X_paa = paa.transform(X)
        if self.sample_range is None:
            X_min, X_max = torch.min(X_paa), torch.max(X_paa)
            eps = 1e-5
            if (X_min < -1 - eps) or (X_max > 1 + eps):
                raise ValueError("If 'sample_range' is None, all the values "
                                 "of X must be between -1 and 1.")
            X_cos = X_paa.clamp(-1.0, 1.0)
        else:
            X_cos = MinMaxScalerTorch(X_paa, self.sample_range)
        X_sin = torch.sqrt(torch.clamp(1 - X_cos**2, min=0, max=1))

        if method == "gasf":
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

    def _resolve_method(self) -> str:
        if self.method in ("summation", "s", "gasf"):
            return "gasf"
        if self.method in ("difference", "d", "gadf"):
            return "gadf"
        raise ValueError(
            "'method' must be one of 'summation', 's', 'difference' or 'd'."
        )

    def _check_params(self, n_timestamps: int) -> tuple[int, int]:
        """
        Validates config and computes PAA ``window_size`` and ``output_size``.

        Returns:
            tuple[int, int]: ``(window_size, paa_output_size)`` for PAA.
        """
        self._resolve_method()

        if self.window_size is None:
            if isinstance(self.image_size, int):
                image_size = self.image_size
                if image_size < 1 or image_size > n_timestamps:
                    raise ValueError(
                        "If 'image_size' is an integer, it must be >= 1 "
                        "and <= n_timestamps."
                    )
            elif isinstance(self.image_size, float):
                if not (0 < self.image_size <= 1.):
                    raise ValueError(
                        "If 'image_size' is a float, it must be greater "
                        "than 0 and lower than or equal to 1 (got {0})."
                        .format(self.image_size)
                    )
                image_size = math.ceil(self.image_size * n_timestamps)
            else:
                raise ValueError(
                    "'image_size' must be either an integer or a float."
                )

            window_size, remainder = divmod(n_timestamps, image_size)
            if remainder != 0:
                window_size += 1
        else:
            if not isinstance(self.window_size, int):
                raise TypeError("'window_size' must be an integer.")
            if self.window_size < 1 or self.window_size > n_timestamps:
                raise ValueError("'window_size' must be >= 1 and <= n_timestamps.")

            window_size = self.window_size
            image_size, remainder = divmod(n_timestamps, window_size)
            if remainder != 0:
                image_size += 1

        return window_size, image_size
