import math
import torch
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.operation.transformation.torch_backend.image_transformation.usefull_transformations import MinMaxScalerTorch, PAA, segmentation_torch
from typing import Dict, Type

TRANSFORMER_REGISTRY: Dict[str, Type] = {}


def register_transformer(name: str):
    def decorator(cls):
        if name in TRANSFORMER_REGISTRY:
            raise ValueError(f"Transformer '{name}' already registered")
        TRANSFORMER_REGISTRY[name] = cls
        return cls
    return decorator



@register_transformer("gaf")
class GAF:
    """
    A PyTorch-based Gramian Angular Field (GAF) transformer for time series
    data.
    
    This class converts time series into Gramian Angular Field (GAF) images, 
    which can be used for visualizing and analyzing time series data as images. 
    The class supports two types of GAF: Gramian Angular Summation Field (GASF)
    and Gramian Angular Difference Field (GADF). It also supports batch
    processing and GPU acceleration.

    Attributes:
        window_size (int or None): The size of the sliding window for Piecewise
            Aggregate Approximation (PAA). If None, it is automatically
            calculated based on `image_size`.
        sample_range (tuple): The range for scaling the time series values.
            Defaults to (-1, 1).
        method (str): The type of GAF to compute. Must be either 'summation'
            (GASF) or 'difference' (GADF). Defaults to 'summation'.
        image_size (float): The size of the output GAF image, expressed as a
            fraction of the time series length. Must be between 0 and 1.
            Defaults to 1.0.
        overlapping (bool): If True, segments will overlap during PAA. Defaults
            to False.
    """
    def __init__(self, params: Optional[OperationParameters] = None):
        self.window_size = params.get('window_size', None)
        self.sample_range = params.get('sample_range', (-1, 1))
        self.method = params.get('method', 'summation')
        self.image_size = params.get('image_size', 1.)
        self.overlapping = params.get('overlapping', False)
        self.output_size = params.get('output_size', None)

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
        self.window_size, self.image_size = self._check_params(X.shape[1])
        paa = PAA(self.window_size, self.image_size, self.overlapping)
        X_paa = paa.transform(X)
        # X_paa = segmentation_torch(X.shape[-1], self.window_size, self.overlapping, self.image_size)
        if self.sample_range is None:
            X_min, X_max = torch.min(X_paa), torch.max(X_paa)
            if (X_min < -1) or (X_max > 1):
                raise ValueError("If 'sample_range' is None, all the values "
                                 "of X must be between -1 and 1.")
            X_cos = X_paa
        else:
            X_cos = MinMaxScalerTorch(X_paa, self.sample_range)
        X_sin = torch.sqrt(torch.clamp(1 - X_cos**2, min=0, max=1))

        if self.method in ['s', 'summation']:
            X_new = self._gasf(X_cos, X_sin)
        else:
            X_new = self._gadf(X_cos, X_sin)
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

    def _check_params(self, n_timestamps: int) -> int:
        """
        Validates and computes the window size and image size for PAA.

        Args:
            n_timestamps (int): Length of the time series.

        Returns:
            tuple: A tuple containing the computed window size and image size.
        """
        if self.window_size is not None:
            image_size = self.image_size
        else:
            if not (0 < self.image_size <= 1.):
                raise ValueError(
                    "If 'image_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {0})."
                    .format(self.image_size)
                )
            image_size = math.ceil(self.image_size * n_timestamps)
            window_size, remainder = divmod(n_timestamps, image_size)
        if remainder != 0:
            window_size += 1
        if self.method not in ['summation', 'difference']:
            raise ValueError("'method' must be either 'summation'"
                             "'difference' or 'd'.")
        return window_size, image_size
