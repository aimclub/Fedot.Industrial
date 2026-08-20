"""PyTorch estimators matching the PyRiemann SPD-estimation interfaces."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import torch
from pyriemann.estimation import Covariances as PyRiemannCovariances

ArrayLike = Union[np.ndarray, torch.Tensor]


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Return the symmetric part of a batch of real square matrices."""
    return (matrix + matrix.transpose(-1, -2)) * 0.5


class TorchSPDBatch(ABC):
    """Validated batch of SPD matrices with an explicit storage structure."""

    structure: Literal["dense", "uniform_blocks", "ragged_blocks"]

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Return the number of SPD matrices in the batch."""

    @property
    @abstractmethod
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """Return channel dimensions of blocks in their feature order."""

    @abstractmethod
    def to_dense(self) -> torch.Tensor:
        """Materialise the batch as dense block-diagonal SPD matrices."""


def _validate_spd_tensor(matrices: torch.Tensor, name: str) -> torch.Tensor:
    """Validate a real floating-point tensor with square trailing matrix axes."""
    tensor = torch.as_tensor(matrices)
    if tensor.ndim < 2 or tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(f"{name} must contain square matrices on its last two axes.")
    if not torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise TypeError(f"{name} must be a real floating-point tensor.")
    return tensor


class DenseSPDBatch(TorchSPDBatch):
    """Batch of dense SPD matrices with shape ``(samples, channels, channels)``."""

    structure = "dense"

    def __init__(self, matrices: torch.Tensor):
        """Store validated dense SPD matrices."""
        self.matrices = _validate_spd_tensor(matrices, "matrices")
        if self.matrices.ndim != 3:
            raise ValueError("matrices must have shape (n_samples, n_channels, n_channels).")

    @property
    def n_samples(self) -> int:
        """Return the number of dense SPD matrices."""
        return self.matrices.shape[0]

    @property
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """Return the single full-matrix block shape."""
        size = self.matrices.shape[-1]
        return ((size, size),)

    def to_dense(self) -> torch.Tensor:
        """Return dense matrices without copying them."""
        return self.matrices


class UniformBlockSPDBatch(TorchSPDBatch):
    """Batch of equally sized SPD blocks with shape ``(samples, blocks, C, C)``."""

    structure = "uniform_blocks"

    def __init__(self, matrices: torch.Tensor):
        """Store validated uniformly sized SPD blocks."""
        self.matrices = _validate_spd_tensor(matrices, "matrices")
        if self.matrices.ndim != 4:
            raise ValueError("matrices must have shape (n_samples, n_blocks, n_channels, n_channels).")

    @property
    def n_samples(self) -> int:
        """Return the number of samples represented by all blocks."""
        return self.matrices.shape[0]

    @property
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """Return one equal shape for every block."""
        size = self.matrices.shape[-1]
        return tuple((size, size) for _ in range(self.matrices.shape[1]))

    def to_dense(self) -> torch.Tensor:
        """Materialise a dense block-diagonal matrix for every sample."""
        n_samples, n_blocks, size, _ = self.matrices.shape
        output = self.matrices.new_zeros((n_samples, n_blocks * size, n_blocks * size))
        for block_index in range(n_blocks):
            start = block_index * size
            output[:, start:start + size, start:start + size] = self.matrices[:, block_index]
        return output


class RaggedBlockSPDBatch(TorchSPDBatch):
    """Batch of SPD blocks with independent channel dimensions."""

    structure = "ragged_blocks"

    def __init__(self, matrices: tuple[torch.Tensor, ...]):
        """Store validated blocks while checking their common sample dimension."""
        if not matrices:
            raise ValueError("matrices must contain at least one block.")
        self.matrices = tuple(
            _validate_spd_tensor(block, f"matrices[{index}]")
            for index, block in enumerate(matrices)
        )
        if any(block.ndim != 3 for block in self.matrices):
            raise ValueError("Each block must have shape (n_samples, n_channels, n_channels).")
        n_samples = self.matrices[0].shape[0]
        if any(block.shape[0] != n_samples for block in self.matrices[1:]):
            raise ValueError("All blocks must contain the same number of samples.")

    @property
    def n_samples(self) -> int:
        """Return the common number of samples across blocks."""
        return self.matrices[0].shape[0]

    @property
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """Return each block shape in its original channel order."""
        return tuple((block.shape[-1], block.shape[-1]) for block in self.matrices)

    def to_dense(self) -> torch.Tensor:
        """Materialise dense block-diagonal SPD matrices for compatibility paths."""
        total_channels = sum(block.shape[-1] for block in self.matrices)
        first = self.matrices[0]
        output = first.new_zeros((self.n_samples, total_channels, total_channels))
        offset = 0
        for block in self.matrices:
            size = block.shape[-1]
            output[:, offset:offset + size, offset:offset + size] = block
            offset += size
        return output


class TorchSPDReference(ABC):
    """Validated centroid whose layout matches a :class:`TorchSPDBatch`."""

    structure: Literal["dense", "uniform_blocks", "ragged_blocks"]

    @property
    @abstractmethod
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """Return the reference block dimensions in feature order."""

    @abstractmethod
    def to_dense(self) -> torch.Tensor:
        """Materialise the reference as one dense block-diagonal matrix."""


class DenseSPDReference(TorchSPDReference):
    """Reference point for a batch of dense SPD matrices."""

    structure = "dense"

    def __init__(self, matrices: torch.Tensor):
        """Store one validated dense SPD reference matrix."""
        self.matrices = _validate_spd_tensor(matrices, "matrices")
        if self.matrices.ndim != 2:
            raise ValueError("matrices must have shape (n_channels, n_channels).")

    @property
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """Return the full reference matrix shape."""
        size = self.matrices.shape[-1]
        return ((size, size),)

    def to_dense(self) -> torch.Tensor:
        """Return the dense reference without copying it."""
        return self.matrices


class UniformBlockSPDReference(TorchSPDReference):
    """Reference point for equally sized SPD blocks."""

    structure = "uniform_blocks"

    def __init__(self, matrices: torch.Tensor):
        """Store validated reference blocks with a shared channel dimension."""
        self.matrices = _validate_spd_tensor(matrices, "matrices")
        if self.matrices.ndim != 3:
            raise ValueError("matrices must have shape (n_blocks, n_channels, n_channels).")

    @property
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """Return the shape of every uniform reference block."""
        size = self.matrices.shape[-1]
        return tuple((size, size) for _ in range(self.matrices.shape[0]))

    def to_dense(self) -> torch.Tensor:
        """Materialise one dense block-diagonal reference matrix."""
        return UniformBlockSPDBatch(self.matrices.unsqueeze(0)).to_dense()[0]


class RaggedBlockSPDReference(TorchSPDReference):
    """Reference point for SPD blocks with different channel dimensions."""

    structure = "ragged_blocks"

    def __init__(self, matrices: tuple[torch.Tensor, ...]):
        """Store validated reference blocks."""
        if not matrices:
            raise ValueError("matrices must contain at least one block.")
        self.matrices = tuple(
            _validate_spd_tensor(block, f"matrices[{index}]")
            for index, block in enumerate(matrices)
        )
        if any(block.ndim != 2 for block in self.matrices):
            raise ValueError("Each reference block must have shape (n_channels, n_channels).")

    @property
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """Return reference block shapes in their original order."""
        return tuple((block.shape[-1], block.shape[-1]) for block in self.matrices)

    def to_dense(self) -> torch.Tensor:
        """Materialise one dense block-diagonal reference matrix."""
        return RaggedBlockSPDBatch(tuple(block.unsqueeze(0) for block in self.matrices)).to_dense()[0]


class TorchShrinkage:
    """PyTorch counterpart of PyRiemann's global SPD shrinkage transformer."""

    def __init__(self, shrinkage: float = 0.1):
        """Initialise the coefficient of the identity-matrix convex combination."""
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be between 0 and 1.")
        self.shrinkage = float(shrinkage)

    def fit(self, X: TorchSPDBatch, y=None) -> "TorchShrinkage":
        """Validate a structured SPD batch; shrinkage has no fitted state."""
        del y
        if not isinstance(X, TorchSPDBatch):
            raise TypeError("X must be a validated TorchSPDBatch.")
        return self

    def _shrink_block(self, matrices: torch.Tensor, trace_mean: torch.Tensor) -> torch.Tensor:
        """Shrink one batch of equal-sized matrices using a supplied trace mean."""
        size = matrices.shape[-1]
        identity = torch.eye(size, dtype=matrices.dtype, device=matrices.device)
        trace_shape = trace_mean.shape + (1,) * (matrices.ndim - trace_mean.ndim)
        return (1.0 - self.shrinkage) * matrices + (
            self.shrinkage * trace_mean.reshape(trace_shape) * identity
        )

    def transform(self, X: TorchSPDBatch) -> TorchSPDBatch:
        """Shrink matrices while preserving the validated input block structure."""
        self.fit(X)
        if isinstance(X, DenseSPDBatch):
            trace_mean = torch.diagonal(X.matrices, dim1=-2, dim2=-1).mean(dim=-1)
            return DenseSPDBatch(self._shrink_block(X.matrices, trace_mean))
        if isinstance(X, UniformBlockSPDBatch):
            n_channels = X.matrices.shape[1] * X.matrices.shape[-1]
            trace_mean = torch.diagonal(X.matrices, dim1=-2, dim2=-1).sum(dim=(-1, -2))
            trace_mean = trace_mean / n_channels
            return UniformBlockSPDBatch(self._shrink_block(X.matrices, trace_mean))
        if isinstance(X, RaggedBlockSPDBatch):
            n_channels = sum(block.shape[-1] for block in X.matrices)
            total_trace = sum(
                torch.diagonal(block, dim1=-2, dim2=-1).sum(dim=-1) for block in X.matrices
            )
            trace_mean = total_trace / n_channels
            return RaggedBlockSPDBatch(
                tuple(self._shrink_block(block, trace_mean) for block in X.matrices)
            )
        raise TypeError(f"Unsupported SPD batch type: {type(X).__name__}.")

    def fit_transform(self, X: TorchSPDBatch, y=None) -> TorchSPDBatch:
        """Validate and shrink a structured SPD batch in one call."""
        return self.fit(X, y=y).transform(X)


class BaseTorchSPDBuilder(ABC):
    """Common interface for Torch transformers that construct SPD batches."""

    @abstractmethod
    def fit(self, X: torch.Tensor, y=None) -> "BaseTorchSPDBuilder":
        """Validate or fit a transformer on input signals."""

    @abstractmethod
    def transform_spd(self, X: torch.Tensor) -> TorchSPDBatch:
        """Build a validated structured SPD batch without losing its layout."""

    def fit_transform_spd(self, X: torch.Tensor, y=None) -> TorchSPDBatch:
        """Fit and build a structured SPD batch in one call."""
        return self.fit(X, y=y).transform_spd(X)


def cospectrum_torch(
    X: ArrayLike,
    window: int = 128,
    overlap: float = 0.75,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    fs: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:

    """Estimate co-spectral matrices for an entire batch of signals.

    The result matches ``pyriemann.utils.covariance.cospectrum`` for every
    sample, while windowing, FFT and spectral products are vectorised across
    the leading sample dimension. Input shape is ``(samples, channels, time)``.
    """

    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X)
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, n_channels, n_times).")
    if torch.is_complex(X):
        raise ValueError("Input must be real-valued.")

    window = int(window)
    if window < 1:
        raise ValueError("Value window must be a positive integer")
    if not 0 < overlap < 1:
        raise ValueError(f"Value overlap must be included in (0, 1) (Got {overlap})")

    _, _, n_times = X.shape
    step = int((1.0 - overlap) * window)
    n_windows = int((n_times - window) / step + 1)
    if n_windows < 1:
        raise ValueError("The input time dimension must be at least `window`.")

    # NumPy FFT, used by PyRiemann, operates in float64 for real-valued input.
    X = X.to(dtype=torch.float64)
    win = torch.hann_window(window, periodic=False, dtype=X.dtype, device=X.device)
    windows = X.unfold(dimension=-1, size=window, step=step)
    fdata = torch.fft.rfft(windows * win, n=window, dim=-1).permute(0, 2, 1, 3)

    freqs: Optional[np.ndarray]
    if fs is not None:
        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        if fmax <= fmin:
            raise ValueError("Parameter fmax must be superior to fmin")
        if 2.0 * fmax > fs:
            raise ValueError("Parameter fmax must be inferior to fs/2")
        all_freqs = np.arange(window // 2 + 1, dtype=int) * float(fs / window)
        mask = (all_freqs >= fmin) & (all_freqs <= fmax)
        fdata = fdata[..., torch.as_tensor(mask, device=X.device)]
        freqs = all_freqs[mask]
    else:
        freqs = None

    spectrum = torch.einsum("nwcf,nwdf->ncdf", fdata.conj(), fdata)
    spectrum /= n_windows * torch.linalg.vector_norm(win).square()
    if window % 2:
        spectrum[..., 1:] *= 2
    else:
        spectrum[..., 1:-1] *= 2

    return spectrum.real, freqs


def _as_real_signal_tensor(X: torch.Tensor) -> torch.Tensor:
    """Validate signals with shape ``(samples, channels, time)``."""
    tensor = torch.as_tensor(X)
    if tensor.ndim != 3:
        raise ValueError("X must have shape (n_matrices, n_channels, n_times).")
    if not torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise TypeError("X must be a real floating-point tensor.")
    if tensor.shape[-1] < 2:
        raise ValueError("X must contain at least two time samples.")
    return tensor


def _center(X: torch.Tensor, assume_centered: bool) -> torch.Tensor:
    """Centre each signal unless its mean is declared to be zero."""
    return X if assume_centered else X - X.mean(dim=-1, keepdim=True)


def _empirical_covariance(X: torch.Tensor, *, assume_centered: bool, denominator: int) -> torch.Tensor:
    """Calculate real-valued batched empirical covariance matrices."""
    centred = _center(X, assume_centered)
    return symmetrize(centred @ centred.transpose(-1, -2) / denominator)


def _covariance(X: torch.Tensor, **kwds: Any) -> torch.Tensor:
    """Implement the unweighted real-valued ``numpy.cov`` estimator."""
    bias = bool(kwds.pop("bias", False))
    ddof = kwds.pop("ddof", None)
    fweights = kwds.pop("fweights", None)
    aweights = kwds.pop("aweights", None)
    if kwds:
        raise TypeError(f"Unsupported covariance keyword arguments: {sorted(kwds)}.")
    if fweights is not None or aweights is not None:
        raise NotImplementedError("Weighted covariance is not implemented in the Torch estimator.")

    denominator = X.shape[-1] if bias else X.shape[-1] - 1
    if ddof is not None:
        denominator = X.shape[-1] - int(ddof)
    if denominator <= 0:
        raise ValueError("Degrees of freedom must be strictly positive.")
    return _empirical_covariance(X, assume_centered=False, denominator=denominator)


def _correlation(X: torch.Tensor, **kwds: Any) -> torch.Tensor:
    """Implement the unweighted real-valued ``numpy.corrcoef`` estimator."""
    bias = kwds.pop("bias", None)
    ddof = kwds.pop("ddof", None)
    if kwds:
        raise TypeError(f"Unsupported correlation keyword arguments: {sorted(kwds)}.")
    # NumPy accepts these deprecated arguments but ignores them in corrcoef.
    del bias, ddof
    centred = X - X.mean(dim=-1, keepdim=True)
    normalised = centred / torch.linalg.vector_norm(centred, dim=-1, keepdim=True)
    return symmetrize(normalised @ normalised.transpose(-1, -2))


def _scm(X: torch.Tensor, **kwds: Any) -> torch.Tensor:
    """Implement PyRiemann's sample covariance matrix estimator."""
    assume_centered = bool(kwds.pop("assume_centered", False))
    if kwds:
        raise TypeError(f"Unsupported SCM keyword arguments: {sorted(kwds)}.")
    return _empirical_covariance(X, assume_centered=assume_centered, denominator=X.shape[-1])


def _ledoit_wolf(X: torch.Tensor, **kwds: Any) -> torch.Tensor:
    """Implement batched Ledoit-Wolf covariance shrinkage on Torch tensors."""
    assume_centered = bool(kwds.pop("assume_centered", False))
    # This parameter only controls sklearn's internal blockwise memory use.
    kwds.pop("block_size", None)
    if kwds:
        raise TypeError(f"Unsupported Ledoit-Wolf keyword arguments: {sorted(kwds)}.")

    centred = _center(X, assume_centered).transpose(-1, -2)
    n_times, n_channels = centred.shape[-2:]
    empirical = symmetrize(centred.transpose(-1, -2) @ centred / n_times)
    empirical_trace = centred.square().sum(dim=-2) / n_times
    mu = empirical_trace.mean(dim=-1)
    beta_raw = centred.square().sum(dim=-1).square().sum(dim=-1)
    delta_raw = (centred.transpose(-1, -2) @ centred).square().sum(dim=(-1, -2))
    delta_raw = delta_raw / (n_times**2)
    beta = (beta_raw / n_times - delta_raw) / (n_channels * n_times)
    delta = (
        delta_raw - 2.0 * mu * empirical_trace.sum(dim=-1) + n_channels * mu.square()
    ) / n_channels
    beta = torch.minimum(beta, delta)
    shrinkage = torch.where(delta == 0, torch.zeros_like(delta), beta / delta)
    identity = torch.eye(n_channels, dtype=X.dtype, device=X.device)
    return symmetrize(
        (1.0 - shrinkage)[..., None, None] * empirical
        + (shrinkage * mu)[..., None, None] * identity
    )


def _oas(X: torch.Tensor, **kwds: Any) -> torch.Tensor:
    """Implement batched Oracle Approximating Shrinkage on Torch tensors."""
    assume_centered = bool(kwds.pop("assume_centered", False))
    if kwds:
        raise TypeError(f"Unsupported OAS keyword arguments: {sorted(kwds)}.")

    n_channels, n_times = X.shape[-2:]
    empirical = _empirical_covariance(X, assume_centered=assume_centered, denominator=n_times)
    alpha = empirical.square().mean(dim=(-1, -2))
    mu = torch.diagonal(empirical, dim1=-2, dim2=-1).sum(dim=-1) / n_channels
    numerator = alpha + mu.square()
    denominator = (n_times + 1) * (alpha - mu.square() / n_channels)
    shrinkage = torch.where(
        denominator == 0,
        torch.ones_like(denominator),
        torch.minimum(numerator / denominator, torch.ones_like(denominator)),
    )
    identity = torch.eye(n_channels, dtype=X.dtype, device=X.device)
    return symmetrize(
        (1.0 - shrinkage)[..., None, None] * empirical
        + (shrinkage * mu)[..., None, None] * identity
    )


class TorchCovariances(BaseTorchSPDBuilder):
    """PyTorch counterpart of PyRiemann's ``Covariances`` transformer.

    ``mcd`` and ``hub`` are intentionally delegated to a temporary PyRiemann
    CPU fallback until their robust iterative estimators are ported to Torch.
    """

    _TORCH_ESTIMATORS = {"corr", "cov", "scm", "lwf", "oas"}
    _PYRIEMANN_FALLBACK_ESTIMATORS = {"mcd", "hub"}

    def __init__(self, estimator: str = "scm", **kwds: Any):
        """Initialise with the same estimator and keyword interface as PyRiemann."""
        self.estimator = estimator
        self.kwds = kwds

    def fit(self, X: torch.Tensor, y=None) -> "TorchCovariances":
        """Validate signals; covariance estimation has no fitted state."""
        _as_real_signal_tensor(X)
        return self

    def _pyriemann_fallback(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate a temporary CPU fallback and restore the source tensor device."""
        warnings.warn(
            f"Estimator '{self.estimator}' temporarily uses a PyRiemann CPU fallback.",
            RuntimeWarning,
            stacklevel=3,
        )
        input_array = X.detach().cpu().numpy().copy()
        result = PyRiemannCovariances(estimator=self.estimator, **self.kwds).fit_transform(
            input_array
        )
        return torch.as_tensor(result, dtype=X.dtype, device=X.device)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate one SPD matrix for each input time series."""
        signals = _as_real_signal_tensor(X)
        if self.estimator == "corr":
            return _correlation(signals, **self.kwds.copy())
        if self.estimator == "cov":
            return _covariance(signals, **self.kwds.copy())
        if self.estimator == "scm":
            return _scm(signals, **self.kwds.copy())
        if self.estimator == "lwf":
            return _ledoit_wolf(signals, **self.kwds.copy())
        if self.estimator == "oas":
            return _oas(signals, **self.kwds.copy())
        if self.estimator in self._PYRIEMANN_FALLBACK_ESTIMATORS:
            return self._pyriemann_fallback(signals)
        supported = sorted(self._TORCH_ESTIMATORS | self._PYRIEMANN_FALLBACK_ESTIMATORS)
        raise ValueError(f"Unsupported estimator: '{self.estimator}'. Valid options are: {supported}.")

    def transform_spd(self, X: torch.Tensor) -> DenseSPDBatch:
        """Estimate a validated dense SPD batch for the supplied time series."""
        return DenseSPDBatch(self.transform(X))

    def fit_transform(self, X: torch.Tensor, y=None) -> torch.Tensor:
        """Validate and estimate SPD matrices in one call."""
        return self.fit(X, y=y).transform(X)


class TorchBlockCovariances(BaseTorchSPDBuilder):
    """PyTorch counterpart of PyRiemann's dense block-diagonal estimator."""

    def __init__(self, block_size: Union[int, list[int]], estimator: str = "scm", **kwds: Any):
        """Initialise with the same block and estimator interface as PyRiemann."""
        self.estimator = estimator
        self.block_size = block_size
        self.kwds = kwds
        self._covariances = TorchCovariances(estimator=estimator, **kwds)

    def fit(self, X: torch.Tensor, y=None) -> "TorchBlockCovariances":
        """Validate signals; block covariance estimation has no fitted state."""
        _as_real_signal_tensor(X)
        return self

    def _block_sizes(self, n_channels: int) -> list[int]:
        """Resolve PyRiemann's integer or list block-size parameter."""
        if isinstance(self.block_size, int):
            return [self.block_size] * (n_channels // self.block_size)
        if isinstance(self.block_size, (list, np.ndarray)):
            return list(self.block_size)
        raise ValueError("Parameter block_size must be int or list.")

    def transform_spd(self, X: torch.Tensor) -> RaggedBlockSPDBatch:
        """Estimate independent SPD blocks without dense block-diagonal assembly."""
        signals = _as_real_signal_tensor(X)
        _, n_channels, _ = signals.shape
        blocks = []
        offset = 0
        for size in self._block_sizes(n_channels):
            if size <= 0 or offset + size > n_channels:
                raise ValueError("block_size must partition the input channel dimension.")
            blocks.append(
                self._covariances.transform(signals[:, offset:offset + size, :])
            )
            offset += size
        if offset != n_channels:
            raise ValueError("block_size must partition the input channel dimension.")
        return RaggedBlockSPDBatch(tuple(blocks))

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate dense block-diagonal SPD matrices for compatibility with PyRiemann."""
        return self.transform_spd(X).to_dense()

    def fit_transform(self, X: torch.Tensor, y=None) -> torch.Tensor:
        """Validate and estimate block-diagonal SPD matrices in one call."""
        return self.fit(X, y=y).transform(X)


class TorchCoSpectra(BaseTorchSPDBuilder):
    """PyTorch counterpart of PyRiemann's ``CoSpectra`` transformer."""

    def __init__(
        self,
        window: int = 128,
        overlap: float = 0.75,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        fs: Optional[float] = None,
    ):
        """Initialise co-spectral estimation with PyRiemann-compatible arguments."""
        self.window = self._next_power_of_two(window)
        self.overlap = overlap
        self.fmin = fmin
        self.fmax = fmax
        self.fs = fs
        self.freqs_: Optional[np.ndarray] = None

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        """Return the power-of-two window size used by PyRiemann."""
        result = 1
        while result < value:
            result *= 2
        return result

    def fit(self, X: torch.Tensor, y=None) -> "TorchCoSpectra":
        """Validate signals; co-spectral estimation has no fitted state."""
        _as_real_signal_tensor(X)
        return self

    def transform_spd(self, X: torch.Tensor) -> UniformBlockSPDBatch:
        """Estimate co-spectral SPD matrices with frequency as the block axis."""
        signals = _as_real_signal_tensor(X)
        spectra, frequencies = cospectrum_torch(
            signals,
            window=self.window,
            overlap=self.overlap,
            fmin=self.fmin,
            fmax=self.fmax,
            fs=self.fs,
        )
        self.freqs_ = frequencies
        return UniformBlockSPDBatch(spectra.permute(0, 3, 1, 2))

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Estimate co-spectra in PyRiemann's ``(samples, C, C, frequencies)`` layout."""
        return self.transform_spd(X).matrices.permute(0, 2, 3, 1)

    def fit_transform(self, X: torch.Tensor, y=None) -> torch.Tensor:
        """Validate and estimate co-spectral matrices in one call."""
        return self.fit(X, y=y).transform(X)
