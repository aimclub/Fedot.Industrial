"""Batched PyTorch primitives for real symmetric positive-definite matrices.

The functions in this module intentionally mirror the subset of PyRiemann
used by :class:`RiemannExtractor`.  They operate on the two trailing matrix
axes and preserve every leading batch axis, so the same implementation works
for ``(samples, channels, channels)`` and uniform block batches such as
``(samples, frequencies, channels, channels)``.

For the affine-invariant tangent map we follow PyRiemann's *coordinate* map:
``log(P ** -1/2 @ C @ P ** -1/2)``.  In particular, it is not the full
Riemannian logarithm with the outer ``P ** 1/2`` factors.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Optional

import torch
from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    DenseSPDBatch,
    DenseSPDReference,
    RaggedBlockSPDBatch,
    RaggedBlockSPDReference,
    TorchSPDBatch,
    TorchSPDReference,
    UniformBlockSPDBatch,
    UniformBlockSPDReference,
)


TangentMetric = Literal["riemann", "logeuclid", "euclid"]


def _check_square(matrix: torch.Tensor, name: str = "matrix") -> None:
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"{name} must have square matrices on its last two axes.")


def _check_real_floating(matrix: torch.Tensor, name: str = "matrix") -> None:
    if not torch.is_floating_point(matrix) or torch.is_complex(matrix):
        raise TypeError(f"{name} must be a real floating-point tensor.")


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Return the symmetric part of a real square matrix batch."""
    _check_square(matrix)
    return (matrix + matrix.transpose(-1, -2)) * 0.5


def _positive_eigenvalues(
    eigenvalues: torch.Tensor,
    eigenvalue_floor: Optional[float],
) -> torch.Tensor:
    floor = torch.finfo(eigenvalues.dtype).eps if eigenvalue_floor is None else eigenvalue_floor
    if floor <= 0:
        raise ValueError("eigenvalue_floor must be strictly positive.")
    return eigenvalues.clamp_min(floor)


def _spectral_function(
    matrix: torch.Tensor,
    function,
    *,
    eigenvalue_floor: Optional[float] = None,
    positive_spectrum: bool = False,
) -> torch.Tensor:
    """Apply a scalar function to eigenvalues of a symmetric matrix batch."""
    _check_square(matrix)
    _check_real_floating(matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetrize(matrix))
    if positive_spectrum:
        eigenvalues = _positive_eigenvalues(eigenvalues, eigenvalue_floor)
    transformed = function(eigenvalues)
    return symmetrize((eigenvectors * transformed.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2))


def matrix_log(matrix: torch.Tensor, *, eigenvalue_floor: Optional[float] = None) -> torch.Tensor:
    """Batched principal logarithm of real SPD matrices."""
    return _spectral_function(
        matrix,
        torch.log,
        eigenvalue_floor=eigenvalue_floor,
        positive_spectrum=True,
    )


def matrix_exp(matrix: torch.Tensor) -> torch.Tensor:
    """Batched exponential of real symmetric matrices."""
    return _spectral_function(matrix, torch.exp)


def matrix_sqrt(matrix: torch.Tensor, *, eigenvalue_floor: Optional[float] = None) -> torch.Tensor:
    """Batched principal square root of real SPD matrices."""
    return _spectral_function(
        matrix,
        torch.sqrt,
        eigenvalue_floor=eigenvalue_floor,
        positive_spectrum=True,
    )


def matrix_invsqrt(matrix: torch.Tensor, *, eigenvalue_floor: Optional[float] = None) -> torch.Tensor:
    """Batched principal inverse square root of real SPD matrices."""
    return _spectral_function(
        matrix,
        torch.rsqrt,
        eigenvalue_floor=eigenvalue_floor,
        positive_spectrum=True,
    )


def upper(matrix: torch.Tensor) -> torch.Tensor:
    """Vectorise the upper triangle with PyRiemann's off-diagonal weighting."""
    _check_square(matrix)
    n_channels = matrix.shape[-1]
    indices = torch.triu_indices(n_channels, n_channels, device=matrix.device)
    values = matrix[..., indices[0], indices[1]]
    weights = torch.where(
        indices[0] == indices[1],
        torch.ones(n_channels * (n_channels + 1) // 2, dtype=matrix.dtype, device=matrix.device),
        torch.full(
            (n_channels * (n_channels + 1) // 2,),
            math.sqrt(2.0),
            dtype=matrix.dtype,
            device=matrix.device,
        ),
    )
    return values * weights


def log_map_riemann(
    matrices: torch.Tensor,
    reference: torch.Tensor,
    *,
    reference_invsqrt: Optional[torch.Tensor] = None,
    eigenvalue_floor: Optional[float] = None,
) -> torch.Tensor:
    """PyRiemann-compatible affine-invariant tangent coordinate map."""
    _check_square(matrices, "matrices")
    _check_square(reference, "reference")
    if matrices.shape[-1] != reference.shape[-1]:
        raise ValueError("matrices and reference must have compatible dimensions.")
    invsqrt = (
        matrix_invsqrt(reference, eigenvalue_floor=eigenvalue_floor)
        if reference_invsqrt is None
        else reference_invsqrt
    )
    whitened = invsqrt @ matrices @ invsqrt
    return matrix_log(whitened, eigenvalue_floor=eigenvalue_floor)


def log_map_logeuclid(
    matrices: torch.Tensor,
    reference: torch.Tensor,
    *,
    reference_log: Optional[torch.Tensor] = None,
    eigenvalue_floor: Optional[float] = None,
) -> torch.Tensor:
    """Log-Euclidean tangent map."""
    log_reference = (
        matrix_log(reference, eigenvalue_floor=eigenvalue_floor)
        if reference_log is None
        else reference_log
    )
    return matrix_log(matrices, eigenvalue_floor=eigenvalue_floor) - log_reference


def log_map_euclid(matrices: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Euclidean tangent map."""
    return matrices - reference


def tangent_space(
    matrices: torch.Tensor,
    reference: torch.Tensor,
    *,
    metric: TangentMetric = "riemann",
    reference_invsqrt: Optional[torch.Tensor] = None,
    reference_log: Optional[torch.Tensor] = None,
    eigenvalue_floor: Optional[float] = None,
) -> torch.Tensor:
    """Project a batch of SPD matrices to PyRiemann-compatible tangent vectors."""
    if metric == "riemann":
        mapped = log_map_riemann(
            matrices,
            reference,
            reference_invsqrt=reference_invsqrt,
            eigenvalue_floor=eigenvalue_floor,
        )
    elif metric == "logeuclid":
        mapped = log_map_logeuclid(
            matrices,
            reference,
            reference_log=reference_log,
            eigenvalue_floor=eigenvalue_floor,
        )
    elif metric == "euclid":
        mapped = log_map_euclid(matrices, reference)
    else:
        raise ValueError(f"Unsupported tangent metric: '{metric}'.")
    return upper(mapped)


def _normalised_weights(matrices: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
    n_matrices = matrices.shape[0]
    if sample_weight is None:
        return torch.full(
            (n_matrices,), 1.0 / n_matrices, dtype=matrices.dtype, device=matrices.device
        )
    weights = torch.as_tensor(sample_weight, dtype=matrices.dtype, device=matrices.device).reshape(-1)
    if weights.numel() != n_matrices or torch.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sample_weight must be non-negative, non-empty, and match matrices.shape[0].")
    return weights / weights.sum()


def mean_euclid(matrices: torch.Tensor, *, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weighted Euclidean mean of an SPD matrix batch."""
    _check_square(matrices, "matrices")
    weights = _normalised_weights(matrices, sample_weight)
    return torch.einsum("n,nij->ij", weights, matrices)


def mean_logeuclid(
    matrices: torch.Tensor,
    *,
    sample_weight: Optional[torch.Tensor] = None,
    eigenvalue_floor: Optional[float] = None,
) -> torch.Tensor:
    """Weighted log-Euclidean mean of an SPD matrix batch."""
    weights = _normalised_weights(matrices, sample_weight)
    mean_log = torch.einsum(
        "n,nij->ij", weights, matrix_log(matrices, eigenvalue_floor=eigenvalue_floor)
    )
    return matrix_exp(mean_log)


def mean_riemann(
    matrices: torch.Tensor,
    *,
    sample_weight: Optional[torch.Tensor] = None,
    tol: float = 1e-9,
    max_iter: int = 50,
    initial_step: float = 1.0,
    eigenvalue_floor: Optional[float] = None,
) -> torch.Tensor:
    """Affine-invariant mean using PyRiemann's adaptive update schedule."""
    _check_square(matrices, "matrices")
    _check_real_floating(matrices, "matrices")
    if tol <= 0 or max_iter < 1 or initial_step <= 0:
        raise ValueError("tol, max_iter, and initial_step must be strictly positive.")

    weights = _normalised_weights(matrices, sample_weight)
    mean = mean_euclid(matrices, sample_weight=weights)
    step = float(initial_step)
    previous_update = float("inf")

    for _ in range(max_iter):
        mean_sqrt = matrix_sqrt(mean, eigenvalue_floor=eigenvalue_floor)
        mean_invsqrt = matrix_invsqrt(mean, eigenvalue_floor=eigenvalue_floor)
        tangent = matrix_log(
            mean_invsqrt @ matrices @ mean_invsqrt,
            eigenvalue_floor=eigenvalue_floor,
        )
        direction = torch.einsum("n,nij->ij", weights, tangent)
        mean = mean_sqrt @ matrix_exp(step * direction) @ mean_sqrt
        mean = symmetrize(mean)

        criterion = torch.linalg.matrix_norm(direction, ord="fro").item()
        update = step * criterion
        step = 0.95 * step if update < previous_update else 0.5 * step
        previous_update = update
        if criterion <= tol or step <= tol:
            break
    else:
        warnings.warn("Riemannian mean did not converge within max_iter.", RuntimeWarning)

    return mean


def median_euclid(
    matrices: torch.Tensor,
    *,
    sample_weight: Optional[torch.Tensor] = None,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> torch.Tensor:
    """Compute the Euclidean geometric median with Weiszfeld iterations."""
    weights = _normalised_weights(matrices, sample_weight)
    median = mean_euclid(matrices, sample_weight=weights)
    for _ in range(max_iter):
        distances = torch.linalg.matrix_norm(matrices - median, ord="fro", dim=(-2, -1))
        nonzero = distances > 0
        if not torch.any(nonzero):
            break
        reweights = weights[nonzero] / distances[nonzero]
        candidate = mean_euclid(matrices[nonzero], sample_weight=reweights)
        if torch.any(~nonzero):
            residual = torch.einsum("n,nij->ij", reweights, matrices[nonzero] - median)
            residual_norm = torch.linalg.matrix_norm(residual, ord="fro")
            inverse_residual = torch.where(
                residual_norm == 0,
                torch.zeros_like(residual_norm),
                weights[~nonzero].mean() / residual_norm,
            )
            candidate = (
                torch.clamp(1.0 - inverse_residual, min=0.0) * candidate
                + torch.clamp(inverse_residual, max=1.0) * median
            )
        criterion = torch.linalg.matrix_norm(candidate - median, ord="fro").item()
        median = symmetrize(candidate)
        if criterion <= tol:
            break
    else:
        warnings.warn("Euclidean geometric median did not converge within max_iter.", RuntimeWarning)
    return median


def median_logeuclid(
    matrices: torch.Tensor,
    *,
    sample_weight: Optional[torch.Tensor] = None,
    tol: float = 1e-5,
    max_iter: int = 50,
    eigenvalue_floor: Optional[float] = None,
) -> torch.Tensor:
    """Compute a geometric median in log-Euclidean SPD coordinates."""
    log_median = median_euclid(
        matrix_log(matrices, eigenvalue_floor=eigenvalue_floor),
        sample_weight=sample_weight,
        tol=tol,
        max_iter=max_iter,
    )
    return matrix_exp(log_median)


def median_riemann(
    matrices: torch.Tensor,
    *,
    sample_weight: Optional[torch.Tensor] = None,
    tol: float = 1e-5,
    max_iter: int = 50,
    step_size: float = 1.0,
    eigenvalue_floor: Optional[float] = None,
) -> torch.Tensor:
    """Compute the affine-invariant Riemannian geometric median of SPD matrices."""
    if not 0 < step_size <= 2:
        raise ValueError("step_size must be in (0, 2].")
    weights = _normalised_weights(matrices, sample_weight)
    median = mean_euclid(matrices, sample_weight=weights)
    for _ in range(max_iter):
        median_sqrt = matrix_sqrt(median, eigenvalue_floor=eigenvalue_floor)
        median_invsqrt = matrix_invsqrt(median, eigenvalue_floor=eigenvalue_floor)
        tangents = matrix_log(
            median_invsqrt @ matrices @ median_invsqrt,
            eigenvalue_floor=eigenvalue_floor,
        )
        distances = torch.linalg.matrix_norm(tangents, ord="fro", dim=(-2, -1))
        nonzero = distances > 0
        if not torch.any(nonzero):
            break
        reweights = weights[nonzero] / distances[nonzero]
        direction = torch.einsum("n,nij->ij", reweights / reweights.sum(), tangents[nonzero])
        median = symmetrize(
            median_sqrt @ matrix_exp(step_size * direction) @ median_sqrt
        )
        if torch.linalg.matrix_norm(direction, ord="fro").item() <= tol:
            break
    else:
        warnings.warn("Riemannian geometric median did not converge within max_iter.", RuntimeWarning)
    return median


def _validate_metric(metric: TangentMetric) -> TangentMetric:
    """Validate the single metric shared by centroid estimation and projection."""
    if not isinstance(metric, str):
        raise TypeError("metric must be a single metric name.")
    supported_metrics = {"riemann", "logeuclid", "euclid"}
    if metric not in supported_metrics:
        raise ValueError(f"Unsupported Torch tangent metric: {metric!r}.")
    return metric


class TorchSPDCentroid:
    """Estimate one SPD centroid from a batch of training matrices.

    The fitted ``centroid_`` can be supplied to :class:`TorchTangentSpace`
    for repeated projection of train, validation, and inference batches.
    """

    def __init__(
        self,
        metric: TangentMetric = "riemann",
        *,
        centroid_type: Literal["mean", "median"] = "mean",
        tol: float = 1e-9,
        max_iter: int = 50,
        initial_step: float = 1.0,
        median_tol: float = 1e-5,
        median_max_iter: int = 50,
        median_step_size: float = 1.0,
        eigenvalue_floor: Optional[float] = None,
    ):
        """Initialise centroid estimation parameters."""
        self.metric = _validate_metric(metric)
        if centroid_type not in {"mean", "median"}:
            raise ValueError("centroid_type must be 'mean' or 'median'.")
        self.centroid_type = centroid_type
        self.tol = tol
        self.max_iter = max_iter
        self.initial_step = initial_step
        self.median_tol = median_tol
        self.median_max_iter = median_max_iter
        self.median_step_size = median_step_size
        self.eigenvalue_floor = eigenvalue_floor
        self.centroid_: Optional[TorchSPDReference] = None

    def _mean(self, matrices: torch.Tensor, sample_weight=None) -> torch.Tensor:
        """Calculate one mean or median centroid using the configured metric."""
        if self.centroid_type == "median":
            if self.metric == "euclid":
                return median_euclid(
                    matrices, sample_weight=sample_weight, tol=self.median_tol,
                    max_iter=self.median_max_iter,
                )
            if self.metric == "logeuclid":
                return median_logeuclid(
                    matrices, sample_weight=sample_weight, tol=self.median_tol,
                    max_iter=self.median_max_iter, eigenvalue_floor=self.eigenvalue_floor,
                )
            return median_riemann(
                matrices, sample_weight=sample_weight, tol=self.median_tol,
                max_iter=self.median_max_iter, step_size=self.median_step_size,
                eigenvalue_floor=self.eigenvalue_floor,
            )
        if self.metric == "euclid":
            return mean_euclid(matrices, sample_weight=sample_weight)
        if self.metric == "logeuclid":
            return mean_logeuclid(matrices, sample_weight=sample_weight, eigenvalue_floor=self.eigenvalue_floor)
        return mean_riemann(
            matrices,
            sample_weight=sample_weight,
            tol=self.tol,
            max_iter=self.max_iter,
            initial_step=self.initial_step,
            eigenvalue_floor=self.eigenvalue_floor,
        )

    def _mean_uniform_blocks(self, matrices: torch.Tensor, sample_weight=None) -> torch.Tensor:
        """Estimate all equally sized block centroids in one batched computation."""
        weights = _normalised_weights(matrices, sample_weight)
        if self.metric == "euclid":
            return torch.einsum("n,nbij->bij", weights, matrices)
        if self.metric == "logeuclid":
            mean_log = torch.einsum(
                "n,nbij->bij", weights, matrix_log(matrices, eigenvalue_floor=self.eigenvalue_floor)
            )
            return matrix_exp(mean_log)

        block_matrices = matrices.transpose(0, 1)
        mean = torch.einsum("n,bnij->bij", weights, block_matrices)
        step = torch.ones(mean.shape[0], dtype=mean.dtype, device=mean.device) * self.initial_step
        previous_update = torch.full_like(step, float("inf"))
        active = torch.ones_like(step, dtype=torch.bool)
        for _ in range(self.max_iter):
            mean_sqrt = matrix_sqrt(mean, eigenvalue_floor=self.eigenvalue_floor)
            mean_invsqrt = matrix_invsqrt(mean, eigenvalue_floor=self.eigenvalue_floor)
            tangent = matrix_log(
                mean_invsqrt.unsqueeze(1) @ block_matrices @ mean_invsqrt.unsqueeze(1),
                eigenvalue_floor=self.eigenvalue_floor,
            )
            direction = torch.einsum("n,bnij->bij", weights, tangent)
            effective_step = torch.where(active, step, torch.zeros_like(step))
            mean = symmetrize(
                mean_sqrt @ matrix_exp(effective_step[:, None, None] * direction) @ mean_sqrt
            )
            criterion = torch.linalg.matrix_norm(direction, ord="fro")
            update = step * criterion
            next_step = torch.where(update < previous_update, step * 0.95, step * 0.5)
            step = torch.where(active, next_step, step)
            previous_update = torch.where(active, update, previous_update)
            active = active & (criterion > self.tol) & (step > self.tol)
            if not torch.any(active):
                break
        else:
            warnings.warn("Riemannian mean did not converge within max_iter.", RuntimeWarning)
        return mean

    def _median_blocks(
        self,
        blocks: tuple[torch.Tensor, ...],
        sample_weight=None,
        block_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Compute a coupled geometric median without materialising block diagonals."""
        if block_weights is None:
            block_weights = blocks[0].new_ones(len(blocks))
        if self.metric == "logeuclid":
            log_blocks = tuple(
                matrix_log(block, eigenvalue_floor=self.eigenvalue_floor) for block in blocks
            )
            original_metric = self.metric
            self.metric = "euclid"
            try:
                median_blocks = self._median_blocks(
                    log_blocks, sample_weight, block_weights
                )
            finally:
                self.metric = original_metric
            return tuple(matrix_exp(block) for block in median_blocks)

        weights = _normalised_weights(blocks[0], sample_weight)
        medians = tuple(mean_euclid(block, sample_weight=weights) for block in blocks)
        for _ in range(self.median_max_iter):
            if self.metric == "euclid":
                differences = tuple(block - median for block, median in zip(blocks, medians))
                squared_distances = sum(
                    weight * torch.linalg.matrix_norm(
                        difference, ord="fro", dim=(-2, -1)
                    ).square()
                    for weight, difference in zip(block_weights, differences)
                )
                directions = differences
            else:
                square_roots = tuple(
                    matrix_sqrt(median, eigenvalue_floor=self.eigenvalue_floor) for median in medians
                )
                inverse_square_roots = tuple(
                    matrix_invsqrt(median, eigenvalue_floor=self.eigenvalue_floor)
                    for median in medians
                )
                directions = tuple(
                    matrix_log(invsqrt @ block @ invsqrt, eigenvalue_floor=self.eigenvalue_floor)
                    for block, invsqrt in zip(blocks, inverse_square_roots)
                )
                squared_distances = sum(
                    weight * torch.linalg.matrix_norm(
                        direction, ord="fro", dim=(-2, -1)
                    ).square()
                    for weight, direction in zip(block_weights, directions)
                )

            distances = torch.sqrt(squared_distances)
            nonzero = distances > 0
            if not torch.any(nonzero):
                break
            reweights = weights[nonzero] / distances[nonzero]
            normalised_reweights = reweights / reweights.sum()
            if self.metric == "euclid":
                candidates = tuple(
                    mean_euclid(block[nonzero], sample_weight=reweights) for block in blocks
                )
                criterion = torch.sqrt(sum(
                    weight * torch.linalg.matrix_norm(
                        candidate - median, ord="fro"
                    ).square()
                    for weight, candidate, median in zip(
                        block_weights, candidates, medians
                    )
                )).item()
            else:
                updates = tuple(
                    torch.einsum("n,nij->ij", normalised_reweights, direction[nonzero])
                    for direction in directions
                )
                candidates = tuple(
                    symmetrize(
                        sqrt @ matrix_exp(self.median_step_size * update) @ sqrt
                    )
                    for sqrt, update in zip(square_roots, updates)
                )
                criterion = torch.sqrt(sum(
                    weight * torch.linalg.matrix_norm(update, ord="fro").square()
                    for weight, update in zip(block_weights, updates)
                )).item()
            medians = candidates
            if criterion <= self.median_tol:
                break
        else:
            warnings.warn("Structured geometric median did not converge within max_iter.", RuntimeWarning)
        return medians

    @staticmethod
    def _validate_block_weights(
        X: TorchSPDBatch, block_weights
    ) -> torch.Tensor:
        """Return finite positive metric weights matching the SPD block layout."""
        reference = X.matrices[0] if isinstance(X, RaggedBlockSPDBatch) else X.matrices
        if block_weights is None:
            return reference.new_ones(len(X.block_shapes))
        try:
            weights = torch.as_tensor(
                block_weights, dtype=reference.dtype, device=reference.device
            )
        except (TypeError, ValueError) as error:
            raise TypeError("block_weights must be a one-dimensional array-like.") from error
        if weights.ndim != 1:
            raise TypeError("block_weights must be a one-dimensional array-like.")
        if weights.numel() != len(X.block_shapes):
            raise ValueError("block_weights must contain one value for every SPD block.")
        if not torch.all(torch.isfinite(weights)):
            raise ValueError("block_weights must contain only finite values.")
        if not torch.all(weights > 0):
            raise ValueError("block_weights must be strictly positive.")
        return weights

    def fit(
        self,
        X: TorchSPDBatch,
        y=None,
        sample_weight=None,
        block_weights=None,
    ) -> "TorchSPDCentroid":
        """Estimate and retain the SPD centroid of ``X``.

        ``y`` is accepted for compatibility with estimator pipelines and is
        intentionally not used for global centroid estimation.
        """
        del y
        if not isinstance(X, TorchSPDBatch):
            raise TypeError("X must be a validated TorchSPDBatch.")
        validated_block_weights = self._validate_block_weights(X, block_weights)
        if isinstance(X, DenseSPDBatch):
            self.centroid_ = DenseSPDReference(self._mean(X.matrices, sample_weight))
        elif isinstance(X, UniformBlockSPDBatch):
            if self.centroid_type == "median":
                blocks = tuple(X.matrices[:, index] for index in range(X.matrices.shape[1]))
                self.centroid_ = UniformBlockSPDReference(
                    torch.stack(self._median_blocks(
                        blocks, sample_weight, validated_block_weights
                    ))
                )
            else:
                self.centroid_ = UniformBlockSPDReference(
                    self._mean_uniform_blocks(X.matrices, sample_weight)
                )
        elif isinstance(X, RaggedBlockSPDBatch):
            if self.centroid_type == "median":
                self.centroid_ = RaggedBlockSPDReference(
                    self._median_blocks(
                        X.matrices, sample_weight, validated_block_weights
                    )
                )
            else:
                self.centroid_ = RaggedBlockSPDReference(
                    tuple(self._mean(block, sample_weight) for block in X.matrices)
                )
        else:
            raise TypeError(f"Unsupported SPD batch type: {type(X).__name__}.")
        return self


class TorchTangentSpace:
    """Project SPD matrices to the tangent space of a supplied centroid.

    The centroid is immutable projection state. Reusable spectral operators
    are computed once during initialisation and remain on its device.
    """

    def __init__(
        self,
        centroid: TorchSPDReference,
        metric: TangentMetric = "riemann",
        *,
        eigenvalue_floor: Optional[float] = None,
    ):
        """Initialise a projector for the given centroid and metric."""
        self.metric = _validate_metric(metric)
        self.eigenvalue_floor = eigenvalue_floor
        if not isinstance(centroid, TorchSPDReference):
            raise TypeError("centroid must be a validated TorchSPDReference.")
        self.centroid_ = centroid
        self.centroid_invsqrt_ = None
        self.centroid_log_ = None
        if self.metric == "riemann":
            self.centroid_invsqrt_ = self._apply_reference_operator(matrix_invsqrt)
        elif self.metric == "logeuclid":
            self.centroid_log_ = self._apply_reference_operator(matrix_log)

    def _apply_reference_operator(self, operator):
        """Apply a cached spectral operator while preserving reference layout."""
        if isinstance(self.centroid_, DenseSPDReference):
            return operator(self.centroid_.matrices, eigenvalue_floor=self.eigenvalue_floor)
        if isinstance(self.centroid_, UniformBlockSPDReference):
            return operator(self.centroid_.matrices, eigenvalue_floor=self.eigenvalue_floor)
        return tuple(operator(block, eigenvalue_floor=self.eigenvalue_floor) for block in self.centroid_.matrices)

    def _check_layout(self, X: TorchSPDBatch) -> None:
        """Ensure input blocks have exactly the layout used by the centroid."""
        if X.structure != self.centroid_.structure or X.block_shapes != self.centroid_.block_shapes:
            raise ValueError("SPD batch block layout must match the centroid block layout.")

    def transform(self, X: TorchSPDBatch) -> torch.Tensor:
        """Project SPD matrices to the tangent space of ``centroid_``."""
        if not isinstance(X, TorchSPDBatch):
            raise TypeError("X must be a validated TorchSPDBatch.")
        self._check_layout(X)
        if isinstance(X, DenseSPDBatch):
            return tangent_space(X.matrices, self.centroid_.matrices, metric=self.metric,
                                 reference_invsqrt=self.centroid_invsqrt_, reference_log=self.centroid_log_,
                                 eigenvalue_floor=self.eigenvalue_floor)
        if isinstance(X, UniformBlockSPDBatch):
            features = tangent_space(X.matrices, self.centroid_.matrices, metric=self.metric,
                                     reference_invsqrt=self.centroid_invsqrt_, reference_log=self.centroid_log_,
                                     eigenvalue_floor=self.eigenvalue_floor)
            return features.flatten(start_dim=1)
        features = [
            tangent_space(block, reference, metric=self.metric, reference_invsqrt=invsqrt,
                          reference_log=reference_log, eigenvalue_floor=self.eigenvalue_floor)
            for block, reference, invsqrt, reference_log in zip(
                X.matrices, self.centroid_.matrices,
                self.centroid_invsqrt_ or (None,) * len(X.matrices),
                self.centroid_log_ or (None,) * len(X.matrices),
            )
        ]
        return torch.cat(features, dim=1)
