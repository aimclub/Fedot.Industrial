"""Torch MDM feature extraction for dense and block-structured SPD matrices."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch

from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    DenseSPDBatch,
    RaggedBlockSPDBatch,
    TorchSPDBatch,
    TorchSPDReference,
    UniformBlockSPDBatch,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_tangent_space import (
    TangentMetric,
    TorchSPDCentroid,
    _validate_metric,
    matrix_invsqrt,
    matrix_log,
)


def _squared_frobenius(matrix: torch.Tensor) -> torch.Tensor:
    """Return squared Frobenius norms over the two trailing matrix axes."""
    return matrix.square().sum(dim=(-2, -1))


class TorchMDMDistances:
    """Calculate one MDM distance feature for every supplied SPD centroid."""

    def __init__(
        self,
        centroids: Sequence[TorchSPDReference],
        metric: TangentMetric = "riemann",
        *,
        eigenvalue_floor: Optional[float] = None,
    ):
        """Initialise a distance transformer with fixed, validated centroids."""
        if not centroids:
            raise ValueError("centroids must contain at least one reference.")
        if not all(isinstance(centroid, TorchSPDReference) for centroid in centroids):
            raise TypeError("centroids must contain validated TorchSPDReference objects.")
        self.centroids_ = tuple(centroids)
        self.metric = _validate_metric(metric)
        self.eigenvalue_floor = eigenvalue_floor
        first = self.centroids_[0]
        if any(
            centroid.structure != first.structure or centroid.block_shapes != first.block_shapes
            for centroid in self.centroids_[1:]
        ):
            raise ValueError("All centroids must share the same block layout.")
        self._invsqrts_ = (
            tuple(self._reference_operator(centroid, matrix_invsqrt) for centroid in self.centroids_)
            if self.metric == "riemann"
            else None
        )
        self._logs_ = (
            tuple(self._reference_operator(centroid, matrix_log) for centroid in self.centroids_)
            if self.metric == "logeuclid"
            else None
        )

    def _reference_operator(self, centroid: TorchSPDReference, operator):
        """Apply a spectral operator to every reference block without changing layout."""
        if centroid.structure == "ragged_blocks":
            return tuple(
                operator(block, eigenvalue_floor=self.eigenvalue_floor)
                for block in centroid.matrices
            )
        return operator(centroid.matrices, eigenvalue_floor=self.eigenvalue_floor)

    def _check_layout(self, spd: TorchSPDBatch) -> None:
        """Ensure an SPD batch is compatible with every stored centroid."""
        first = self.centroids_[0]
        if spd.structure != first.structure or spd.block_shapes != first.block_shapes:
            raise ValueError("SPD batch block layout must match centroid block layout.")

    def _distance(self, spd: TorchSPDBatch, centroid: TorchSPDReference, index: int) -> torch.Tensor:
        """Calculate batched distances to one centroid while retaining block structure."""
        cache_invsqrt = None if self._invsqrts_ is None else self._invsqrts_[index]
        cache_log = None if self._logs_ is None else self._logs_[index]

        def squared_distance(matrices, reference, invsqrt, reference_log):
            """Calculate squared distances for one dense or uniform block tensor."""
            if self.metric == "euclid":
                return _squared_frobenius(matrices - reference)
            if self.metric == "logeuclid":
                return _squared_frobenius(
                    matrix_log(matrices, eigenvalue_floor=self.eigenvalue_floor) - reference_log
                )
            tangent = matrix_log(
                invsqrt @ matrices @ invsqrt,
                eigenvalue_floor=self.eigenvalue_floor,
            )
            return _squared_frobenius(tangent)

        if isinstance(spd, DenseSPDBatch):
            return torch.sqrt(squared_distance(spd.matrices, centroid.matrices, cache_invsqrt, cache_log))
        if isinstance(spd, UniformBlockSPDBatch):
            squared = squared_distance(spd.matrices, centroid.matrices, cache_invsqrt, cache_log)
            return torch.sqrt(squared.sum(dim=1))
        squared = sum(
            squared_distance(block, reference, invsqrt, reference_log)
            for block, reference, invsqrt, reference_log in zip(
                spd.matrices,
                centroid.matrices,
                cache_invsqrt or (None,) * len(spd.matrices),
                cache_log or (None,) * len(spd.matrices),
            )
        )
        return torch.sqrt(squared)

    def transform(self, X: TorchSPDBatch) -> torch.Tensor:
        """Return MDM features with shape ``(n_samples, n_centroids)``."""
        if not isinstance(X, TorchSPDBatch):
            raise TypeError("X must be a validated TorchSPDBatch.")
        self._check_layout(X)
        return torch.stack(
            [self._distance(X, centroid, index) for index, centroid in enumerate(self.centroids_)],
            dim=1,
        )


class TorchClassCentroids:
    """Estimate one SPD centroid per observed class label."""

    def __init__(
        self,
        metric: TangentMetric = "riemann",
        *,
        centroid_type: Literal["mean", "median"] = "mean",
        **centroid_kwargs,
    ):
        """Initialise a class-wise wrapper around ``TorchSPDCentroid`` settings."""
        self.metric = _validate_metric(metric)
        self.centroid_type = centroid_type
        self.centroid_kwargs = centroid_kwargs
        self.classes_: Optional[np.ndarray] = None
        self.centroids_: Optional[tuple[TorchSPDReference, ...]] = None

    @staticmethod
    def _subset(spd: TorchSPDBatch, indices: torch.Tensor) -> TorchSPDBatch:
        """Select samples while preserving the concrete structured SPD type."""
        if isinstance(spd, DenseSPDBatch):
            return DenseSPDBatch(spd.matrices[indices])
        if isinstance(spd, UniformBlockSPDBatch):
            return UniformBlockSPDBatch(spd.matrices[indices])
        return RaggedBlockSPDBatch(tuple(block[indices] for block in spd.matrices))

    def fit(self, X: TorchSPDBatch, y, sample_weight=None) -> "TorchClassCentroids":
        """Estimate a centroid for each sorted class in ``y``."""
        if not isinstance(X, TorchSPDBatch):
            raise TypeError("X must be a validated TorchSPDBatch.")
        labels = np.asarray(y).reshape(-1)
        if labels.shape[0] != X.n_samples:
            raise ValueError("y must contain one label for every SPD sample.")
        self.classes_ = np.unique(labels)
        centroids = []
        device = X.matrices.device if not isinstance(X, RaggedBlockSPDBatch) else X.matrices[0].device
        for label in self.classes_:
            indices = torch.as_tensor(np.flatnonzero(labels == label), device=device)
            class_weight = None if sample_weight is None else torch.as_tensor(sample_weight)[indices.cpu()]
            centroids.append(
                TorchSPDCentroid(
                    metric=self.metric, centroid_type=self.centroid_type, **self.centroid_kwargs
                ).fit(self._subset(X, indices), sample_weight=class_weight).centroid_
            )
        self.centroids_ = tuple(centroids)
        return self
