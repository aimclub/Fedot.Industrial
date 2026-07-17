from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from fedot_ind.core.kernel_learning.cache import (
    InMemoryKernelCache,
    KernelCacheKey,
    KernelCachePolicy,
    fingerprint_array,
    fingerprint_mapping,
)
from fedot_ind.core.kernel_learning.contracts import (
    KernelApproximation,
    KernelBundle,
    KernelMatrixPolicy,
    KernelNormalization,
    PSDCorrectionPolicy,
)
from fedot_ind.core.kernel_learning.generators import normalize_feature_matrix
from fedot_ind.core.kernel_learning.kernels.approximation import (
    NystromApproximationPolicy,
    NystromKernelApproximator,
)


def _safe_frobenius_norm(matrix: np.ndarray) -> float:
    value = float(np.linalg.norm(matrix, ord="fro"))
    return value if value > 1e-12 else 1.0


def kernel_complexity(kernel: np.ndarray) -> float:
    matrix = np.asarray(kernel, dtype=float)
    return float(np.trace(matrix) / _safe_frobenius_norm(matrix))


@dataclass
class KernelMatrixBuilder:
    kernel: str = "rbf"
    gamma: str | float = "scale"
    degree: int = 3
    coef0: float = 1.0
    normalize: KernelNormalization | str | None = KernelNormalization.TRACE
    center: bool = False
    psd_correction: PSDCorrectionPolicy | str | None = PSDCorrectionPolicy.CLIP
    psd_tol: float = 1e-8
    approximation: KernelApproximation | str | None = None
    nystrom_components: int | None = None
    cache_policy: KernelCachePolicy = KernelCachePolicy()
    cache: InMemoryKernelCache | None = None

    @property
    def policy(self) -> KernelMatrixPolicy:
        return KernelMatrixPolicy(
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            normalize=self.normalize,
            center=self.center,
            psd_correction=self.psd_correction,
            psd_tol=self.psd_tol,
            approximation=self.approximation,
            nystrom_components=self.nystrom_components,
        ).normalized_policy

    def fit(self, features: Any):
        self._policy_ = self.policy
        self.train_features_ = normalize_feature_matrix(features)
        self._validate_feature_matrix(self.train_features_, source="train")
        self.gamma_ = self._resolve_gamma(self.train_features_)
        raw_train = self._compute_train_kernel(self.train_features_)
        self._validate_train_kernel(raw_train, stage="raw")
        self._train_column_mean_ = raw_train.mean(axis=0, keepdims=True)
        self._train_total_mean_ = float(raw_train.mean())
        train_kernel = self._apply_centering(raw_train, is_train=True)
        train_kernel, scale = self._apply_normalization(train_kernel, fit=True)
        train_kernel, diagnostics = self._validate_and_correct_psd(train_kernel)
        self._validate_train_kernel(train_kernel, stage="final")
        self.train_kernel_ = train_kernel
        self.normalization_scale_ = scale
        self.train_diagnostics_ = diagnostics
        return self

    def fit_transform(self, features: Any, *, name: str = "kernel",
                      train_features: np.ndarray | None = None) -> KernelBundle:
        source_features = normalize_feature_matrix(train_features if train_features is not None else features)
        cache_key = self._cache_key(name, source_features)
        if cache_key is not None and self._cache_hit_supported():
            cached_bundle = self.cache.get(cache_key) if self.cache is not None else None
            if cached_bundle is not None:
                self._restore_from_cached_bundle(cached_bundle)
                return replace(
                    cached_bundle,
                    diagnostics={
                        **cached_bundle.diagnostics,
                        "cache": self._cache_diagnostics(cache_key, hit=True),
                    },
                )

        self.fit(features)
        bundle = KernelBundle(
            name=name,
            train_kernel=self.train_kernel_,
            train_features=source_features,
            is_psd=bool(self.train_diagnostics_["is_psd"]),
            psd_correction=self.train_diagnostics_["psd_correction"],
            complexity={"kernel_complexity": kernel_complexity(self.train_kernel_)},
            diagnostics={
                **self._diagnostics(),
                "cache": self._cache_diagnostics(cache_key, hit=False),
            },
        )
        if cache_key is not None and self._cache_hit_supported() and self.cache is not None:
            self.cache.put(cache_key, bundle)
        return bundle

    def transform(self, features: Any) -> np.ndarray:
        if not hasattr(self, "train_features_"):
            raise ValueError("KernelMatrixBuilder must be fitted before transform.")
        left = normalize_feature_matrix(features)
        self._validate_feature_matrix(left, source="test")
        cross_kernel = self._compute_cross_kernel(left)
        self._validate_cross_kernel(cross_kernel, n_left=left.shape[0])
        cross_kernel = self._apply_centering(cross_kernel, is_train=False)
        cross_kernel, _ = self._apply_normalization(cross_kernel, fit=False)
        self._validate_cross_kernel(cross_kernel, n_left=left.shape[0])
        return cross_kernel

    def build_test_bundle(self, features: Any, train_bundle: KernelBundle) -> KernelBundle:
        test_features = normalize_feature_matrix(features)
        return KernelBundle(
            name=train_bundle.name,
            train_kernel=train_bundle.train_kernel,
            test_kernel=self.transform(test_features),
            train_features=train_bundle.train_features,
            test_features=test_features,
            is_psd=train_bundle.is_psd,
            psd_correction=train_bundle.psd_correction,
            complexity=train_bundle.complexity,
            diagnostics=train_bundle.diagnostics,
        )

    def _resolve_gamma(self, features: np.ndarray) -> float:
        gamma = self._active_policy().gamma
        if isinstance(gamma, (int, float)):
            return float(gamma)
        n_features = max(1, features.shape[1])
        variance = float(np.var(features))
        if str(gamma).lower() == "auto":
            return 1.0 / n_features
        if str(gamma).lower() == "scale":
            return 1.0 / (n_features * variance) if variance > 1e-12 else 1.0
        raise ValueError(f"Unsupported gamma value: {gamma}")

    def _compute_train_kernel(self, features: np.ndarray) -> np.ndarray:
        if self._active_policy().approximation is KernelApproximation.NYSTROM:
            self._approximator_ = NystromKernelApproximator(
                NystromApproximationPolicy(n_components=self._policy_.nystrom_components)
            ).fit(features, self._compute_kernel)
            return self._approximator_.transform_train()
        self._approximator_ = None
        return self._compute_kernel(features, features)

    def _compute_cross_kernel(self, features: np.ndarray) -> np.ndarray:
        if getattr(self, "_approximator_", None) is not None:
            return self._approximator_.transform_cross(features)
        return self._compute_kernel(features, self.train_features_)

    def _compute_kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        kernel_name = self._active_policy().kernel.lower()
        if kernel_name == "rbf":
            distances = cdist(left, right, metric="sqeuclidean")
            return np.exp(-self.gamma_ * distances)
        if kernel_name == "laplacian":
            distances = cdist(left, right, metric="cityblock")
            return np.exp(-self.gamma_ * distances)
        if kernel_name == "cosine":
            left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
            right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
            return left_norm @ right_norm.T
        if kernel_name == "linear":
            return left @ right.T
        if kernel_name == "polynomial":
            return (self.gamma_ * (left @ right.T) + self.coef0) ** self.degree
        raise ValueError(f"Unsupported kernel type: {self._active_policy().kernel}")

    def _validate_feature_matrix(self, features: np.ndarray, *, source: str) -> None:
        if features.ndim != 2:
            raise ValueError(f"{source} features must be a 2D matrix after normalization.")
        if features.shape[0] < 1:
            raise ValueError(f"{source} features must contain at least one sample.")
        if features.shape[1] < 1:
            raise ValueError(f"{source} features must contain at least one feature.")
        if not np.all(np.isfinite(features)):
            raise ValueError(f"{source} features contain non-finite values after normalization.")

    def _validate_train_kernel(self, kernel: np.ndarray, *, stage: str) -> None:
        matrix = np.asarray(kernel, dtype=float)
        n_samples = self.train_features_.shape[0]
        if matrix.shape != (n_samples, n_samples):
            raise ValueError(
                f"{stage} train kernel must have shape {(n_samples, n_samples)}, got {matrix.shape}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{stage} train kernel contains non-finite values.")
        if _looks_like_distance_matrix(matrix):
            raise ValueError(
                f"{stage} train kernel looks like a distance matrix; convert distances with a valid kernel first."
            )

    def _validate_cross_kernel(self, kernel: np.ndarray, *, n_left: int) -> None:
        matrix = np.asarray(kernel, dtype=float)
        expected = (n_left, self.train_features_.shape[0])
        if matrix.shape != expected:
            raise ValueError(f"test kernel must have shape {expected}, got {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("test kernel contains non-finite values.")

    def _apply_centering(self, kernel: np.ndarray, *, is_train: bool) -> np.ndarray:
        if not self.center:
            return kernel
        if is_train:
            row_mean = kernel.mean(axis=1, keepdims=True)
            column_mean = kernel.mean(axis=0, keepdims=True)
            total_mean = float(kernel.mean())
            return kernel - row_mean - column_mean + total_mean
        row_mean = kernel.mean(axis=1, keepdims=True)
        return kernel - row_mean - self._train_column_mean_ + self._train_total_mean_

    def _apply_normalization(self, kernel: np.ndarray, *, fit: bool) -> tuple[np.ndarray, float]:
        normalization = self._active_policy().normalize
        if normalization is None:
            return kernel, 1.0
        if fit:
            if normalization is KernelNormalization.TRACE:
                trace = float(np.trace(kernel))
                scale = trace / max(1, kernel.shape[0]) if abs(trace) > 1e-12 else 1.0
            elif normalization is KernelNormalization.FROBENIUS:
                scale = _safe_frobenius_norm(kernel)
            else:
                raise ValueError(f"Unsupported kernel normalization: {normalization}")
            return kernel / scale, scale
        return kernel / self.normalization_scale_, self.normalization_scale_

    def _validate_and_correct_psd(self, kernel: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        policy = self._active_policy()
        symmetric = (kernel + kernel.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(symmetric)
        min_eigenvalue = float(np.min(eigvals)) if eigvals.size else 0.0
        correction = None
        corrected = symmetric
        if min_eigenvalue < -policy.psd_tol and policy.psd_correction is PSDCorrectionPolicy.CLIP:
            clipped = np.clip(eigvals, a_min=0.0, a_max=None)
            corrected = (eigvecs * clipped) @ eigvecs.T
            corrected = (corrected + corrected.T) / 2.0
            correction = PSDCorrectionPolicy.CLIP.value
            eigvals = clipped
            min_eigenvalue = float(np.min(eigvals)) if eigvals.size else 0.0
        positive = eigvals[eigvals > policy.psd_tol]
        condition_number = (
            float(np.max(eigvals) / np.min(positive))
            if positive.size and np.max(eigvals) > policy.psd_tol
            else float("inf")
        )
        return corrected, {
            "min_eigenvalue": min_eigenvalue,
            "condition_number": condition_number,
            "is_psd": bool(min_eigenvalue >= -policy.psd_tol),
            "psd_correction": correction,
        }

    def _diagnostics(self) -> dict[str, Any]:
        policy = self._active_policy()
        diagnostics = {
            **self.train_diagnostics_,
            "kernel": policy.kernel,
            "gamma": float(self.gamma_),
            "normalize": None if policy.normalize is None else policy.normalize.value,
            "normalization_scale": float(self.normalization_scale_),
            "center": bool(policy.center),
            "policy": policy.to_dict(),
        }
        if getattr(self, "_approximator_", None) is not None:
            diagnostics.update(self._approximator_.diagnostics())
        return diagnostics

    def _active_policy(self) -> KernelMatrixPolicy:
        return getattr(self, "_policy_", self.policy)

    def _cache_key(self, name: str, features: np.ndarray) -> KernelCacheKey | None:
        if not self.cache_policy.enabled:
            return None
        return KernelCacheKey(
            namespace=self.cache_policy.namespace,
            generator_name=name,
            kernel_policy_hash=fingerprint_mapping(self.policy.to_dict()),
            data_fingerprint=fingerprint_array(features),
        )

    def _cache_hit_supported(self) -> bool:
        policy = self.policy
        return not bool(policy.center) and policy.approximation is None

    def _cache_diagnostics(self, key: KernelCacheKey | None, *, hit: bool) -> dict[str, Any]:
        return {
            "enabled": bool(self.cache_policy.enabled),
            "supported": bool(self._cache_hit_supported()),
            "hit": bool(hit),
            "key": None if key is None else key.as_tuple(),
        }

    def _restore_from_cached_bundle(self, bundle: KernelBundle) -> None:
        diagnostics = dict(bundle.diagnostics)
        self._policy_ = self.policy
        self.train_features_ = normalize_feature_matrix(bundle.train_features)
        self.train_kernel_ = np.asarray(bundle.train_kernel, dtype=float)
        self.gamma_ = float(diagnostics["gamma"])
        self.normalization_scale_ = float(diagnostics["normalization_scale"])
        self.train_diagnostics_ = {
            "min_eigenvalue": diagnostics.get("min_eigenvalue"),
            "condition_number": diagnostics.get("condition_number"),
            "is_psd": diagnostics.get("is_psd", bundle.is_psd),
            "psd_correction": diagnostics.get("psd_correction", bundle.psd_correction),
        }
        self._approximator_ = None


def _looks_like_distance_matrix(matrix: np.ndarray) -> bool:
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] <= 1:
        return False
    symmetric = np.allclose(matrix, matrix.T, atol=1e-8)
    zero_diagonal = np.allclose(np.diag(matrix), 0.0, atol=1e-10)
    non_negative = bool(np.all(matrix >= -1e-10))
    has_positive_off_diagonal = bool(np.any(np.abs(matrix - np.diag(np.diag(matrix))) > 1e-10))
    return symmetric and zero_diagonal and non_negative and has_positive_off_diagonal
