from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from fedot_ind.core.kernel_learning.contracts import KernelBundle
from fedot_ind.core.kernel_learning.generators import normalize_feature_matrix


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
    normalize: str | None = "trace"
    center: bool = False
    psd_correction: str | None = "clip"
    psd_tol: float = 1e-8

    def fit(self, features: Any):
        self.train_features_ = normalize_feature_matrix(features)
        self.gamma_ = self._resolve_gamma(self.train_features_)
        raw_train = self._compute_kernel(self.train_features_, self.train_features_)
        self._train_column_mean_ = raw_train.mean(axis=0, keepdims=True)
        self._train_total_mean_ = float(raw_train.mean())
        train_kernel = self._apply_centering(raw_train, is_train=True)
        train_kernel, scale = self._apply_normalization(train_kernel, fit=True)
        train_kernel, diagnostics = self._validate_and_correct_psd(train_kernel)
        self.train_kernel_ = train_kernel
        self.normalization_scale_ = scale
        self.train_diagnostics_ = diagnostics
        return self

    def fit_transform(self, features: Any, *, name: str = "kernel",
                      train_features: np.ndarray | None = None) -> KernelBundle:
        self.fit(features)
        source_features = normalize_feature_matrix(train_features if train_features is not None else features)
        return KernelBundle(
            name=name,
            train_kernel=self.train_kernel_,
            train_features=source_features,
            is_psd=bool(self.train_diagnostics_["is_psd"]),
            psd_correction=self.train_diagnostics_["psd_correction"],
            complexity={"kernel_complexity": kernel_complexity(self.train_kernel_)},
            diagnostics=self._diagnostics(),
        )

    def transform(self, features: Any) -> np.ndarray:
        if not hasattr(self, "train_features_"):
            raise ValueError("KernelMatrixBuilder must be fitted before transform.")
        left = normalize_feature_matrix(features)
        cross_kernel = self._compute_kernel(left, self.train_features_)
        cross_kernel = self._apply_centering(cross_kernel, is_train=False)
        cross_kernel, _ = self._apply_normalization(cross_kernel, fit=False)
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
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        n_features = max(1, features.shape[1])
        variance = float(np.var(features))
        if str(self.gamma).lower() == "auto":
            return 1.0 / n_features
        if str(self.gamma).lower() == "scale":
            return 1.0 / (n_features * variance) if variance > 1e-12 else 1.0
        raise ValueError(f"Unsupported gamma value: {self.gamma}")

    def _compute_kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        kernel_name = self.kernel.lower()
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
        raise ValueError(f"Unsupported kernel type: {self.kernel}")

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
        if self.normalize is None:
            return kernel, 1.0
        normalized = self.normalize.lower()
        if fit:
            if normalized == "trace":
                trace = float(np.trace(kernel))
                scale = trace / max(1, kernel.shape[0]) if abs(trace) > 1e-12 else 1.0
            elif normalized == "frobenius":
                scale = _safe_frobenius_norm(kernel)
            else:
                raise ValueError(f"Unsupported kernel normalization: {self.normalize}")
            return kernel / scale, scale
        return kernel / self.normalization_scale_, self.normalization_scale_

    def _validate_and_correct_psd(self, kernel: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        symmetric = (kernel + kernel.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(symmetric)
        min_eigenvalue = float(np.min(eigvals)) if eigvals.size else 0.0
        correction = None
        corrected = symmetric
        if min_eigenvalue < -self.psd_tol and self.psd_correction == "clip":
            clipped = np.clip(eigvals, a_min=0.0, a_max=None)
            corrected = (eigvecs * clipped) @ eigvecs.T
            corrected = (corrected + corrected.T) / 2.0
            correction = "clip"
            eigvals = clipped
            min_eigenvalue = float(np.min(eigvals)) if eigvals.size else 0.0
        positive = eigvals[eigvals > self.psd_tol]
        condition_number = (
            float(np.max(eigvals) / np.min(positive))
            if positive.size and np.max(eigvals) > self.psd_tol
            else float("inf")
        )
        return corrected, {
            "min_eigenvalue": min_eigenvalue,
            "condition_number": condition_number,
            "is_psd": bool(min_eigenvalue >= -self.psd_tol),
            "psd_correction": correction,
        }

    def _diagnostics(self) -> dict[str, Any]:
        return {
            **self.train_diagnostics_,
            "kernel": self.kernel,
            "gamma": float(self.gamma_),
            "normalize": self.normalize,
            "normalization_scale": float(self.normalization_scale_),
            "center": bool(self.center),
        }
