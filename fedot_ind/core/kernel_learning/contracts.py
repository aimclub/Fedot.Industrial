from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np


class KernelTaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"


class KernelNormalization(str, Enum):
    TRACE = "trace"
    FROBENIUS = "frobenius"
    NONE = "none"


class PSDCorrectionPolicy(str, Enum):
    CLIP = "clip"
    NONE = "none"


@dataclass(frozen=True)
class KernelMatrixPolicy:
    kernel: str = "rbf"
    gamma: str | float = "scale"
    degree: int = 3
    coef0: float = 1.0
    normalize: str | None = KernelNormalization.TRACE.value
    center: bool = False
    psd_correction: str | None = PSDCorrectionPolicy.CLIP.value
    psd_tol: float = 1e-8
    approximation: str | None = None
    nystrom_components: int | None = None

    def __post_init__(self):
        if self.degree < 1:
            raise ValueError("degree must be at least 1.")
        if self.psd_tol < 0.0:
            raise ValueError("psd_tol must be non-negative.")
        if self.normalize is not None and str(self.normalize).lower() not in {
            KernelNormalization.TRACE.value,
            KernelNormalization.FROBENIUS.value,
            KernelNormalization.NONE.value,
        }:
            raise ValueError(f"Unsupported kernel normalization: {self.normalize}")
        if self.psd_correction is not None and str(self.psd_correction).lower() not in {
            PSDCorrectionPolicy.CLIP.value,
            PSDCorrectionPolicy.NONE.value,
        }:
            raise ValueError(f"Unsupported PSD correction policy: {self.psd_correction}")
        if self.approximation is not None and str(self.approximation).lower() not in {"nystrom"}:
            raise ValueError(f"Unsupported kernel approximation: {self.approximation}")
        if self.nystrom_components is not None and self.nystrom_components < 1:
            raise ValueError("nystrom_components must be at least 1.")

    @property
    def normalized_policy(self) -> "KernelMatrixPolicy":
        normalize = None if self.normalize is None or str(self.normalize).lower() == "none" else str(self.normalize)
        psd_correction = (
            None
            if self.psd_correction is None or str(self.psd_correction).lower() == "none"
            else str(self.psd_correction)
        )
        approximation = None if self.approximation is None else str(self.approximation).lower()
        return KernelMatrixPolicy(
            kernel=str(self.kernel).lower(),
            gamma=self.gamma,
            degree=int(self.degree),
            coef0=float(self.coef0),
            normalize=normalize,
            center=bool(self.center),
            psd_correction=psd_correction,
            psd_tol=float(self.psd_tol),
            approximation=approximation,
            nystrom_components=self.nystrom_components,
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


@dataclass(frozen=True)
class FeatureBundle:
    name: str
    features: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": tuple(int(size) for size in self.features.shape),
            "diagnostics": _jsonable(self.diagnostics),
        }


@dataclass(frozen=True)
class KernelBundle:
    name: str
    train_kernel: np.ndarray
    test_kernel: np.ndarray | None = None
    train_features: np.ndarray | None = None
    test_features: np.ndarray | None = None
    is_psd: bool = True
    psd_correction: str | None = None
    complexity: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "train_shape": tuple(int(size) for size in self.train_kernel.shape),
            "test_shape": None if self.test_kernel is None else tuple(int(size) for size in self.test_kernel.shape),
            "is_psd": bool(self.is_psd),
            "psd_correction": self.psd_correction,
            "complexity": _jsonable(self.complexity),
            "diagnostics": _jsonable(self.diagnostics),
        }
        if self.train_features is not None:
            payload["train_features_shape"] = tuple(int(size) for size in self.train_features.shape)
        if self.test_features is not None:
            payload["test_features_shape"] = tuple(int(size) for size in self.test_features.shape)
        return payload


@dataclass(frozen=True)
class KernelSelectionReport:
    generator_names: tuple[str, ...]
    weights: tuple[float, ...]
    selected_generators: tuple[str, ...]
    selected_weights: tuple[float, ...]
    scores: dict[str, float]
    alignments: dict[str, float]
    complexities: dict[str, float]
    redundancies: dict[str, float]
    task_type: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_names": list(self.generator_names),
            "weights": [float(weight) for weight in self.weights],
            "selected_generators": list(self.selected_generators),
            "selected_weights": [float(weight) for weight in self.selected_weights],
            "scores": _jsonable(self.scores),
            "alignments": _jsonable(self.alignments),
            "complexities": _jsonable(self.complexities),
            "redundancies": _jsonable(self.redundancies),
            "task_type": self.task_type,
            "diagnostics": _jsonable(self.diagnostics),
        }


class FeatureGeneratorProtocol(Protocol):
    name: str

    def fit(self, X: Any, y: Any | None = None, *, task_type: str = "classification"):
        ...

    def transform(self, X: Any) -> FeatureBundle:
        ...

    def fit_transform(self, X: Any, y: Any | None = None, *, task_type: str = "classification") -> FeatureBundle:
        ...


class KernelGeneratorProtocol(FeatureGeneratorProtocol, Protocol):
    def kernel(self, X_left: Any, X_right: Any | None = None) -> KernelBundle:
        ...
