from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class FeatureBundle:
    name: str
    features: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


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
