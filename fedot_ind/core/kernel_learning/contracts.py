from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol, Union

import numpy as np


FeatureInput = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]
TargetInput = Union[np.ndarray, Sequence[float], Sequence[str]]


class KernelConfigValidationError(ValueError):
    """Expected validation failure for Kernel Learning configuration values."""


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


class KernelApproximation(str, Enum):
    NYSTROM = "nystrom"


@dataclass(frozen=True)
class KernelMatrixPolicy:
    kernel: str = "rbf"
    gamma: str | float = "scale"
    degree: int = 3
    coef0: float = 1.0
    normalize: KernelNormalization | None = KernelNormalization.TRACE
    center: bool = False
    psd_correction: PSDCorrectionPolicy | None = PSDCorrectionPolicy.CLIP
    psd_tol: float = 1e-8
    approximation: KernelApproximation | None = None
    nystrom_components: int | None = None

    def __post_init__(self):
        if self.degree < 1:
            raise KernelConfigValidationError("degree must be at least 1.")
        if self.psd_tol < 0.0:
            raise KernelConfigValidationError("psd_tol must be non-negative.")
        object.__setattr__(
            self,
            "normalize",
            _normalize_optional_enum(self.normalize, KernelNormalization, "kernel normalization"),
        )
        object.__setattr__(
            self,
            "psd_correction",
            _normalize_optional_enum(self.psd_correction, PSDCorrectionPolicy, "PSD correction policy"),
        )
        object.__setattr__(
            self,
            "approximation",
            _normalize_optional_enum(self.approximation, KernelApproximation, "kernel approximation"),
        )
        if self.nystrom_components is not None and self.nystrom_components < 1:
            raise KernelConfigValidationError("nystrom_components must be at least 1.")

    @property
    def normalized_policy(self) -> "KernelMatrixPolicy":
        return KernelMatrixPolicy(
            kernel=str(self.kernel).lower(),
            gamma=self.gamma,
            degree=int(self.degree),
            coef0=float(self.coef0),
            normalize=self.normalize,
            center=bool(self.center),
            psd_correction=self.psd_correction,
            psd_tol=float(self.psd_tol),
            approximation=self.approximation,
            nystrom_components=self.nystrom_components,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.normalized_policy)


def _normalize_optional_enum(value: Any, enum_type: type[Enum], label: str):
    if value is None:
        return None
    if isinstance(value, enum_type):
        normalized = value
    else:
        raw_value = str(value).strip().lower()
        try:
            normalized = enum_type(raw_value)
        except ValueError as exc:
            raise KernelConfigValidationError(f"Unsupported {label}: {value}") from exc
    return None if getattr(normalized, "value", None) == "none" else normalized


def _normalize_task_type(value: Any) -> KernelTaskType:
    if isinstance(value, KernelTaskType):
        return value
    try:
        return KernelTaskType(str(value).strip().lower())
    except ValueError as exc:
        raise KernelConfigValidationError(f"Unsupported kernel task type: {value}") from exc


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
    task_type: KernelTaskType
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "task_type", _normalize_task_type(self.task_type))

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
            "task_type": self.task_type.value,
            "diagnostics": _jsonable(self.diagnostics),
        }


class FeatureGeneratorProtocol(Protocol):
    name: str

    def fit(
            self,
            X: FeatureInput,
            y: TargetInput | None = None,
            *,
            task_type: KernelTaskType | str = KernelTaskType.CLASSIFICATION,
    ):
        ...

    def transform(self, X: FeatureInput) -> FeatureBundle:
        ...

    def fit_transform(
            self,
            X: FeatureInput,
            y: TargetInput | None = None,
            *,
            task_type: KernelTaskType | str = KernelTaskType.CLASSIFICATION,
    ) -> FeatureBundle:
        ...


class KernelGeneratorProtocol(FeatureGeneratorProtocol, Protocol):
    def kernel(
            self,
            X_left: FeatureInput,
            X_right: FeatureInput | None = None,
            *,
            task_type: KernelTaskType | str | None = None,
    ) -> KernelBundle:
        ...
