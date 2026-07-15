from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class Either(Generic[T]):
    """Minimal Either container used by the OKHS-DMD reference checks."""

    def __init__(self, value: T, *, right: bool):
        self.value = value
        self._right = bool(right)

    @classmethod
    def right(cls, value: T) -> "Either[T]":
        return cls(value, right=True)

    @classmethod
    def left(cls, value: T) -> "Either[T]":
        return cls(value, right=False)

    def is_right(self) -> bool:
        return self._right

    def is_left(self) -> bool:
        return not self._right

    def either(self, left: Callable[[T], U], right: Callable[[T], U]) -> U:
        return right(self.value) if self._right else left(self.value)


class OKHSMethod(Enum):
    DIRECT = "direct"
    DMD = "dmd"
    OCCUPATION = "occupation"


@dataclass(frozen=True)
class HoldoutSplit:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class AnalysisReport:
    title: str
    alignments: tuple[str, ...]
    leakage_checks: tuple[str, ...]
    dependency_gaps: tuple[str, ...]


def build_analysis_report() -> AnalysisReport:
    return AnalysisReport(
        title="OKHS-DMD Alignment Analysis",
        alignments=(
            "Alignment Matrix: OKHS occupation kernels, DMD basis, and forecast head contracts",
            "Projected trajectory representation is checked before forecast decoding",
        ),
        leakage_checks=("Train/test leakage: holdout trajectories must not overlap",),
        dependency_gaps=("undeclared dependency: pymittagleffler",),
    )


def render_markdown_report(report: AnalysisReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        "## Alignment Matrix",
        *[f"- {item}" for item in report.alignments],
        "",
        "## Train/test leakage",
        *[f"- {item}" for item in report.leakage_checks],
        "",
        "## Dependency gaps",
        *[f"- {item}" for item in report.dependency_gaps],
        "",
    ]
    return "\n".join(lines)


def choose_holdout_split(total_trajectories: int, holdout_size: int) -> Either[HoldoutSplit | str]:
    if total_trajectories <= 0:
        return Either.left("Total trajectory count must be positive")
    if holdout_size <= 0:
        return Either.left("Holdout size must be positive")
    if holdout_size >= total_trajectories:
        return Either.left("Holdout size must be smaller than total trajectory count")
    split_at = total_trajectories - holdout_size
    return Either.right(
        HoldoutSplit(
            train_indices=tuple(range(split_at)),
            test_indices=tuple(range(split_at, total_trajectories)),
        )
    )


def validate_initial_segment_length(
    *,
    initial_segment_length: int,
    n_modes: int,
    n_features: int,
) -> Either[bool | str]:
    required = max(2, int(n_modes) * int(n_features))
    if int(initial_segment_length) < required:
        return Either.left(
            f"Insufficient initial segment length: got {initial_segment_length}, required at least {required}"
        )
    return Either.right(True)


def validate_liouville_shapes(
    *,
    gram_shape: tuple[int, int],
    liouville_shape: tuple[int, int],
    eigen_count: int,
) -> Either[bool | str]:
    if tuple(gram_shape) != tuple(liouville_shape):
        return Either.left(f"Gram/Liouville mismatch: {gram_shape} vs {liouville_shape}")
    if len(gram_shape) != 2 or gram_shape[0] != gram_shape[1]:
        return Either.left(f"Gram matrix must be square: {gram_shape}")
    if int(eigen_count) > min(gram_shape):
        return Either.left("Eigen count exceeds the Liouville operator dimension")
    return Either.right(True)


def normalize_okhs_method(method: str | OKHSMethod) -> Either[OKHSMethod | str]:
    if isinstance(method, OKHSMethod):
        return Either.right(method)
    normalized = str(method).strip().lower()
    aliases = {
        "direct": OKHSMethod.DIRECT,
        "dmd": OKHSMethod.DMD,
        "occupation": OKHSMethod.OCCUPATION,
        "occupation_kernel": OKHSMethod.OCCUPATION,
    }
    if normalized not in aliases:
        return Either.left(f"Unsupported OKHS method: {method}")
    return Either.right(aliases[normalized])


def detect_gaps(alignments: Iterable[str], dependency_gaps: Iterable[str]) -> list[str]:
    gaps = list(dependency_gaps)
    if not any("Alignment Matrix" in item for item in alignments):
        gaps.append("missing alignment matrix description")
    return gaps
