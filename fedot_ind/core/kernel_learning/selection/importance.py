from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fedot_ind.core.kernel_learning.contracts import KernelSelectionReport


@dataclass(frozen=True)
class KernelImportanceConfig:
    weight_threshold: float = 0.05
    fallback_top_n: int = 1
    max_union_size: int = 3

    def __post_init__(self):
        if self.weight_threshold < 0.0:
            raise ValueError("weight_threshold must be non-negative.")
        if self.fallback_top_n < 1:
            raise ValueError("fallback_top_n must be at least 1.")
        if self.max_union_size < 1:
            raise ValueError("max_union_size must be at least 1.")


@dataclass(frozen=True)
class KernelImportanceItem:
    name: str
    weight: float
    original_index: int
    rank: int
    selected_by: str


@dataclass(frozen=True)
class KernelImportanceReport:
    items: tuple[KernelImportanceItem, ...]
    selected_generators: tuple[str, ...]
    selected_weights: tuple[float, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def select_significant_generators(
        report: KernelSelectionReport,
        config: KernelImportanceConfig | None = None,
) -> KernelImportanceReport:
    config = config or KernelImportanceConfig()
    if len(report.generator_names) != len(report.weights):
        raise ValueError("KernelSelectionReport generator_names and weights must have the same length.")
    if not report.generator_names:
        raise ValueError("KernelSelectionReport must contain at least one generator.")

    ranked = sorted(
        (
            (index, name, float(weight))
            for index, (name, weight) in enumerate(zip(report.generator_names, report.weights))
        ),
        key=lambda entry: (-entry[2], entry[0]),
    )
    selected = [
        (index, name, weight, "threshold")
        for index, name, weight in ranked
        if weight >= config.weight_threshold
    ]
    used_fallback = False
    if not selected:
        selected = [
            (index, name, weight, "fallback")
            for index, name, weight in ranked[:config.fallback_top_n]
        ]
        used_fallback = True

    items = tuple(
        KernelImportanceItem(
            name=name,
            weight=weight,
            original_index=index,
            rank=rank,
            selected_by=selected_by,
        )
        for rank, (index, name, weight, selected_by) in enumerate(selected, start=1)
    )
    return KernelImportanceReport(
        items=items,
        selected_generators=tuple(item.name for item in items),
        selected_weights=tuple(item.weight for item in items),
        diagnostics={
            "weight_threshold": float(config.weight_threshold),
            "fallback_top_n": int(config.fallback_top_n),
            "max_union_size": int(config.max_union_size),
            "used_fallback": used_fallback,
        },
    )
