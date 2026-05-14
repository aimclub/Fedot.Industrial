from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fedot_ind.core.kernel_learning.contracts import KernelBundle, KernelSelectionReport
from fedot_ind.core.kernel_learning.kernels import kernel_complexity
from fedot_ind.core.kernel_learning.selection.targets import TargetKernelBuilder


def _center_kernel(kernel: np.ndarray) -> np.ndarray:
    matrix = np.asarray(kernel, dtype=float)
    row_mean = matrix.mean(axis=1, keepdims=True)
    column_mean = matrix.mean(axis=0, keepdims=True)
    return matrix - row_mean - column_mean + float(matrix.mean())


def _normalized_inner(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = _center_kernel(left)
    right_centered = _center_kernel(right)
    denominator = float(np.linalg.norm(left_centered, ord="fro") * np.linalg.norm(right_centered, ord="fro"))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def combine_kernels(kernels: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    kernel_list = [np.asarray(kernel, dtype=float) for kernel in kernels]
    weight_array = np.asarray(tuple(weights), dtype=float)
    if not kernel_list:
        raise ValueError("At least one kernel is required for combination.")
    combined = np.zeros_like(kernel_list[0], dtype=float)
    for kernel, weight in zip(kernel_list, weight_array):
        combined += float(weight) * kernel
    return combined


@dataclass
class SparseMKLSelector:
    complexity_penalty: float = 0.01
    redundancy_penalty: float = 0.05
    min_weight: float = 0.05
    target_gamma: str | float = "scale"

    def fit(self, kernel_bundles: list[KernelBundle], y, *, task_type: str) -> KernelSelectionReport:
        if not kernel_bundles:
            raise ValueError("SparseMKLSelector requires at least one KernelBundle.")

        target_kernel = TargetKernelBuilder(task_type=task_type, gamma=self.target_gamma).build(y)
        names = tuple(bundle.name for bundle in kernel_bundles)
        train_kernels = [bundle.train_kernel for bundle in kernel_bundles]

        alignments = {
            bundle.name: max(0.0, _normalized_inner(bundle.train_kernel, target_kernel))
            for bundle in kernel_bundles
        }
        complexities = {
            bundle.name: float(bundle.complexity.get("kernel_complexity", kernel_complexity(bundle.train_kernel)))
            for bundle in kernel_bundles
        }
        redundancies = self._compute_redundancies(names, train_kernels)
        scores = {
            name: (
                    alignments[name]
                    - self.complexity_penalty * complexities[name]
                    - self.redundancy_penalty * redundancies[name]
            )
            for name in names
        }
        weights = self._scores_to_simplex(names, scores)
        selected = tuple(name for name, weight in zip(names, weights) if weight > 0.0)
        selected_weights = tuple(float(weight) for weight in weights if weight > 0.0)
        self.report_ = KernelSelectionReport(
            generator_names=names,
            weights=tuple(float(weight) for weight in weights),
            selected_generators=selected,
            selected_weights=selected_weights,
            scores={name: float(scores[name]) for name in names},
            alignments={name: float(alignments[name]) for name in names},
            complexities={name: float(complexities[name]) for name in names},
            redundancies={name: float(redundancies[name]) for name in names},
            task_type=task_type,
            diagnostics={
                "complexity_penalty": float(self.complexity_penalty),
                "redundancy_penalty": float(self.redundancy_penalty),
                "min_weight": float(self.min_weight),
            },
        )
        return self.report_

    def _compute_redundancies(self, names: tuple[str, ...], kernels: list[np.ndarray]) -> dict[str, float]:
        if len(kernels) <= 1:
            return {name: 0.0 for name in names}
        result = {}
        for index, name in enumerate(names):
            similarities = [
                max(0.0, _normalized_inner(kernels[index], other))
                for other_index, other in enumerate(kernels)
                if other_index != index
            ]
            result[name] = float(np.mean(similarities)) if similarities else 0.0
        return result

    def _scores_to_simplex(self, names: tuple[str, ...], scores: dict[str, float]) -> np.ndarray:
        raw = np.asarray([max(0.0, scores[name]) for name in names], dtype=float)
        if float(np.sum(raw)) <= 1e-12:
            return np.ones(len(names), dtype=float) / len(names)
        weights = raw / np.sum(raw)
        weights[weights < self.min_weight] = 0.0
        if float(np.sum(weights)) <= 1e-12:
            weights = np.zeros(len(names), dtype=float)
            weights[int(np.argmax(raw))] = 1.0
            return weights
        return weights / np.sum(weights)
