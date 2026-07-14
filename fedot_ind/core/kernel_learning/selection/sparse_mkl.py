from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class MKLObjectiveConfig:
    optimizer: str = "projected_gradient"
    max_iter: int = 100
    tol: float = 1e-6
    step_size: float = 0.2
    complexity_penalty: float = 0.01
    redundancy_penalty: float = 0.05
    min_weight: float = 0.05

    def __post_init__(self):
        if self.optimizer not in {"projected_gradient", "score"}:
            raise ValueError(f"Unsupported MKL optimizer: {self.optimizer}")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if self.tol < 0.0:
            raise ValueError("tol must be non-negative.")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if self.min_weight < 0.0:
            raise ValueError("min_weight must be non-negative.")
        if self.complexity_penalty < 0.0 or self.redundancy_penalty < 0.0:
            raise ValueError("MKL penalties must be non-negative.")


@dataclass(frozen=True)
class MKLOptimizationResult:
    weights: tuple[float, ...]
    objective_history: tuple[float, ...]
    iterations: int
    converged: bool
    optimizer: str


@dataclass
class AdaptiveKernelWeightSelector:
    complexity_penalty: float = 0.01
    redundancy_penalty: float = 0.05
    min_weight: float = 0.05
    target_gamma: str | float = "scale"
    optimizer: str = "projected_gradient"
    max_iter: int = 100
    tol: float = 1e-6
    step_size: float = 0.2
    objective_config_: MKLObjectiveConfig = field(init=False)

    def fit(self, kernel_bundles: list[KernelBundle], y, *, task_type: str) -> KernelSelectionReport:
        if not kernel_bundles:
            raise ValueError("AdaptiveKernelWeightSelector requires at least one KernelBundle.")

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
        redundancy_matrix = self._compute_redundancy_matrix(train_kernels)
        scores = {
            name: (
                alignments[name]
                - self.complexity_penalty * complexities[name]
                - self.redundancy_penalty * redundancies[name]
            )
            for name in names
        }
        self.objective_config_ = MKLObjectiveConfig(
            optimizer=self.optimizer,
            max_iter=self.max_iter,
            tol=self.tol,
            step_size=self.step_size,
            complexity_penalty=self.complexity_penalty,
            redundancy_penalty=self.redundancy_penalty,
            min_weight=self.min_weight,
        )
        optimization = self._optimize_weights(
            names,
            scores,
            alignments,
            complexities,
            redundancy_matrix,
        )
        weights = np.asarray(optimization.weights, dtype=float)
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
                "selector_family": "adaptive_kernel_weight_selector",
                "optimizer": optimization.optimizer,
                "iterations": int(optimization.iterations),
                "converged": bool(optimization.converged),
                "objective_history": tuple(float(value) for value in optimization.objective_history),
                "zeroed_generators": tuple(name for name, weight in zip(names, weights) if weight == 0.0),
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

    def _compute_redundancy_matrix(self, kernels: list[np.ndarray]) -> np.ndarray:
        n_kernels = len(kernels)
        matrix = np.zeros((n_kernels, n_kernels), dtype=float)
        for left_idx in range(n_kernels):
            for right_idx in range(left_idx + 1, n_kernels):
                similarity = max(0.0, _normalized_inner(kernels[left_idx], kernels[right_idx]))
                matrix[left_idx, right_idx] = similarity
                matrix[right_idx, left_idx] = similarity
        return matrix

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

    def _optimize_weights(
            self,
            names: tuple[str, ...],
            scores: dict[str, float],
            alignments: dict[str, float],
            complexities: dict[str, float],
            redundancy_matrix: np.ndarray,
    ) -> MKLOptimizationResult:
        if self.objective_config_.optimizer == "score" or len(names) == 1:
            weights = self._scores_to_simplex(names, scores)
            return MKLOptimizationResult(
                weights=tuple(float(weight) for weight in weights),
                objective_history=(
                float(self._objective(weights, names, alignments, complexities, redundancy_matrix)),),
                iterations=1,
                converged=True,
                optimizer="score",
            )

        weights = self._scores_to_simplex(names, scores)
        if _is_uniform_fallback(weights, names, scores):
            return MKLOptimizationResult(
                weights=tuple(float(weight) for weight in weights),
                objective_history=(0.0,),
                iterations=0,
                converged=True,
                optimizer="projected_gradient",
            )

        history = [float(self._objective(weights, names, alignments, complexities, redundancy_matrix))]
        converged = False
        for iteration in range(1, self.objective_config_.max_iter + 1):
            gradient = self._objective_gradient(weights, names, alignments, complexities, redundancy_matrix)
            updated = _project_to_simplex(weights + self.objective_config_.step_size * gradient)
            updated = self._apply_min_weight_threshold(updated)
            objective = float(self._objective(updated, names, alignments, complexities, redundancy_matrix))
            history.append(objective)
            if abs(history[-1] - history[-2]) <= self.objective_config_.tol:
                weights = updated
                converged = True
                break
            weights = updated

        return MKLOptimizationResult(
            weights=tuple(float(weight) for weight in weights),
            objective_history=tuple(history),
            iterations=len(history) - 1,
            converged=converged,
            optimizer="projected_gradient",
        )

    def _objective(
            self,
            weights: np.ndarray,
            names: tuple[str, ...],
            alignments: dict[str, float],
            complexities: dict[str, float],
            redundancy_matrix: np.ndarray,
    ) -> float:
        alignment_vector = np.asarray([alignments[name] for name in names], dtype=float)
        complexity_vector = np.asarray([complexities[name] for name in names], dtype=float)
        return float(
            weights @ alignment_vector
            - self.complexity_penalty * (weights @ complexity_vector)
            - self.redundancy_penalty * (weights @ redundancy_matrix @ weights)
        )

    def _objective_gradient(
            self,
            weights: np.ndarray,
            names: tuple[str, ...],
            alignments: dict[str, float],
            complexities: dict[str, float],
            redundancy_matrix: np.ndarray,
    ) -> np.ndarray:
        alignment_vector = np.asarray([alignments[name] for name in names], dtype=float)
        complexity_vector = np.asarray([complexities[name] for name in names], dtype=float)
        return (
                alignment_vector
                - self.complexity_penalty * complexity_vector
                - 2.0 * self.redundancy_penalty * (redundancy_matrix @ weights)
        )

    def _apply_min_weight_threshold(self, weights: np.ndarray) -> np.ndarray:
        thresholded = weights.copy()
        thresholded[thresholded < self.min_weight] = 0.0
        if float(np.sum(thresholded)) <= 1e-12:
            winner = int(np.argmax(weights))
            thresholded = np.zeros_like(weights)
            thresholded[winner] = 1.0
            return thresholded
        return thresholded / np.sum(thresholded)


# Backward-compatible public alias kept for existing benchmark configs and imports.
SparseMKLSelector = AdaptiveKernelWeightSelector


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.size == 0:
        return vector
    sorted_values = np.sort(vector)[::-1]
    cssv = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, vector.size + 1)
    condition = sorted_values - cssv / indices > 0
    if not np.any(condition):
        return np.ones(vector.size, dtype=float) / vector.size
    rho = indices[condition][-1]
    theta = cssv[condition][-1] / rho
    return np.maximum(vector - theta, 0.0)


def _is_uniform_fallback(weights: np.ndarray, names: tuple[str, ...], scores: dict[str, float]) -> bool:
    if not np.allclose(weights, np.ones(len(names), dtype=float) / len(names)):
        return False
    return all(max(0.0, scores[name]) <= 1e-12 for name in names)
