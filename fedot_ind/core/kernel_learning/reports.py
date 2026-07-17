from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fedot_ind.core.kernel_learning.contracts import KernelBundle, KernelSelectionReport
from fedot_ind.core.kernel_learning.selection import KernelImportanceReport

TEXT2IMAGE_PROMPTS = {
    "contracts": (
        "Scientific diagram of time-series samples flowing into feature maps phi_j, then into valid positive "
        "semidefinite kernel matrices K_j, with a highlighted train/test contract block and PSD cone geometry."
    ),
    "sparse_mkl": (
        "Mathematical visualization of multiple kernel matrices as basis vectors combined by sparse simplex "
        "weights alpha, optimizing alignment with target kernel Y while penalizing complexity and redundancy."
    ),
    "warm_start": (
        "Algorithmic flowchart where selected kernel weights generate important feature operators, which seed "
        "a FEDOT initial population graph for classification and regression heads."
    ),
    "topology": (
        "Time series transformed into persistence diagrams and then into a topological kernel matrix, with "
        "birth-death points connected to kernel similarity heatmap."
    ),
    "shapelets": (
        "Several time-series curves with highlighted recurring shapelets, mapped into a feature space and then "
        "a sparse kernel ensemble selecting motif-based similarity."
    ),
    "embeddings": (
        "A neural embedding tower converting time series into latent vectors z, then producing RBF and cosine "
        "kernels that compete with classical statistical kernels."
    ),
    "forecast_targets": (
        "Forecasting diagram with past trajectories mapped into occupation kernels, horizon-specific target "
        "kernels Y_h, and weights changing across forecast horizons."
    ),
    "forecaster": (
        "Kernel ensemble forecaster where several kernels vote over future horizons, shown as weighted heatmaps "
        "feeding a multi-step forecast curve with uncertainty bands."
    ),
    "cache_benchmark": (
        "Performance-aware kernel learning system: exact kernels and Nystrom approximations on a compute budget "
        "axis, ending in a dashboard of selected kernels and benchmark metrics."
    ),
}


@dataclass(frozen=True)
class KernelLearningReport:
    selection: dict[str, Any]
    importance: dict[str, Any] | None = None
    kernel_bundles: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    prompts: dict[str, str] = field(default_factory=lambda: dict(TEXT2IMAGE_PROMPTS))


def build_kernel_learning_report(
        selection_report: KernelSelectionReport,
        *,
        importance_report: KernelImportanceReport | None = None,
        kernel_bundles: tuple[KernelBundle, ...] | list[KernelBundle] = (),
        include_prompts: bool = True,
) -> KernelLearningReport:
    return KernelLearningReport(
        selection=selection_report.to_dict(),
        importance=None if importance_report is None else _importance_to_dict(importance_report),
        kernel_bundles=tuple(bundle.to_dict() for bundle in kernel_bundles),
        prompts=dict(TEXT2IMAGE_PROMPTS) if include_prompts else {},
    )


def _importance_to_dict(report: KernelImportanceReport) -> dict[str, Any]:
    return {
        "items": [
            {
                "name": item.name,
                "weight": float(item.weight),
                "original_index": int(item.original_index),
                "rank": int(item.rank),
                "selected_by": item.selected_by,
            }
            for item in report.items
        ],
        "selected_generators": list(report.selected_generators),
        "selected_weights": [float(weight) for weight in report.selected_weights],
        "diagnostics": dict(report.diagnostics),
    }
