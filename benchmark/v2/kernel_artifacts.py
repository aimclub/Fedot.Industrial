from __future__ import annotations

from typing import Any

import numpy as np

from .artifacts import sanitize_artifact_payload
from .core import to_plain_data


def export_kernel_learning_artifacts(model: Any) -> dict[str, Any]:
    if model is None or not hasattr(model, "selection_report_"):
        return {}

    selection_report = getattr(model, "selection_report_", None)
    kernel_importance = getattr(model, "kernel_importance_", None)
    selected_generators = tuple(getattr(model, "selected_generators_", ()) or ())
    important_generators = tuple(getattr(model, "important_generators_", ()) or ())
    selected_weights = tuple(float(weight) for weight in (getattr(model, "selected_weights_", ()) or ()))
    important_weights = tuple(float(weight) for weight in (getattr(model, "important_weights_", ()) or ()))

    kernel_selection = {
        "selection_report": to_plain_data(selection_report),
        "kernel_importance": to_plain_data(kernel_importance),
        "selected_generators": selected_generators,
        "selected_weights": selected_weights,
        "important_generators": important_generators,
        "important_weights": important_weights,
    }
    kernel_diagnostics = {
        "kernels": _kernel_bundle_summaries(model),
    }
    summary = {
        "selected_generators": selected_generators,
        "selected_weights": selected_weights,
        "important_generators": important_generators,
        "important_weights": important_weights,
        "n_kernels": len(kernel_diagnostics["kernels"]),
    }
    return {
        "kernel_selection": sanitize_artifact_payload(kernel_selection),
        "kernel_diagnostics": sanitize_artifact_payload(kernel_diagnostics),
        "summary": sanitize_artifact_payload(summary),
    }


def _kernel_bundle_summaries(model: Any) -> list[dict[str, Any]]:
    test_bundles = {
        bundle.name: bundle
        for bundle in (getattr(model, "last_test_kernel_bundles_", ()) or ())
    }
    summaries = []
    for bundle in (getattr(model, "kernel_bundles_", ()) or ()):
        test_bundle = test_bundles.get(bundle.name)
        diagnostics = dict(getattr(bundle, "diagnostics", {}) or {})
        summaries.append(
            {
                "name": bundle.name,
                "diagnostics": diagnostics,
                "complexity": dict(getattr(bundle, "complexity", {}) or {}),
                "is_psd": bool(getattr(bundle, "is_psd", diagnostics.get("is_psd", True))),
                "psd_correction": getattr(bundle, "psd_correction", diagnostics.get("psd_correction")),
                "train_kernel_shape": _shape(getattr(bundle, "train_kernel", None)),
                "test_kernel_shape": _shape(getattr(test_bundle, "test_kernel", None)) if test_bundle else None,
            }
        )
    return summaries


def _shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(size) for size in np.asarray(value).shape]
