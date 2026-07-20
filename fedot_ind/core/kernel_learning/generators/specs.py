from __future__ import annotations

from typing import Any

from fedot_ind.core.kernel_learning.generators.base import OperationSpec

BASIS_ONLY_GENERATORS = frozenset(
    ("wavelet_basis", "fourier_basis", "eigen_basis"))

_DEFAULT_STATISTICAL_PARAMS = {
    "window_size": 10,
    "use_sliding_window": True,
    "stride": 1,
    "add_global_features": True,
    "use_cache": False,
}


def torch_quantile_spec(params: dict[str, Any]) -> OperationSpec:
    return OperationSpec(
        name="quantile_extractor_torch",
        module_path="fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor",
        class_name="TorchQuantileExtractor",
        params=params,
        use_torch=True,
    )


def wavelet_spec() -> OperationSpec:
    return OperationSpec(
        name="wavelet_basis",
        module_path="fedot_ind.core.operation.transformation.basis.wavelet",
        class_name="WaveletBasisImplementation",
        params={"wavelet": "mexh", "n_components": 2, "use_cache": False},
    )


def fourier_spec() -> OperationSpec:
    return OperationSpec(
        name="fourier_basis",
        module_path="fedot_ind.core.operation.transformation.basis.fourier",
        class_name="FourierBasisImplementation",
        params={
            "spectrum_type": "smoothed",
            "threshold": 0.9,
            "output_format": "signal",
            "approximation": "exact",
            "low_rank": 5,
            "use_cache": False,
        },
    )


def eigen_spec() -> OperationSpec:
    return OperationSpec(
        name="eigen_basis",
        module_path="fedot_ind.core.operation.transformation.basis.eigen_basis",
        class_name="EigenBasisImplementation",
        params={
            "window_size": 20,
            "rank_regularization": "explained_dispersion",
            "decomposition_type": "svd",
            "use_cache": False,
        },
    )


def recurrence_spec() -> OperationSpec:
    return OperationSpec(
        name="recurrence_extractor",
        module_path="fedot_ind.core.operation.transformation.representation.recurrence.recurrence_extractor",
        class_name="RecurrenceExtractor",
        params={
            "window_size": 10,
            "stride": 1,
            "rec_metric": "cosine",
            "use_sliding_window": False,
            "image_mode": False,
            "use_cache": False,
        },
    )


def topological_spec() -> OperationSpec:
    return OperationSpec(
        name="topological_extractor",
        module_path="fedot_ind.core.operation.transformation.representation.topological.topological_extractor",
        class_name="TopologicalExtractor",
        params={"window_size": 10, "stride": 1, "use_cache": False},
    )


def riemann_spec() -> OperationSpec:
    return OperationSpec(
        name="riemann_extractor",
        module_path="fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding",
        class_name="RiemannExtractor",
        params={
            "Classes": None,
            "estimator": "scm",
            "SPD_metric": "riemann",
            "tangent_metric": "riemann",
            "spd_space": None,
            "tangent_space": None,
            "centroid_strategy": "global",
            "centroid_type": "mean",
            "extraction_strategy": "tangent",
            "use_cache": False,
        },
    )


def tabular_spec() -> OperationSpec:
    return OperationSpec(
        name="tabular_extractor",
        module_path="fedot_ind.core.operation.transformation.representation.tabular.tabular_extractor",
        class_name="TabularExtractor",
        params={"feature_domain": "all",
                "reduce_dimension": True, "use_cache": False},
    )


_torch_quantile_spec = torch_quantile_spec
_wavelet_spec = wavelet_spec
_fourier_spec = fourier_spec
_eigen_spec = eigen_spec
_recurrence_spec = recurrence_spec
_topological_spec = topological_spec
_riemann_spec = riemann_spec
_tabular_spec = tabular_spec


__all__ = [
    "BASIS_ONLY_GENERATORS",
    "eigen_spec",
    "fourier_spec",
    "recurrence_spec",
    "riemann_spec",
    "tabular_spec",
    "topological_spec",
    "torch_quantile_spec",
    "wavelet_spec",
]
