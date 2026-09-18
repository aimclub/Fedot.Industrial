"""Torch-backed product-manifold features for multichannel time series."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import torch
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum

from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.representation.manifold.torch_mdm import (
    TorchClassCentroids,
    TorchMDMDistances,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_spd import (
    BaseTorchSPDBuilder,
    DenseSPDBatch,
    DenseSPDReference,
    RaggedBlockSPDBatch,
    RaggedBlockSPDReference,
    TorchBlockCovariances,
    TorchCoSpectra,
    TorchCovariances,
    TorchShrinkage,
    TorchSPDBatch,
    TorchSPDReference,
    UniformBlockSPDBatch,
    UniformBlockSPDReference,
)
from fedot_ind.core.operation.transformation.representation.manifold.torch_tangent_space import (
    TorchSPDCentroid,
    TorchTangentSpace,
)


class RiemannExtractor(BaseExtractor):
    """Generate weighted product-manifold tangent and MDM features.

    Every configured view builds one structured SPD representation. Views stay
    separate during centroid estimation, projection, and distance evaluation;
    only their resulting Euclidean features are combined. A view weight is
    divided equally between its atomic SPD blocks so representations such as
    co-spectra do not dominate solely through their block count.

    Args:
        params: Extractor configuration with the following keys:

            - ``views``: Required ordered list of SPD-view mappings. Every view
              contains a unique ``name``, a registered ``builder`` name,
              positive ``weight`` (default ``1.0``), ``shrinkage`` in
              ``[0, 1]`` (default ``0.1``), and builder-specific ``params``.
            - ``feature_mode``: ``"tangent"``, ``"mdm"``, or ``"both"``.
              Default is ``"both"``; tangent coordinates precede MDM features.
            - ``tangent_metric``: Metric for tangent centroids and projection:
              ``"riemann"``, ``"logeuclid"``, or ``"euclid"``. Default is
              ``"riemann"``.
            - ``mdm_metric``: Metric for MDM centroids and distances with the
              same choices. Default is ``"riemann"``.
            - ``mdm_centroid_scope``: ``"class"`` for one product centroid per
              sorted class or ``"global"`` for one centroid. Default is
              ``"class"``. Class MDM requires ``input_data.target`` during fit.
            - ``centroid_type``: ``"mean"`` or ``"median"``. Default is
              ``"mean"``. A product mean is componentwise; a product median is
              coupled through one weighted product distance.
            - ``centroid_params``: Keyword arguments forwarded to
              :class:`TorchSPDCentroid`, including convergence tolerances,
              iteration limits, and ``eigenvalue_floor``.
            - ``torch_device``: Torch device such as ``"cpu"`` or ``"cuda"``.
              Default is ``"cpu"``.
            - ``torch_dtype``: ``"float64"`` / ``torch.float64`` or
              ``"float32"`` / ``torch.float32``. Default is ``"float64"``.

        Builder-specific ``params`` are:

            - ``covariance``: ``estimator`` and nested ``estimator_params``;
            - ``grouped_covariance``: ``group_sizes``, ``estimator``, and
              nested ``estimator_params``;
            - ``cospectra``: ``window``, ``overlap``, ``fmin``, ``fmax``, and
              ``fs``.

    Example:
        A raw covariance view and a frequency-block co-spectral view::

            params = {
                "views": [
                    {
                        "name": "raw",
                        "builder": "covariance",
                        "weight": 1.0,
                        "shrinkage": 0.1,
                        "params": {
                            "estimator": "scm",
                            "estimator_params": {},
                        },
                    },
                    {
                        "name": "spectral",
                        "builder": "cospectra",
                        "weight": 1.0,
                        "shrinkage": 0.1,
                        "params": {
                            "window": 128,
                            "overlap": 0.75,
                            "fmin": 1.0,
                            "fmax": 32.0,
                            "fs": 100.0,
                        },
                    },
                ],
                "feature_mode": "both",
                "tangent_metric": "riemann",
                "mdm_metric": "riemann",
                "mdm_centroid_scope": "class",
                "centroid_type": "mean",
                "centroid_params": {},
                "torch_device": "cpu",
                "torch_dtype": "float64",
            }
    """

    _SPD_BUILDERS = {
        "covariance": TorchCovariances,
        "grouped_covariance": TorchBlockCovariances,
        "cospectra": TorchCoSpectra,
    }
    _SUPPORTED_METRICS = {"riemann", "logeuclid", "euclid"}
    _UNPORTED_METRICS = {"logdet", "kullback", "wasserstein"}
    _LEGACY_PARAMS = {
        "estimator",
        "estimator_params",
        "SPD_metric",
        "extraction_strategy",
        "extraction_method",
        "representation_type",
        "block_sizes",
        "fmin",
        "fmax",
        "fs",
        "shrinkage",
        "centroid_strategy",
    }

    def __init__(self, params: Optional[OperationParameters] = None):
        """Initialise view builders and validate the product configuration."""
        params = params or {}
        super().__init__(params)
        config = params.to_dict() if isinstance(params, OperationParameters) else dict(params)

        legacy_params = sorted(self._LEGACY_PARAMS.intersection(config))
        if legacy_params:
            raise ValueError(
                "Legacy RiemannExtractor parameters are not supported: "
                f"{legacy_params}. Configure SPD representations through 'views'."
            )

        self.views = self._validate_views(config.get("views"))
        self.feature_mode = config.get("feature_mode", "both")
        self.tangent_metric = config.get("tangent_metric", "riemann")
        self.mdm_metric = config.get("mdm_metric", "riemann")
        self.mdm_centroid_scope = config.get("mdm_centroid_scope", "class")
        self.centroid_type = config.get("centroid_type", "mean")
        self.centroid_params = config.get("centroid_params", {})
        self.device = torch.device(config.get("torch_device", "cpu"))
        self.dtype = self._resolve_dtype(config.get("torch_dtype", torch.float64))
        self._validate_geometry_params()

        self.view_builders_ = {
            view["name"]: self._make_spd_builder(view) for view in self.views
        }
        self.view_shrinkages_ = {
            view["name"]: TorchShrinkage(view["shrinkage"]) for view in self.views
        }
        self.view_weights_ = {view["name"]: view["weight"] for view in self.views}

        self.view_block_counts_: dict[str, int] = {}
        self.view_slices_: dict[str, slice] = {}
        self.product_block_weights_: tuple[float, ...] = ()
        self.tangent_product_centroid_: Optional[dict[str, TorchSPDReference]] = None
        self.tangent_spaces_: dict[str, TorchTangentSpace] = {}
        self.class_product_centroids_: Optional[
            tuple[dict[str, TorchSPDReference], ...]
        ] = None
        self.mdm_distances_: dict[str, TorchMDMDistances] = {}
        self.classes_: Optional[np.ndarray] = None
        self.effective_feature_mode_: Optional[str] = None
        self.is_fitted = False
        self.predict = None
        self._fit_features_cache_: Optional[np.ndarray] = None
        self._fit_input_signature_: Optional[tuple[int, tuple[int, ...]]] = None

        self.logging_params.update({
            "views": self.views,
            "feature_mode": self.feature_mode,
            "tangent_metric": self.tangent_metric,
            "mdm_metric": self.mdm_metric,
            "mdm_centroid_scope": self.mdm_centroid_scope,
            "centroid_type": self.centroid_type,
            "torch_device": str(self.device),
            "torch_dtype": str(self.dtype),
        })

    @staticmethod
    def _resolve_dtype(dtype: Any) -> torch.dtype:
        """Resolve a user-facing floating-point dtype name to ``torch.dtype``."""
        if isinstance(dtype, str):
            try:
                dtype = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"Unknown torch_dtype: {dtype!r}.") from error
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("torch_dtype must be torch.float32 or torch.float64.")
        return dtype

    @classmethod
    def _validate_views(cls, views: Any) -> tuple[dict[str, Any], ...]:
        """Validate and normalise ordered SPD-view configurations."""
        if not isinstance(views, (list, tuple)) or not views:
            raise ValueError("views must contain at least one SPD view configuration.")

        validated = []
        names = set()
        for index, raw_view in enumerate(views):
            if not isinstance(raw_view, Mapping):
                raise TypeError(f"views[{index}] must be a mapping.")
            view = dict(raw_view)
            name = view.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"views[{index}]['name'] must be a non-empty string.")
            if name in names:
                raise ValueError("View names must be unique.")
            names.add(name)

            builder = view.get("builder")
            if builder not in cls._SPD_BUILDERS:
                raise ValueError(
                    f"Unknown SPD builder: {builder!r}. "
                    f"Available builders are: {sorted(cls._SPD_BUILDERS)}."
                )
            try:
                weight = float(view.get("weight", 1.0))
                shrinkage = float(view.get("shrinkage", 0.1))
            except (TypeError, ValueError) as error:
                raise TypeError("View weight and shrinkage must be real numbers.") from error
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("View weight must be finite and strictly positive.")
            if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
                raise ValueError("View shrinkage must be finite and between 0 and 1.")
            builder_params = view.get("params", {})
            if not isinstance(builder_params, Mapping):
                raise TypeError(f"views[{index}]['params'] must be a mapping.")
            validated.append({
                "name": name,
                "builder": builder,
                "weight": weight,
                "shrinkage": shrinkage,
                "params": dict(builder_params),
            })
        return tuple(validated)

    def _validate_geometry_params(self) -> None:
        """Validate feature families, centroid settings, metrics, and device."""
        if self.feature_mode not in {"tangent", "mdm", "both"}:
            raise ValueError("feature_mode must be 'tangent', 'mdm', or 'both'.")
        if self.mdm_centroid_scope not in {"class", "global"}:
            raise ValueError("mdm_centroid_scope must be 'class' or 'global'.")
        if self.centroid_type not in {"mean", "median"}:
            raise ValueError("centroid_type must be 'mean' or 'median'.")
        if not isinstance(self.centroid_params, Mapping):
            raise TypeError("centroid_params must be a mapping.")
        self.centroid_params = dict(self.centroid_params)
        for name, metric in (
            ("tangent_metric", self.tangent_metric),
            ("mdm_metric", self.mdm_metric),
        ):
            if metric in self._UNPORTED_METRICS:
                raise NotImplementedError(
                    f"Metric '{metric}' is not implemented for the Torch backend. "
                    f"Supported metrics are: {sorted(self._SUPPORTED_METRICS)}."
                )
            if metric not in self._SUPPORTED_METRICS:
                raise ValueError(f"Unsupported {name}: '{metric}'.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("torch_device='cuda' was requested, but CUDA is not available.")

    def __repr__(self) -> str:
        """Return the short representation used by the extractor infrastructure."""
        return "Riemann Manifold Class for TS representation"

    def _make_spd_builder(self, view: Mapping[str, Any]) -> BaseTorchSPDBuilder:
        """Instantiate one built-in or registered SPD builder for a view."""
        builder_name = view["builder"]
        builder_class = self._SPD_BUILDERS[builder_name]
        builder = builder_class.from_params(view["params"])
        if not isinstance(builder, BaseTorchSPDBuilder):
            raise TypeError(
                f"Registered builder '{builder_name}' must inherit BaseTorchSPDBuilder."
            )
        return builder

    def _prepare_tensor(self, x: Any) -> torch.Tensor:
        """Sanitise input and return signals in ``(samples, channels, time)`` layout."""
        tensor = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 3:
            raise ValueError("Input data must have one, two, or three dimensions.")
        if tensor.shape[1] == 1:
            warnings.warn(
                "Input data is univariate (single channel). RiemannExtractor evaluates "
                "cross-channel spatial covariance, so its manifold representation is trivial. "
                "Build a trajectory matrix before extraction when appropriate.",
                UserWarning,
                stacklevel=2,
            )
        return tensor

    def _build_views(
        self,
        signals: torch.Tensor,
        *,
        fit: bool,
        labels: Optional[np.ndarray] = None,
    ) -> dict[str, TorchSPDBatch]:
        """Build and shrink every configured SPD view without dense expansion."""
        result = {}
        for view in self.views:
            name = view["name"]
            builder = self.view_builders_[name]
            shrinkage = self.view_shrinkages_[name]
            if fit:
                spd = builder.fit_transform_spd(signals, y=labels)
                result[name] = shrinkage.fit_transform(spd, y=labels)
            else:
                result[name] = shrinkage.transform(builder.transform_spd(signals))
        return result

    @staticmethod
    def _atomic_blocks(spd: TorchSPDBatch) -> tuple[torch.Tensor, ...]:
        """Return one batched matrix tensor per atomic SPD block."""
        if isinstance(spd, DenseSPDBatch):
            return (spd.matrices,)
        if isinstance(spd, UniformBlockSPDBatch):
            return tuple(spd.matrices[:, index] for index in range(spd.matrices.shape[1]))
        if isinstance(spd, RaggedBlockSPDBatch):
            return spd.matrices
        raise TypeError(f"Unsupported SPD batch type: {type(spd).__name__}.")

    def _record_product_layout(self, views: Mapping[str, TorchSPDBatch]) -> None:
        """Record view slices and distribute each view weight over its blocks."""
        sample_counts = {spd.n_samples for spd in views.values()}
        if len(sample_counts) != 1:
            raise ValueError("All SPD views must contain the same number of samples.")
        offset = 0
        block_weights = []
        for view in self.views:
            name = view["name"]
            block_count = len(self._atomic_blocks(views[name]))
            self.view_block_counts_[name] = block_count
            self.view_slices_[name] = slice(offset, offset + block_count)
            gamma = self.view_weights_[name] / block_count
            block_weights.extend([gamma] * block_count)
            offset += block_count
        self.product_block_weights_ = tuple(block_weights)

    def _pack_product(self, views: Mapping[str, TorchSPDBatch]) -> RaggedBlockSPDBatch:
        """Pack atomic blocks into the existing ragged product representation."""
        blocks = tuple(
            block
            for view in self.views
            for block in self._atomic_blocks(views[view["name"]])
        )
        return RaggedBlockSPDBatch(blocks)

    def _split_product_reference(
        self,
        reference: TorchSPDReference,
        views: Mapping[str, TorchSPDBatch],
    ) -> dict[str, TorchSPDReference]:
        """Restore per-view reference types after a joint product median."""
        if not isinstance(reference, RaggedBlockSPDReference):
            raise TypeError("A packed product centroid must be a ragged reference.")
        result = {}
        for view in self.views:
            name = view["name"]
            spd = views[name]
            blocks = reference.matrices[self.view_slices_[name]]
            if isinstance(spd, DenseSPDBatch):
                result[name] = DenseSPDReference(blocks[0])
            elif isinstance(spd, UniformBlockSPDBatch):
                result[name] = UniformBlockSPDReference(torch.stack(blocks))
            else:
                result[name] = RaggedBlockSPDReference(tuple(blocks))
        return result

    def _fit_product_centroid(
        self,
        views: Mapping[str, TorchSPDBatch],
        metric: str,
    ) -> dict[str, TorchSPDReference]:
        """Fit one product mean componentwise or one coupled product median."""

        if self.centroid_type == "mean":
            return {
                view["name"]: TorchSPDCentroid(
                    metric=metric,
                    centroid_type="mean",
                    **self.centroid_params,
                ).fit(views[view["name"]]).centroid_
                for view in self.views
            }
        
        if self.centroid_type == "median":
            packed = self._pack_product(views)
            reference = TorchSPDCentroid(
                metric=metric,
                centroid_type="median",
                **self.centroid_params,
            ).fit(packed, block_weights=self.product_block_weights_).centroid_
            return self._split_product_reference(reference, views)

    def _fit_class_product_centroids(
        self,
        views: Mapping[str, TorchSPDBatch],
        labels: np.ndarray,
    ) -> tuple[dict[str, TorchSPDReference], ...]:
        """Fit one componentwise mean or coupled product median per class."""

        if self.centroid_type == "median":
            estimator = TorchClassCentroids(
                metric=self.mdm_metric,
                centroid_type="median",
                **self.centroid_params,
            ).fit(
                self._pack_product(views),
                labels,
                block_weights=self.product_block_weights_,
            )
            self.classes_ = estimator.classes_
            return tuple(
                self._split_product_reference(reference, views)
                for reference in estimator.centroids_
            )

        if self.centroid_type == "mean":
            estimators = {
                view["name"]: TorchClassCentroids(
                    metric=self.mdm_metric,
                    centroid_type="mean",
                    **self.centroid_params,
                ).fit(views[view["name"]], labels)
                for view in self.views
            }
            first = estimators[self.views[0]["name"]]
            self.classes_ = first.classes_
            if any(
                not np.array_equal(estimator.classes_, self.classes_)
                for estimator in estimators.values()
            ):
                raise RuntimeError("All SPD views must produce the same sorted classes.")
            return tuple(
                {
                    view["name"]: estimators[view["name"]].centroids_[class_index]
                    for view in self.views
                }
                for class_index in range(len(self.classes_))
            )

    def _fit_geometry(
        self,
        views: Mapping[str, TorchSPDBatch],
        labels: Optional[np.ndarray],
    ) -> None:
        """Fit tangent references and product MDM centroids requested by the mode."""

        eigenvalue_floor = self.centroid_params.get("eigenvalue_floor")
        if self.effective_feature_mode_ in {"tangent", "both"}:
            self.tangent_product_centroid_ = self._fit_product_centroid(
                views, self.tangent_metric
            )
            self.tangent_spaces_ = {
                view["name"]: TorchTangentSpace(
                    self.tangent_product_centroid_[view["name"]],
                    metric=self.tangent_metric,
                    eigenvalue_floor=eigenvalue_floor,
                )
                for view in self.views
            }

        if self.effective_feature_mode_ in {"mdm", "both"}:
            if self.mdm_centroid_scope == "class":
                if labels is None or labels.size == 0:
                    raise ValueError("Target data is required to fit class MDM centroids.")
                product_centroids = self._fit_class_product_centroids(views, labels)
            else:
                self.classes_ = None
                product_centroids = (self._fit_product_centroid(views, self.mdm_metric),)
            self.class_product_centroids_ = product_centroids
            self.mdm_distances_ = {
                view["name"]: TorchMDMDistances(
                    [centroid[view["name"]] for centroid in product_centroids],
                    metric=self.mdm_metric,
                    eigenvalue_floor=eigenvalue_floor,
                )
                for view in self.views
            }

    def _features_from_views(
        self,
        views: Mapping[str, TorchSPDBatch],
    ) -> torch.Tensor:
        """Project views and concatenate tangent then product-MDM features."""
        parts = []
        if self.effective_feature_mode_ in {"tangent", "both"}:
            tangent_parts = []
            for view in self.views:
                name = view["name"]
                gamma = self.view_weights_[name] / self.view_block_counts_[name]
                tangent_parts.append(
                    math.sqrt(gamma) * self.tangent_spaces_[name].transform(views[name])
                )
            parts.append(torch.cat(tangent_parts, dim=1))

        if self.effective_feature_mode_ in {"mdm", "both"}:
            squared_distance = None
            for view in self.views:
                name = view["name"]
                gamma = self.view_weights_[name] / self.view_block_counts_[name]
                contribution = gamma * self.mdm_distances_[name].transform(views[name]).square()
                squared_distance = (
                    contribution
                    if squared_distance is None
                    else squared_distance + contribution
                )
            parts.append(torch.sqrt(torch.clamp_min(squared_distance, 0.0)))

        return torch.cat(parts, dim=1)

    def _effective_mode(self, input_data: InputData) -> str:
        """Resolve regression-specific feature-mode semantics."""
        task_type = getattr(getattr(input_data, "task", None), "task_type", None)
        if task_type != TaskTypesEnum.regression:
            return self.feature_mode
        if self.feature_mode == "mdm":
            raise ValueError("feature_mode='mdm' is not supported for regression.")
        return "tangent" if self.feature_mode == "both" else self.feature_mode

    @staticmethod
    def _input_signature(input_data: InputData) -> tuple[int, tuple[int, ...]]:
        """Return a lightweight identity signature for the one-shot train cache."""
        features = input_data.features
        return id(features), tuple(features.shape)

    def _reset_geometry_state(self) -> None:
        """Discard all geometry and one-shot train features before refitting."""
        self.view_block_counts_ = {}
        self.view_slices_ = {}
        self.product_block_weights_ = ()
        self.tangent_product_centroid_ = None
        self.tangent_spaces_ = {}
        self.class_product_centroids_ = None
        self.mdm_distances_ = {}
        self.classes_ = None
        self.effective_feature_mode_ = None
        self.is_fitted = False
        self.predict = None
        self._fit_features_cache_ = None
        self._fit_input_signature_ = None

    def _to_numpy_features(self, features: torch.Tensor) -> np.ndarray:
        """Clean Torch feature values and move them to a NumPy array."""
        return self._clean_predict_torch(features).detach().cpu().numpy()

    def _take_fit_features(self, input_data: InputData) -> Optional[np.ndarray]:
        """Consume cached train features if they belong to the supplied input object."""
        if self._fit_features_cache_ is None:
            return None
        if self._fit_input_signature_ != self._input_signature(input_data):
            self._fit_features_cache_ = None
            self._fit_input_signature_ = None
            return None
        features = self._fit_features_cache_
        self._fit_features_cache_ = None
        self._fit_input_signature_ = None
        self.predict = features
        return features

    def fit(self, input_data: InputData) -> "RiemannExtractor":
        """Fit all product geometry and cache its already-computed train features."""
        self._reset_geometry_state()
        self.effective_feature_mode_ = self._effective_mode(input_data)
        labels = None if input_data.target is None else np.asarray(input_data.target).reshape(-1)
        signals = self._prepare_tensor(input_data.features)
        views = self._build_views(signals, fit=True, labels=labels)
        self._record_product_layout(views)
        self._fit_geometry(views, labels)
        features = self._to_numpy_features(self._features_from_views(views))
        self._fit_features_cache_ = features
        self._fit_input_signature_ = self._input_signature(input_data)
        self.predict = features
        self.is_fitted = True
        return self

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        """Return one-shot train features, rebuilding only for mismatched input."""
        features = self._take_fit_features(input_data)
        if features is None:
            return self.transform(input_data, use_cache=self.use_cache)
        return self._convert_to_fedot_datatype(input_data, features)

    def fit_transform(self, input_data: InputData) -> OutputData:
        """Fit product geometry and return train features without rebuilding SPD views."""
        return self.fit(input_data).transform_for_fit(input_data)

    def _transform(self, input_data: InputData) -> np.ndarray:
        """Apply fitted product geometry or auto-fit once when no state exists."""
        if not self.is_fitted:
            warnings.warn(
                "RiemannExtractor is not fitted. Calling 'fit' inside 'transform' with provided input data. "
                "Warning: If this is test data, it may cause data leakage.",
                UserWarning,
                stacklevel=2,
            )
            self.fit(input_data)
            features = self._take_fit_features(input_data)
            if features is None:
                raise RuntimeError("Auto-fit did not produce train features.")
            return features

        views = self._build_views(self._prepare_tensor(input_data.features), fit=False)
        self.predict = self._to_numpy_features(self._features_from_views(views))
        return self.predict
