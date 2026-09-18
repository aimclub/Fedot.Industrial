"""Pair feature builders for Pairwise Difference Learning."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import PairwiseLearningConfig, SUPPORTED_PAIR_FEATURE_MODES

try:  # pragma: no cover - exercised through backend resolution in tests when torch is installed.
    import torch
except Exception:  # pragma: no cover
    torch = None


def normalize_feature_matrix(features: Any) -> np.ndarray:
    """Normalize time-series inputs to a finite 2D feature matrix."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def resolve_pairwise_backend(backend: str = "auto") -> tuple[str, Any | None]:
    """Map a user-facing backend name to an internal backend id and torch device."""
    normalized = str(backend or "auto").lower()
    if normalized in {"numpy", "np"}:
        return "numpy", None
    if normalized in {"auto", "torch", "cpu", "cuda"}:
        if torch is None:
            if normalized == "cuda":
                raise RuntimeError(
                    "PDL backend='cuda' was requested, but torch is unavailable."
                )
            return "numpy", None
        if normalized == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "PDL backend='cuda' was requested, but CUDA is unavailable."
                )
            return "torch_cuda", torch.device("cuda")
        if normalized == "auto" and torch.cuda.is_available():
            return "torch_cuda", torch.device("cuda")
        return "torch_cpu", torch.device("cpu")
    raise ValueError(f"Unsupported PDL backend={backend!r}.")


def pair_feature_dim(n_features: int, mode: str) -> int:
    """Return the pair-feature width for a given base feature count and mode."""
    if mode in {"concat_diff", "concat_absdiff"}:
        return n_features * 3
    if mode == "diff_only":
        return n_features
    raise ValueError(f"Unsupported pair feature mode={mode!r}.")


def _combine_pair_blocks(
    left: np.ndarray, anchors: np.ndarray, difference: np.ndarray, mode: str
) -> np.ndarray:
    if mode == "concat_diff":
        return np.hstack((left, anchors, difference)).astype(np.float32, copy=False)
    if mode == "concat_absdiff":
        return np.hstack((left, anchors, np.abs(difference))).astype(
            np.float32, copy=False
        )
    if mode == "diff_only":
        return difference.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported pair feature mode={mode!r}.")


def _build_pair_features_numpy(
    left: np.ndarray, anchors: np.ndarray, mode: str
) -> np.ndarray:
    left_repeated = np.repeat(left, len(anchors), axis=0)
    anchors_tiled = np.tile(anchors, (len(left), 1))
    difference = left_repeated - anchors_tiled
    return _combine_pair_blocks(left_repeated, anchors_tiled, difference, mode)


def _build_pair_features_torch(
    left: np.ndarray, anchors: np.ndarray, mode: str, device: Any
) -> np.ndarray:
    left_tensor = torch.as_tensor(left, dtype=torch.float32, device=device)
    anchor_tensor = torch.as_tensor(
        anchors, dtype=torch.float32, device=device)
    left_repeated = left_tensor.repeat_interleave(
        anchor_tensor.shape[0], dim=0)
    anchors_tiled = anchor_tensor.repeat(left_tensor.shape[0], 1)
    difference = left_repeated - anchors_tiled
    if mode == "concat_diff":
        paired = torch.cat((left_repeated, anchors_tiled, difference), dim=1)
    elif mode == "concat_absdiff":
        paired = torch.cat((left_repeated, anchors_tiled,
                           torch.abs(difference)), dim=1)
    elif mode == "diff_only":
        paired = difference
    else:  # pragma: no cover - normalized config prevents this branch.
        raise ValueError(f"Unsupported pair feature mode={mode!r}.")
    return paired.detach().cpu().numpy()


class _ModePairFeatureBuilder:
    def __init__(self, mode: str, *, backend: str):
        self._mode = mode
        self._backend = backend

    def build(self, left: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Build pair features for every (left, anchor) cross-product row."""
        left = normalize_feature_matrix(left)
        anchors = normalize_feature_matrix(anchors)
        if left.shape[1] != anchors.shape[1]:
            raise ValueError(
                f"Pair matrices must have equal feature counts: {left.shape[1]} != {anchors.shape[1]}."
            )
        backend_name, device = resolve_pairwise_backend(self._backend)
        if backend_name == "numpy":
            return _build_pair_features_numpy(left, anchors, self._mode)
        return _build_pair_features_torch(left, anchors, self._mode, device)


class ConcatDiffPairFeatureBuilder(_ModePairFeatureBuilder):
    def __init__(self, *, backend: str = "auto"):
        super().__init__("concat_diff", backend=backend)


class ConcatAbsdiffPairFeatureBuilder(_ModePairFeatureBuilder):
    def __init__(self, *, backend: str = "auto"):
        super().__init__("concat_absdiff", backend=backend)


class DiffOnlyPairFeatureBuilder(_ModePairFeatureBuilder):
    def __init__(self, *, backend: str = "auto"):
        super().__init__("diff_only", backend=backend)


def resolve_pair_feature_builder(config: PairwiseLearningConfig):
    """Instantiate the pair-feature builder implied by ``config.pair_feature_mode``."""
    config = config.normalized()
    builders = {
        "concat_diff": ConcatDiffPairFeatureBuilder,
        "concat_absdiff": ConcatAbsdiffPairFeatureBuilder,
        "diff_only": DiffOnlyPairFeatureBuilder,
    }
    # As a rule, this error will not appear because there is such a check in config.py or
    # it can raises if there is mistake in builders = {...}
    if config.pair_feature_mode not in builders:
        raise ValueError(
            f"Unsupported pair_feature_mode={config.pair_feature_mode!r}. "
            f"Supported modes: {SUPPORTED_PAIR_FEATURE_MODES}."
            f"Supported builders: {list(builders.keys())}"
        )
    return builders[config.pair_feature_mode](backend=config.backend)


def build_pair_features(
    left_features: Any,
    anchor_features: Any,
    config: PairwiseLearningConfig,
) -> np.ndarray:
    """Backward-compatible wrapper around the configured pair-feature builder."""
    config = config.normalized()
    builder = resolve_pair_feature_builder(config)
    return builder.build(left_features, anchor_features)
