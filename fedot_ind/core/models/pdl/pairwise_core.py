from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

try:  # pragma: no cover - exercised through backend resolution in tests when torch is installed.
    import torch
except Exception:  # pragma: no cover
    torch = None

SUPPORTED_PAIR_FEATURE_MODES = ("concat_diff", "concat_absdiff", "diff_only")
CLASSIFICATION_SAME_LABEL = 0
CLASSIFICATION_DIFFERENT_LABEL = 1
REGRESSION_DELTA_SIGN = "left_minus_anchor"

@dataclass(frozen=True)
class PairwiseLearningConfig:
    backend: str = "auto"
    pairing_policy: str = "adaptive_anchors"
    max_pairs: int = 250_000
    anchors_per_class: int = 20
    pair_feature_mode: str = "concat_diff"
    chunk_size: int = 8192
    random_state: int = 42

    def normalized(self) -> "PairwiseLearningConfig":
        if self.max_pairs <= 0:
            raise ValueError("max_pairs must be positive.")
        if self.anchors_per_class <= 0:
            raise ValueError("anchors_per_class must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.pair_feature_mode not in SUPPORTED_PAIR_FEATURE_MODES:
            raise ValueError(
                f"Unsupported pair_feature_mode={self.pair_feature_mode!r}. "
                f"Supported modes: {SUPPORTED_PAIR_FEATURE_MODES}."
            )
        if self.pairing_policy not in {"adaptive_anchors", "all_pairs", "exact", "stratified_anchors"}:
            raise ValueError(f"Unsupported pairing_policy={self.pairing_policy!r}.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairwiseBatch:
    features: np.ndarray
    target: np.ndarray
    left_indices: np.ndarray
    anchor_indices: np.ndarray
    diagnostics: dict[str, Any] # TODO: TypedContract добавить 


def normalize_feature_matrix(features: Any) -> np.ndarray:
    """Normalize FEDOT/sklearn time-series inputs to a finite 2D feature matrix."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_target_vector(target: Any) -> np.ndarray:
    return np.asarray(target).reshape(-1)


def resolve_pairwise_backend(backend: str = "auto") -> tuple[str, Any | None]:
    normalized = str(backend or "auto").lower()
    if normalized in {"numpy", "np"}:
        return "numpy", None
    if normalized in {"auto", "torch", "cpu", "cuda"}:
        if torch is None:
            if normalized == "cuda":
                raise RuntimeError("PDL backend='cuda' was requested, but torch is unavailable.")
            return "numpy", None
        if normalized == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PDL backend='cuda' was requested, but CUDA is unavailable.")
            return "torch_cuda", torch.device("cuda")
        if normalized == "auto" and torch.cuda.is_available():
            return "torch_cuda", torch.device("cuda")
        return "torch_cpu", torch.device("cpu")
    raise ValueError(f"Unsupported PDL backend={backend!r}.")

# TODO нет смысла во втором if так как в третьем делается то же самое и можно убрать первую часть второго if
# TODO а если в target не будет какого-то класса, то мб сломается 
def select_classification_anchor_indices(
        encoded_target: Any,
        config: PairwiseLearningConfig,
) -> np.ndarray:
    config = config.normalized()
    target = normalize_target_vector(encoded_target).astype(int)
    n_samples = len(target)
    if n_samples == 0:
        raise ValueError("Cannot select PDL anchors for an empty target.")
    if config.pairing_policy in {"all_pairs", "exact"} and n_samples * n_samples <= config.max_pairs:
        return np.arange(n_samples, dtype=int)
    if n_samples * n_samples <= config.max_pairs:
        return np.arange(n_samples, dtype=int)

    selected: list[int] = []
    for label in np.unique(target):
        class_indices = np.flatnonzero(target == label)
        selected.extend(_evenly_spaced_indices(class_indices, config.anchors_per_class))
    return np.asarray(sorted(set(selected)), dtype=int)


def select_regression_anchor_indices( # TODO: реализовать механизмы выбора якорей
        target: Any,
        config: PairwiseLearningConfig,
) -> np.ndarray:
    config = config.normalized()
    target_vector = normalize_target_vector(target)
    n_samples = len(target_vector)
    if n_samples == 0:
        raise ValueError("Cannot select PDL anchors for an empty target.")
    if config.pairing_policy in {"all_pairs", "exact"} and n_samples * n_samples <= config.max_pairs:
        return np.arange(n_samples, dtype=int)
    if n_samples * n_samples <= config.max_pairs:
        return np.arange(n_samples, dtype=int)

    max_anchor_count = max(1, min(n_samples, config.max_pairs // max(1, n_samples)))
    return _evenly_spaced_indices(np.arange(n_samples, dtype=int), max_anchor_count)


def build_classification_pairs(
        features: Any,
        encoded_target: Any,
        anchor_indices: Iterable[int],
        config: PairwiseLearningConfig,
) -> PairwiseBatch:
    feature_matrix = normalize_feature_matrix(features)
    target = normalize_target_vector(encoded_target).astype(int)
    anchors = np.asarray(tuple(anchor_indices), dtype=int)
    pair_features = build_pair_features(feature_matrix, feature_matrix[anchors], config)
    left_indices, repeated_anchor_indices = _pair_indices(len(feature_matrix), anchors)
    # pair_target = (target[left_indices] != target[repeated_anchor_indices]).astype(int)
    dissimilarity_target = (target[left_indices] != target[repeated_anchor_indices]).astype(int)
    diagnostics = _pair_diagnostics(
        config=config,
        n_left=len(feature_matrix),
        n_anchors=len(anchors),
        pair_feature_dim=pair_features.shape[1],
        anchor_indices=anchors,
        task="classification",
    )
    return PairwiseBatch(pair_features, dissimilarity_target, left_indices, repeated_anchor_indices, diagnostics)


def build_regression_pairs(
        features: Any,
        target: Any,
        anchor_indices: Iterable[int],
        config: PairwiseLearningConfig,
) -> PairwiseBatch:
    feature_matrix = normalize_feature_matrix(features)
    target_vector = normalize_target_vector(target).astype(float)
    anchors = np.asarray(tuple(anchor_indices), dtype=int)
    pair_features = build_pair_features(feature_matrix, feature_matrix[anchors], config)
    left_indices, repeated_anchor_indices = _pair_indices(len(feature_matrix), anchors)
    delta_left_minus_anchor = target_vector[left_indices] - target_vector[repeated_anchor_indices]
    diagnostics = _pair_diagnostics(
        config=config,
        n_left=len(feature_matrix),
        n_anchors=len(anchors),
        pair_feature_dim=pair_features.shape[1],
        anchor_indices=anchors,
        task="regression",
    )
    return PairwiseBatch(pair_features, delta_left_minus_anchor.astype(float), left_indices, repeated_anchor_indices, diagnostics)


def build_pair_features(
        left_features: Any,
        anchor_features: Any,
        config: PairwiseLearningConfig,
) -> np.ndarray:
    config = config.normalized()
    left = normalize_feature_matrix(left_features)
    anchors = normalize_feature_matrix(anchor_features)
    if left.shape[1] != anchors.shape[1]:
        raise ValueError(f"Pair matrices must have equal feature counts: {left.shape[1]} != {anchors.shape[1]}.")
    backend_name, device = resolve_pairwise_backend(config.backend)
    if backend_name == "numpy":
        return _build_pair_features_numpy(left, anchors, config.pair_feature_mode)
    return _build_pair_features_torch(left, anchors, config.pair_feature_mode, device)


def predict_similarity_by_chunks(
        base_model: Any,
        features: Any,
        anchor_features: Any,
        config: PairwiseLearningConfig,
) -> np.ndarray:
    feature_matrix = normalize_feature_matrix(features)
    anchors = normalize_feature_matrix(anchor_features)
    n_samples = len(feature_matrix)
    n_anchors = len(anchors)
    if n_anchors == 0:
        raise ValueError("PDL prediction requires at least one anchor.")
    rows_per_chunk = max(1, int(config.chunk_size) // max(1, n_anchors))
    chunks = []
    for start in range(0, n_samples, rows_per_chunk):
        chunk = feature_matrix[start:start + rows_per_chunk]
        pair_features = build_pair_features(chunk, anchors, config)
        same_probability = _predict_same_probability(base_model, pair_features)
        chunks.append(same_probability.reshape(len(chunk), n_anchors))
    return np.vstack(chunks) if chunks else np.empty((0, n_anchors), dtype=float)


def predict_regression_by_chunks(
        base_model: Any,
        features: Any,
        anchor_features: Any,
        anchor_target: Any,
        config: PairwiseLearningConfig,
) -> np.ndarray:
    feature_matrix = normalize_feature_matrix(features)
    anchors = normalize_feature_matrix(anchor_features)
    anchor_target = normalize_target_vector(anchor_target).astype(float)
    n_samples = len(feature_matrix)
    n_anchors = len(anchors)
    rows_per_chunk = max(1, int(config.chunk_size) // max(1, n_anchors))
    predictions = []
    for start in range(0, n_samples, rows_per_chunk):
        chunk = feature_matrix[start:start + rows_per_chunk]
        pair_features = build_pair_features(chunk, anchors, config)
        deltas = np.asarray(base_model.predict(pair_features), dtype=float).reshape(len(chunk), n_anchors)
        predictions.append(anchor_target.reshape(1, -1) + deltas)
    return np.concatenate([chunk.mean(axis=1) for chunk in predictions]) if predictions else np.empty(0, dtype=float)


def aggregate_similarity_to_class_proba(
        similarity: Any,
        anchor_labels: Any,
        n_classes: int,
) -> np.ndarray:
    similarity_matrix = np.asarray(similarity, dtype=float)
    anchor_labels = normalize_target_vector(anchor_labels).astype(int)
    probabilities = np.zeros((similarity_matrix.shape[0], n_classes), dtype=float)
    for class_id in range(n_classes):
        mask = anchor_labels == class_id
        if np.any(mask):
            probabilities[:, class_id] = np.mean(similarity_matrix[:, mask], axis=1)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    empty_rows = row_sums.squeeze(axis=1) <= np.finfo(float).eps
    if np.any(empty_rows):
        probabilities[empty_rows, :] = 1.0 / max(1, n_classes)
        row_sums = probabilities.sum(axis=1, keepdims=True)
    return probabilities / row_sums


def pair_feature_dim(n_features: int, mode: str) -> int:
    if mode in {"concat_diff", "concat_absdiff"}:
        return n_features * 3
    if mode == "diff_only":
        return n_features
    raise ValueError(f"Unsupported pair feature mode={mode!r}.")


def _evenly_spaced_indices(indices: np.ndarray, max_count: int) -> np.ndarray:
    if len(indices) <= max_count:
        return indices.astype(int)
    positions = np.linspace(0, len(indices) - 1, num=max_count, dtype=int)
    return indices[positions].astype(int)


def _pair_indices(n_left: int, anchor_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_indices = np.repeat(np.arange(n_left, dtype=int), len(anchor_indices))
    repeated_anchor_indices = np.tile(anchor_indices.astype(int), n_left)
    return left_indices, repeated_anchor_indices


def _build_pair_features_numpy(left: np.ndarray, anchors: np.ndarray, mode: str) -> np.ndarray:
    left_repeated = np.repeat(left, len(anchors), axis=0)
    anchors_tiled = np.tile(anchors, (len(left), 1))
    difference = left_repeated - anchors_tiled
    return _combine_pair_blocks(left_repeated, anchors_tiled, difference, mode)


def _build_pair_features_torch(left: np.ndarray, anchors: np.ndarray, mode: str, device: Any) -> np.ndarray:
    left_tensor = torch.as_tensor(left, dtype=torch.float32, device=device)
    anchor_tensor = torch.as_tensor(anchors, dtype=torch.float32, device=device)
    left_repeated = left_tensor.repeat_interleave(anchor_tensor.shape[0], dim=0)
    anchors_tiled = anchor_tensor.repeat(left_tensor.shape[0], 1)
    difference = left_repeated - anchors_tiled # 
    if mode == "concat_diff":
        paired = torch.cat((left_repeated, anchors_tiled, difference), dim=1)
    elif mode == "concat_absdiff":
        paired = torch.cat((left_repeated, anchors_tiled, torch.abs(difference)), dim=1)
    elif mode == "diff_only":
        paired = difference
    else:  # pragma: no cover - normalized config prevents this branch.
        raise ValueError(f"Unsupported pair feature mode={mode!r}.")
    return paired.detach().cpu().numpy()


def _pair_target_semantics(*, task: str) -> dict[str, Any]:
    # TODO: заменить через монады?
    if task == "classification":
        return {
            "task": "classification",
            "same_label": CLASSIFICATION_SAME_LABEL,
            "different_label": CLASSIFICATION_DIFFERENT_LABEL,
            "target_type": "dissimilarity",
            "target_formula": "int(left_class != anchor_class)",
            "inference_output": "same_probability",  # P(same_label)
        }
    if task == "regression":
        return {
            "task": "regression",
            "delta_sign": REGRESSION_DELTA_SIGN,
            "target_formula": "target_left - target_anchor",
            "inference_reconstruction": "anchor_target + predicted_delta",
        }
    raise ValueError(f"Unsupported PDL task={task!r}.")


def _combine_pair_blocks(left: np.ndarray, anchors: np.ndarray, difference: np.ndarray, mode: str) -> np.ndarray:
    if mode == "concat_diff":
        return np.hstack((left, anchors, difference)).astype(np.float32, copy=False)
    if mode == "concat_absdiff":
        return np.hstack((left, anchors, np.abs(difference))).astype(np.float32, copy=False)
    if mode == "diff_only":
        return difference.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported pair feature mode={mode!r}.")


def _predict_same_probability(base_model: Any, pair_features: np.ndarray) -> np.ndarray:
    if hasattr(base_model, "predict_proba"):
        probability = np.asarray(base_model.predict_proba(pair_features), dtype=float)
        classes = np.asarray(getattr(base_model, "classes_", np.arange(probability.shape[1])))
        same_columns = np.flatnonzero(classes == CLASSIFICATION_SAME_LABEL)
        if len(same_columns) == 0:
            return np.zeros(pair_features.shape[0], dtype=float)
        return probability[:, int(same_columns[0])]
    prediction = np.asarray(base_model.predict(pair_features)).reshape(-1)
    return (prediction == CLASSIFICATION_SAME_LABEL).astype(float)


def _pair_diagnostics(
        *,
        config: PairwiseLearningConfig,
        n_left: int,
        n_anchors: int,
        pair_feature_dim: int,
        anchor_indices: np.ndarray,
        task: str
) -> dict[str, Any]: # TODO: добавть контракт на все использование diagnostic
    backend_name, _ = resolve_pairwise_backend(config.backend)
    return {
        "backend": backend_name,
        "pairing_policy": config.pairing_policy,
        "n_train": int(n_left),
        "n_anchors": int(n_anchors),
        "n_pairs": int(n_left * n_anchors),
        "pair_feature_dim": int(pair_feature_dim),
        "anchor_indices": [int(index) for index in anchor_indices],
        "config": config.to_dict(),
        "pair_target_semantics": _pair_target_semantics(task=task),
    }
