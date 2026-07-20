"""Estimator facades for Pairwise Difference Learning (PDL).

This module wires the task-agnostic core in ``pairwise_core`` into
scikit-learn / FEDOT compatible classifier and regressor classes, plus a thin
legacy helper (``PairwiseDifferenceEstimator``) kept for backward compatibility.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.preprocessing import LabelEncoder

from .pairwise_core import (
    PairwiseLearningConfig,
    build_pair_batch,
    build_pair_features,
    normalize_feature_matrix,
    normalize_target_vector,
    predict_regression_by_chunks,
    predict_similarity_by_chunks,
    resolve_pdl_strategies,
)
from fedot_ind.core.repository.constanst_repository import SKLEARN_CLF_IMP, SKLEARN_REG_IMP

__all__ = [
    "PairwiseDifferenceClassifier",
    "PairwiseDifferenceEstimator",
    "PairwiseDifferenceRegressor",
    "PairwiseLearningConfig",
]


class PairwiseDifferenceEstimator:
    """Compatibility helper exposing the legacy pair construction methods without pandas cross-merge."""

    def __init__(self, config: PairwiseLearningConfig | None = None):
        """Initialize the helper with a normalized configuration.

        Args:
            config: Optional pairwise learning configuration; defaults are used
                when omitted.
        """
        self.config = (config or PairwiseLearningConfig()).normalized()

    def _convert_to_pandas(
        self, arr1: Any, arr2: Any
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Coerce two array-like inputs to ``pandas.DataFrame`` objects."""
        if not isinstance(arr1, pd.DataFrame):
            arr1 = pd.DataFrame(arr1)
        if not isinstance(arr2, pd.DataFrame):
            arr2 = pd.DataFrame(arr2)
        return arr1, arr2

    def pair_input(self, X1: Any, X2: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build pair-feature frames and their symmetric counterparts.

        Args:
            X1: Left samples.
            X2: Anchor samples.

        Returns:
            A tuple ``(pair_frame, symmetric_frame)`` of pair-feature
            DataFrames.
        """
        x1_frame, x2_frame = self._convert_to_pandas(X1, X2)
        x1 = normalize_feature_matrix(x1_frame.values)
        x2 = normalize_feature_matrix(x2_frame.values)
        left_repeated = np.repeat(x1, len(x2), axis=0)
        right_tiled = np.tile(x2, (len(x1), 1))
        difference = left_repeated - right_tiled
        pair_frame = self._pair_frame(
            left_repeated, right_tiled, difference, x1_frame.columns
        )

        sym_left = right_tiled
        sym_right = left_repeated
        sym_difference = sym_left - sym_right
        sym_frame = self._pair_frame(
            sym_left, sym_right, sym_difference, x1_frame.columns
        )
        return pair_frame, sym_frame

    def pair_output(self, y1: Any, y2: Any) -> np.ndarray:
        """Build regression pair targets using the left-minus-anchor sign.

        Args:
            y1: Left target values.
            y2: Anchor target values.

        Returns:
            Per-pair deltas (``target_left - target_anchor``) in cross-product
            order.
        """
        left = normalize_target_vector(y1).astype(float)
        anchors = normalize_target_vector(y2).astype(float)
        delta_left_minus_anchor = (
            np.repeat(left, len(anchors)) - np.tile(anchors, len(left))
        ).astype(float)
        return delta_left_minus_anchor

    def pair_output_difference(
        self, y1: Any, y2: Any, nb_classes: int | None = None
    ) -> np.ndarray:
        """Build classification pair targets (dissimilarity) for the legacy API.

        The contract is aligned with the active PDL path: ``0`` means the same
        class (``CLASSIFICATION_SAME_LABEL``) and ``1`` means different classes
        (``CLASSIFICATION_DIFFERENT_LABEL``).

        Args:
            y1: Left class labels.
            y2: Anchor class labels.
            nb_classes: Unused; kept for backward-compatible signatures.

        Returns:
            One dissimilarity value per ``(left, anchor)`` pair in cross-product
            order.
        """
        del nb_classes
        left = normalize_target_vector(y1)
        anchors = normalize_target_vector(y2)
        dissimilarity_target = (
            np.repeat(left, len(anchors)) != np.tile(anchors, len(left))
        ).astype(int)
        return dissimilarity_target

    @staticmethod
    def _pair_frame(
        left: np.ndarray, right: np.ndarray, difference: np.ndarray, columns: Any
    ) -> pd.DataFrame:
        """Assemble a labelled pair DataFrame from left/right/difference blocks."""
        column_names = [str(column) for column in columns]
        payload = np.hstack((left, right, difference))
        return pd.DataFrame(
            payload,
            columns=(
                [f"{column}_x" for column in column_names]
                + [f"{column}_y" for column in column_names]
                + [f"{column}_diff" for column in column_names]
            ),
        )

    @staticmethod
    def predict(
        y_prob: np.ndarray, output_mode: str = "default", min_label_zero: bool = True
    ) -> np.ndarray:
        """Convert probabilities to labels or pass them through unchanged.

        Args:
            y_prob: Probability matrix of shape ``(n_samples, n_classes)``.
            output_mode: When it contains ``"label"`` return argmax labels,
                otherwise return ``y_prob`` unchanged.
            min_label_zero: If ``False`` shift labels so the minimum label is
                one.

        Returns:
            Either class labels or the original probability matrix.
        """
        if "label" in output_mode:
            predicted_classes = np.argmax(y_prob, axis=1).reshape(-1, 1)
            return predicted_classes if min_label_zero else predicted_classes + 1
        return y_prob


class PairwiseDifferenceClassifier:
    """Pairwise Difference Learning classifier with deterministic anchors and torch/numpy pair generation."""

    def __init__(self, params: Optional[OperationParameters] = None):
        """Initialize the classifier and its base pair model.

        Args:
            params: Optional FEDOT operation parameters. Configuration keys are
                routed into ``PairwiseLearningConfig`` and the remaining keys are
                forwarded to the base estimator.
        """
        raw_params = _operation_params_to_dict(params)
        self.config = _extract_pairwise_config(raw_params)
        self.strategies_ = resolve_pdl_strategies(
            self.config, task="classification")
        self.model_name = raw_params.pop("model", "rf")
        self.base_model_params = dict(raw_params)
        self.base_model = SKLEARN_CLF_IMP[self.model_name](
            **self.base_model_params)
        self.pde = PairwiseDifferenceEstimator(self.config)
        self.sample_weight_ = None
        self.diagnostics_: dict[str, Any] = {}

    def fit(self, input_data: InputData | np.ndarray, target: Any | None = None):
        """Fit the base pair classifier on anchor-relative classification pairs.

        Args:
            input_data: FEDOT ``InputData`` or a raw feature matrix.
            target: Class labels when ``input_data`` is a raw matrix.

        Returns:
            The fitted estimator (``self``).

        Raises:
            ValueError: If labels are missing or fewer than two classes exist.
        """
        features, raw_target, _ = _extract_features_target(input_data, target)
        if raw_target is None:
            raise ValueError(
                "PairwiseDifferenceClassifier.fit expects target labels.")

        self.train_features_ = normalize_feature_matrix(features)
        self.train_features = self.train_features_
        self.target = normalize_target_vector(raw_target)
        self.label_encoder_ = LabelEncoder()
        self.target_encoded_ = self.label_encoder_.fit_transform(self.target)
        self.classes_ = self.label_encoder_.classes_
        self.num_classes = len(self.classes_)
        if self.num_classes < 2:
            raise ValueError(
                "PairwiseDifferenceClassifier requires at least two classes."
            )

        self.anchor_indices_ = self.strategies_.anchor_selector.select(
            self.train_features_, self.target_encoded_
        )
        self.anchor_features_ = self.train_features_[self.anchor_indices_]
        self.anchor_labels_ = self.target_encoded_[self.anchor_indices_]

        batch = build_pair_batch(
            self.train_features_,
            self.target_encoded_,
            self.anchor_indices_,
            self.config,
            task="classification",
            strategies=self.strategies_,
        )
        self.base_model.fit(batch.features, batch.target)
        self.diagnostics_ = {
            **batch.diagnostics,
            "task": "classification",
            "n_classes": int(self.num_classes),
            "class_labels": [str(label) for label in self.classes_],
            "base_model": self.model_name,
            "base_model_params": deepcopy(self.base_model_params),
        }
        return self

    def predict(
        self, input_data: InputData | np.ndarray, output_mode: str = "labels"
    ) -> np.ndarray:
        """Predict class labels or probabilities.

        Args:
            input_data: FEDOT ``InputData`` or a raw feature matrix.
            output_mode: When it contains ``"label"`` return decoded labels,
                otherwise return the probability matrix.

        Returns:
            Decoded labels of shape ``(n_samples, 1)`` or a probability matrix.
        """
        probabilities = self._predict_encoded_proba(
            _extract_features(input_data))
        if "label" not in output_mode:
            return probabilities
        encoded_labels = np.argmax(probabilities, axis=1)
        labels = self.label_encoder_.inverse_transform(encoded_labels)
        return labels.reshape(-1, 1)

    def predict_proba(
        self, input_data: InputData | np.ndarray, output_mode: str = "default"
    ) -> np.ndarray:
        """Predict class probabilities (or labels when requested).

        Args:
            input_data: FEDOT ``InputData`` or a raw feature matrix.
            output_mode: When it contains ``"label"`` delegate to ``predict``.

        Returns:
            A probability matrix of shape ``(n_samples, n_classes)``.
        """
        probabilities = self._predict_encoded_proba(
            _extract_features(input_data))
        if "label" in output_mode:
            return self.predict(input_data, output_mode=output_mode)
        return probabilities

    def predict_for_fit(
        self, input_data: InputData | np.ndarray, output_mode: str = "default"
    ) -> np.ndarray:
        """Predict probabilities (or labels) for the FEDOT fit pipeline."""
        if "label" in output_mode:
            return self.predict(input_data, output_mode=output_mode)
        return self.predict_proba(input_data)

    def score_difference(
        self, input_data: InputData | np.ndarray, target: Any | None = None
    ) -> float:
        """Return the mean absolute error between target and predicted dissimilarity.

        Args:
            input_data: FEDOT ``InputData`` or a raw feature matrix.
            target: Class labels when ``input_data`` is a raw matrix.

        Returns:
            Mean absolute dissimilarity error, where ``1.0 = different`` and
            ``0.0 = same``.

        Raises:
            ValueError: If labels are missing.
        """
        features, raw_target, _ = _extract_features_target(input_data, target)
        if raw_target is None:
            raise ValueError("score_difference expects target labels.")
        encoded_target = self.label_encoder_.transform(
            normalize_target_vector(raw_target)
        )
        same_probability = predict_similarity_by_chunks(
            self.base_model, features, self.anchor_features_, self.config
        )
        # Dissimilarity target: 1.0 = different, 0.0 = same.
        dissimilarity_target = (
            np.repeat(encoded_target, len(self.anchor_labels_))
            != np.tile(self.anchor_labels_, len(encoded_target))
        ).astype(float)
        predicted_dissimilarity = 1.0 - same_probability.reshape(-1)
        return float(np.mean(np.abs(dissimilarity_target - predicted_dissimilarity)))

    def get_diagnostics(self) -> dict[str, Any]:
        """Return a copy of the diagnostics collected during ``fit``."""
        return dict(self.diagnostics_)

    def _predict_encoded_proba(self, features: Any) -> np.ndarray:
        """Predict encoded class probabilities for already-extracted features."""
        _check_is_fitted(
            self, ("anchor_features_", "anchor_labels_", "label_encoder_"))
        similarity = predict_similarity_by_chunks(
            self.base_model, features, self.anchor_features_, self.config
        )
        assert self.strategies_.pair_aggregator is not None
        return self.strategies_.pair_aggregator.aggregate(
            similarity, self.anchor_labels_, n_classes=self.num_classes
        )


class PairwiseDifferenceRegressor:
    """Pairwise Difference Learning regressor with deterministic anchor aggregation."""

    def __init__(self, params: Optional[OperationParameters] = None):
        """Initialize the regressor and its base pair model.

        Args:
            params: Optional FEDOT operation parameters. Configuration keys are
                routed into ``PairwiseLearningConfig`` and the remaining keys are
                forwarded to the base estimator.
        """
        raw_params = _operation_params_to_dict(params)
        self.config = _extract_pairwise_config(raw_params)
        self.strategies_ = resolve_pdl_strategies(
            self.config, task="regression")
        self.model_name = raw_params.pop("model", "treg")
        self.base_model_params = dict(raw_params)
        self.base_model = SKLEARN_REG_IMP[self.model_name](
            **self.base_model_params)
        self.pde = PairwiseDifferenceEstimator(self.config)
        self.sample_weight_ = None
        self.diagnostics_: dict[str, Any] = {}

    def fit(self, input_data: InputData | np.ndarray, target: Any | None = None):
        """Fit the base pair regressor on anchor-relative delta pairs.

        Args:
            input_data: FEDOT ``InputData`` or a raw feature matrix.
            target: Target values when ``input_data`` is a raw matrix.

        Returns:
            The fitted estimator (``self``).

        Raises:
            ValueError: If target values are missing.
        """
        features, raw_target, _ = _extract_features_target(input_data, target)
        if raw_target is None:
            raise ValueError(
                "PairwiseDifferenceRegressor.fit expects target values.")

        self.train_features_ = normalize_feature_matrix(features)
        self.train_features = self.train_features_
        self.target_ = normalize_target_vector(raw_target).astype(float)
        self.target = self.target_
        self.num_classes = getattr(input_data, "num_classes", None)
        self.y_train_ = pd.Series(self.target_)
        self.anchor_indices_ = self.strategies_.anchor_selector.select(
            self.train_features_, self.target_
        )
        self.anchor_features_ = self.train_features_[self.anchor_indices_]
        self.anchor_target_ = self.target_[self.anchor_indices_]

        batch = build_pair_batch(
            self.train_features_,
            self.target_,
            self.anchor_indices_,
            self.config,
            task="regression",
            strategies=self.strategies_,
        )
        self.base_model.fit(batch.features, batch.target)
        self.diagnostics_ = {
            **batch.diagnostics,
            "task": "regression",
            "base_model": self.model_name,
            "base_model_params": deepcopy(self.base_model_params),
        }
        return self

    def predict(
        self, input_data: InputData | np.ndarray, output_mode: str = "default"
    ) -> np.ndarray:
        """Predict regression targets via anchor-relative delta aggregation.

        Args:
            input_data: FEDOT ``InputData`` or a raw feature matrix.
            output_mode: Unused; kept for interface compatibility.

        Returns:
            Predicted target vector of shape ``(n_samples,)``.
        """
        del output_mode
        _check_is_fitted(self, ("anchor_features_", "anchor_target_"))
        return predict_regression_by_chunks(
            self.base_model,
            _extract_features(input_data),
            self.anchor_features_,
            self.anchor_target_,
            self.config,
        ).reshape(-1)

    def predict_proba(self, input_data: InputData | np.ndarray) -> np.ndarray:
        """Return point predictions; provided for interface compatibility."""
        return self.predict(input_data)

    def predict_for_fit(
        self, input_data: InputData | np.ndarray, output_mode: str = "default"
    ) -> np.ndarray:
        """Return point predictions for the FEDOT fit pipeline."""
        del output_mode
        return self.predict(input_data)

    def get_diagnostics(self) -> dict[str, Any]:
        """Return a copy of the diagnostics collected during ``fit``."""
        return dict(self.diagnostics_)

    def _predict_samples(
        self, input_data: InputData | np.ndarray, force_symmetry: bool = False
    ):
        """Return per-anchor reconstructed samples and their raw deltas.

        Args:
            input_data: FEDOT ``InputData`` or a raw feature matrix.
            force_symmetry: Unused; reserved for symmetric inference.

        Returns:
            A tuple ``(samples, deltas)`` of DataFrames shaped
            ``(n_samples, n_anchors)``.
        """
        del force_symmetry
        _check_is_fitted(self, ("anchor_features_", "anchor_target_"))
        features = normalize_feature_matrix(_extract_features(input_data))
        pair_features = build_pair_features(
            features, self.anchor_features_, self.config
        )
        deltas = np.asarray(
            self.base_model.predict(pair_features), dtype=float
        ).reshape(
            len(features),
            len(self.anchor_target_),
        )
        samples = self.anchor_target_.reshape(1, -1) + deltas
        return pd.DataFrame(samples), pd.DataFrame(deltas)

    def set_sample_weight(self, sample_weight: Any):
        """Validate and store normalized sample weights.

        Args:
            sample_weight: Either a ``pandas.Series`` matching the train size or
                an array matching the anchor count.

        Returns:
            The estimator (``self``).

        Raises:
            ValueError: If the size mismatches or non-finite values are present.
        """
        if isinstance(sample_weight, pd.Series):
            if len(sample_weight) != len(self.y_train_):
                raise ValueError(
                    f"sample_weight size {len(sample_weight)} should be equal to the train size {len(self.y_train_)}"
                )
            self.sample_weight_ = sample_weight
            return self
        weights = normalize_target_vector(sample_weight).astype(float)
        if len(weights) != len(self.anchor_indices_):
            raise ValueError(
                f"sample_weight size {len(weights)} should be equal to the anchor size {len(self.anchor_indices_)}"
            )
        if np.any(~np.isfinite(weights)):
            raise ValueError("sample_weight contains non-finite values.")
        if np.all(weights <= 0):
            weights = np.ones_like(weights, dtype=float)
        self.sample_weight_ = weights / np.sum(weights)
        return self


def _extract_features_target(
    input_data: InputData | np.ndarray, target: Any | None = None
) -> tuple[Any, Any, Any]:
    """Return ``(features, target, task)`` from ``InputData`` or raw inputs."""
    if isinstance(input_data, InputData):
        return input_data.features, input_data.target, input_data.task
    return input_data, target, None


def _extract_features(input_data: InputData | np.ndarray) -> Any:
    """Return the feature matrix from ``InputData`` or pass the input through."""
    if isinstance(input_data, InputData):
        return input_data.features
    return input_data


def _operation_params_to_dict(params: Optional[OperationParameters]) -> dict[str, Any]:
    """Coerce FEDOT operation parameters into a plain dictionary."""
    if params is None:
        return {}
    if hasattr(params, "_parameters"):
        return dict(params._parameters)
    if isinstance(params, dict):
        return dict(params)
    return dict(params)


def _extract_pairwise_config(params: dict[str, Any]) -> PairwiseLearningConfig:
    """Pop ``PairwiseLearningConfig`` fields out of ``params`` and build a config."""
    config_fields = PairwiseLearningConfig.__dataclass_fields__.keys()
    config_payload = {
        field_name: params.pop(field_name)
        for field_name in tuple(config_fields)
        if field_name in params
    }
    return PairwiseLearningConfig(**config_payload).normalized()


def _check_is_fitted(model: Any, attributes: tuple[str, ...]) -> None:
    """Raise if any of the required fitted attributes are missing."""
    missing = [attribute for attribute in attributes if not hasattr(
        model, attribute)]
    if missing:
        raise ValueError(
            f"{model.__class__.__name__} is not fitted yet. Missing attributes: {missing}."
        )
