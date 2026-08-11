"""Contiguous class-label encoding utilities."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


class LabelMappingError(ValueError):
    """Raised when labels or class indices are outside a fitted mapping."""


class ContiguousLabelEncoder:
    """Map arbitrary class labels to contiguous indices ``0..C-1``.

    Fitted state is stored only as ``classes_`` (label at position ``i`` has
    index ``i``). Forward/inverse dicts are derived on demand.
    """

    def __init__(self) -> None:
        self.classes_: tuple[Any, ...] | None = None

    @property
    def num_classes(self) -> int:
        return 0 if self.classes_ is None else len(self.classes_)

    @property
    def is_fitted(self) -> bool:
        return self.classes_ is not None

    @property
    def label_to_index_(self) -> dict[Any, int]:
        classes = self._require_classes()
        return {label: index for index, label in enumerate(classes)}

    @property
    def index_to_label_(self) -> dict[int, Any]:
        classes = self._require_classes()
        return {index: label for index, label in enumerate(classes)}

    def fit(self, labels: Iterable[Any]) -> ContiguousLabelEncoder:
        unique = tuple(sorted(set(labels)))
        if not unique:
            raise LabelMappingError('Cannot fit ContiguousLabelEncoder on an empty label set.')
        self.classes_ = unique
        return self

    def transform(self, labels: Sequence[Any]) -> list[int]:
        mapping = self.label_to_index_
        unknown = sorted({label for label in labels if label not in mapping})
        if unknown:
            raise LabelMappingError(f'Unknown target labels: {unknown}.')
        return [mapping[label] for label in labels]

    def fit_transform(self, labels: Sequence[Any]) -> list[int]:
        return self.fit(labels).transform(labels)

    def inverse_transform(self, class_indices: Sequence[Any]) -> list[Any]:
        classes = self._require_classes()
        known_indices = list(range(len(classes)))
        decoded: list[Any] = []
        for index in class_indices:
            class_index = int(index)
            if class_index < 0 or class_index >= len(classes):
                raise LabelMappingError(
                    f'Predicted class index {class_index} is out of vocabulary; '
                    f'expected contiguous indices {known_indices}.'
                )
            decoded.append(classes[class_index])
        return decoded

    def as_label_mapping(self) -> dict[Any, int]:
        return dict(self.label_to_index_)

    def _require_classes(self) -> tuple[Any, ...]:
        if self.classes_ is None:
            raise LabelMappingError('ContiguousLabelEncoder must be fitted first.')
        return self.classes_
