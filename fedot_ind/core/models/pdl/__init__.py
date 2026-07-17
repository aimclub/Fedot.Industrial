from __future__ import annotations

from typing import TYPE_CHECKING

from .pairwise_core import PairwiseLearningConfig

if TYPE_CHECKING:
    from .pairwise_model import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor

__all__ = [
    "PairwiseDifferenceClassifier",
    "PairwiseDifferenceRegressor",
    "PairwiseLearningConfig",
]


def __getattr__(name: str):
    if name in {"PairwiseDifferenceClassifier", "PairwiseDifferenceRegressor"}:
        from .pairwise_model import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor

        return {
            "PairwiseDifferenceClassifier": PairwiseDifferenceClassifier,
            "PairwiseDifferenceRegressor": PairwiseDifferenceRegressor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
