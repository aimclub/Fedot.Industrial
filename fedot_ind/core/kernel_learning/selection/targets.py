from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TargetKernelBuilder:
    task_type: str
    gamma: str | float = "scale"

    def build(self, y: Any) -> np.ndarray:
        if self.task_type == "classification":
            labels = np.asarray(y).reshape(-1)
            return (labels[:, None] == labels[None, :]).astype(float)
        if self.task_type == "regression":
            values = np.asarray(y, dtype=float).reshape(-1, 1)
            gamma = self._resolve_gamma(values)
            diff = values - values.T
            return np.exp(-gamma * diff ** 2)
        raise ValueError(f"Unsupported target kernel task_type: {self.task_type}")

    def _resolve_gamma(self, values: np.ndarray) -> float:
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        variance = float(np.var(values))
        if str(self.gamma).lower() == "scale":
            return 1.0 / variance if variance > 1e-12 else 1.0
        if str(self.gamma).lower() == "auto":
            return 1.0
        raise ValueError(f"Unsupported target gamma: {self.gamma}")
