from abc import ABC
from abc import abstractmethod

import torch


class AbstractNormalizer(ABC):
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.state_: dict[str, object] = {}
        self.is_fitted_ = False

    def fit(self, X: torch.Tensor) -> None:
        self._set_state({})

    @abstractmethod
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

    def get_state(self) -> dict[str, object]:
        return dict(self.state_)

    def _set_state(self, state: dict[str, object]) -> None:
        self.state_ = state
        self.is_fitted_ = True

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError(f"{self.__class__.__name__} must be fitted before transform.")


class ImputationNormalizer(AbstractNormalizer):
    def fit(self, X: torch.Tensor) -> None:
        values = X.float()
        if values.ndim < 2:
            raise ValueError(
                "Imputation expects at least a 2D tensor (batch, features), "
                f"got shape={tuple(values.shape)}."
            )
        finite = torch.isfinite(values)
        masked = torch.where(finite, values, torch.zeros_like(values))
        counts = finite.sum(dim=0, keepdim=True).clamp_min(1)
        column_mean = masked.sum(dim=0, keepdim=True) / counts
        self._set_state({
            "column_mean": column_mean.detach().clone(),
        })

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        values = X.float()
        column_mean = self.state_["column_mean"].to(device=X.device, dtype=X.dtype)
        invalid = ~torch.isfinite(values)
        return torch.where(invalid, column_mean, values)


class FeatureStandardizationNormalizer(AbstractNormalizer):
    def fit(self, X: torch.Tensor) -> None:
        values = X.float()
        if values.ndim < 2:
            raise ValueError(
                "Feature standardization expects at least a 2D tensor (batch, features), "
                f"got shape={tuple(values.shape)}."
            )
        mean = values.mean(dim=0, keepdim=True)
        std = values.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        self._set_state({
            "mean": mean.detach().clone(),
            "std": std.detach().clone(),
            "eps": float(self.eps),
        })

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        mean = self.state_["mean"].to(device=X.device, dtype=X.dtype)
        std = self.state_["std"].to(device=X.device, dtype=X.dtype)
        return torch.nan_to_num(((X.float() - mean) / std))


class ImageStandardizationNormalizer(AbstractNormalizer):
    def fit(self, X: torch.Tensor) -> None:
        if X.ndim not in (3, 4):
            raise ValueError(
                "Image standardization expects a 3D or 4D tensor, "
                f"got shape={tuple(X.shape)}."
            )
        dims = (0, 2, 3) if X.ndim == 4 else (0, 1, 2)
        mean = X.mean(dim=dims, keepdim=True)
        std = X.std(dim=dims, unbiased=False, keepdim=True).clamp_min(self.eps)
        self._set_state({
            "mean": mean.detach().clone(),
            "std": std.detach().clone(),
            "dims": dims,
            "eps": float(self.eps),
        })

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        mean = self.state_["mean"].to(device=X.device, dtype=X.dtype)
        std = self.state_["std"].to(device=X.device, dtype=X.dtype)
        return torch.nan_to_num(((X - mean) / std).float())


class Log1pNormalizer(AbstractNormalizer):
    def fit(self, X: torch.Tensor) -> None:
        self._set_state({})

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        return torch.log1p(X.float().clamp_min(0))
