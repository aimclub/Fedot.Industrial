import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.linalg import vector_norm, matrix_norm


class SVDLoss(Module):
    """Base class for singular value decomposition losses."""

    def __init__(self, factor: float = 1) -> None:
        super().__init__()
        self.factor = factor


class OrthogonalLoss(SVDLoss):
    """Calculates orthogonality loss for complex unitary matrices."""

    def __init__(self, device: torch.device, factor: float = 1) -> None:
        super().__init__(factor=factor)
        self.device = device

    def forward(self, model: Module) -> Tensor:
        loss = 0
        n = 0
        for name, parameter in model.named_parameters():
            if name.split(".")[-1] == "U":
                n += 1
                U = parameter
                r = U.size()[1]
                E = torch.eye(r, device=self.device)
                loss += matrix_norm(U.transpose(0, 1) @ U - E) ** 2 / r

            elif name.split(".")[-1] == "Vh":
                Vh = parameter
                r = Vh.size()[0]
                E = torch.eye(r, device=self.device)
                loss += matrix_norm(Vh @ Vh.transpose(0, 1) - E) ** 2 / r
        return self.factor * loss / n


class HoyerLoss(SVDLoss):
    """Calculates Hoyer Loss for diagonal matrix with singular values."""

    def __init__(self, factor: float = 1) -> None:
        super().__init__(factor=factor)

    def forward(self, model: Module) -> Tensor:
        loss = 0
        n = 0
        for name, parameter in model.named_parameters():
            if name.split(".")[-1] == "S":
                n += 1
                S = parameter
                loss += vector_norm(S, ord=1) / vector_norm(S, ord=2)
        return self.factor * loss / n
