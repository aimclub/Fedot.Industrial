import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.linalg import vector_norm, matrix_norm


class OrthogonalLoss(Module):
    """Calculates orthogonality loss for complex unitary matrices."""

    def __init__(self, device: torch.device) -> None:
        super(OrthogonalLoss, self).__init__()
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

            elif name.split(".")[-1] == "U" or name.split(".")[-1] == "Vh":
                Vh = parameter
                r = Vh.size()[0]
                E = torch.eye(r, device=self.device)
                loss += matrix_norm(Vh @ Vh.transpose(0, 1) - E) ** 2 / r
        return loss / n


class HoyerLoss(Module):
    """Calculates Hoyer Loss for diagonal matrix with singular values."""

    def __init__(self) -> None:
        super(HoyerLoss, self).__init__()

    def forward(self, model: Module) -> Tensor:
        loss = 0
        n = 0
        for name, parameter in model.named_parameters():
            if name.split(".")[-1] == "S":
                n += 1
                S = parameter
                loss += vector_norm(S, ord=1) / vector_norm(S, ord=2)
        return loss / n
