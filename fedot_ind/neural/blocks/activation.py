"""Activation modules without fastai or task-runtime imports."""
import torch
from torch import nn
from torch.nn import functional as F


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class ReGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.relu(gates)


class SwishBeta(nn.Module):
    def __multiinit__(self, beta=1.):
        self.sigmoid = torch.sigmoid
        self.beta = nn.Parameter(torch.Tensor(1).fill_(beta))

    def forward(self, x):
        return x.mul(self.sigmoid(x * self.beta))


class SmeLU(nn.Module):
    def __init__(self, beta: float = 2.) -> None:
        super().__init__()
        self.beta = abs(beta)

    def forward(self, x):
        return torch.where(torch.abs(x) <= self.beta, ((x + self.beta) ** 2) / (4. * self.beta), F.relu(x))


Mish = nn.Mish
_ACTIVATIONS = {
    "ELU": nn.ELU, "LeakyReLU": nn.LeakyReLU, "PReLU": nn.PReLU, "ReLU": nn.ReLU,
    "ReLU6": nn.ReLU6, "SELU": nn.SELU, "CELU": nn.CELU, "GELU": nn.GELU,
    "SwishBeta": SwishBeta, "Sigmoid": nn.Sigmoid, "Mish": Mish, "Softplus": nn.Softplus,
    "Tanh": nn.Tanh, "Softmax": nn.Softmax, "GEGLU": GEGLU, "ReGLU": ReGLU, "SmeLU": SmeLU,
}


def get_activation_fn(activation):
    return _ACTIVATIONS[activation]()


pytorch_acts = [nn.ELU, nn.LeakyReLU, nn.PReLU, nn.ReLU, nn.ReLU6, nn.SELU, nn.CELU,
                nn.GELU, nn.Sigmoid, Mish, nn.Softplus, nn.Tanh, nn.Softmax, GEGLU, ReGLU, SmeLU]
pytorch_act_names = [a.__name__.lower() for a in pytorch_acts]
