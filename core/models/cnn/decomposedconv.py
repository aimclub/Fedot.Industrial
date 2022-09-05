import torch

from torch import Tensor
from torch.nn.modules.conv import Conv2d
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t


from typing import Union

__all__ = ['DecomposedConv2d']


class DecomposedConv2d(Conv2d):
    """Extends the Conv2d layer by implementing the singular value decomposition of
    the weight matrix.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type,
        decomposing: bool = True,
        decomposing_mode: str = "channel",
        device=None,
        dtype=None,
    ) -> None:

        super(DecomposedConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        self.decomposing = decomposing
        self.decomposing_mode = decomposing_mode

        valid_decomposing_modes = {"channel", "spatial"}
        if decomposing_mode not in valid_decomposing_modes:
            raise ValueError(
                "decomposing_mode must be one of {}, but got decomposing_mode='{}'".format(
                    valid_decomposing_modes, decomposing_mode
                )
            )

        if decomposing:
            self.decompose()
        else:
            self.U = None
            self.S = None
            self.Vh = None

    def decompose(self) -> None:
        """Decompose the weight matrix in singular value decomposition."""

        n, c, w, h = self.weight.size()
        if self.decomposing_mode == "channel":
            W = self.weight.view(n, c * w * h)
        else:
            W = self.weight.view(n * w, c * h)

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        self.U = Parameter(U)
        self.S = Parameter(S)
        self.Vh = Parameter(Vh)
        self.register_parameter("weight", None)
        self.decomposing = True

    def compose(self):
        """Compose the weight matrix from singular value decomposition."""

        W = self.U @ torch.diag(self.S) @ self.Vh
        self.weight = Parameter(
            W.view(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size
            )
        )

        self.register_parameter("U", None)
        self.register_parameter("S", None)
        self.register_parameter("Vh", None)
        self.decomposing = False

    def pruning(self, e: float) -> float:
        """Prune the weight matrix at the energy threshold.
        Returns the compression ratio.
        """

        assert self.decomposing, "for pruning, the model must be decomposed"

        len_S = self.S.numel()
        S, indices = self.S.sort()
        U = self.U[:, indices]
        Vh = self.Vh[indices, :]
        sum = (S**2).sum()
        threshold = e * sum
        for i, s in enumerate(S):
            sum -= s**2
            if sum < threshold:
                self.S = torch.nn.Parameter(S[i:])
                self.U = torch.nn.Parameter(U[:, i:])
                self.Vh = torch.nn.Parameter(Vh[i:, :])
                break
        return self.S.numel() / len_S

    def forward(self, input: Tensor) -> Tensor:

        if self.decomposing:
            W = self.U @ torch.diag(self.S) @ self.Vh
            return self._conv_forward(
                input,
                W.view(
                    self.out_channels,
                    self.in_channels // self.groups,
                    *self.kernel_size
                ),
                self.bias,
            )
        else:
            return self._conv_forward(input, self.weight, self.bias)
