from typing import Union

import torch
from torch import Tensor
from torch.nn import Conv2d, Parameter
from torch.nn.common_types import _size_2_t

__all__ = ["DecomposedConv2d"]


class DecomposedConv2d(Conv2d):
    """Extends the Conv2d layer by implementing the singular value decomposition of
    the weight matrix.

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: ``1``
        padding: Padding added to all four sides of the input. Default: ``0``
        padding_mode: ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'zeros'``
        decomposing: If ``True``, decomposes weights after initialization.
            Default: ``True``
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
            Default: ``'channel'``
        dilation: Spacing between kernel elements. Default: ``1``
        groups: Number of blocked connections from input channels to output channels.
            Default: ``1``
        bias: If ``True``, adds a learnable bias to the output. Default: ``True``

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
        padding_mode: str = 'zeros',
        decomposing: bool = True,
        decomposing_mode: str = 'channel',
        device=None,
        dtype=None,
    ) -> None:

        super().__init__(
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

        n, c, w, h = self.weight.size()
        self.decomposing_modes_dict = {
            'channel': (n, c * w * h),
            'spatial': (n * w, c * h),
        }

        if decomposing:
            self.decompose(decomposing_mode)
        else:
            self.U = None
            self.S = None
            self.Vh = None
            self.decomposing = False

    def decompose(self, decomposing_mode: str) -> None:
        """Decomposes the weight matrix in singular value decomposition.
        Replaces the weights with U, S, Vh matrices such that weights = U * S * Vh.
        Args:
            decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        Raises:
            ValueError: If ``decomposing_mode`` not in valid values.
        """
        if decomposing_mode not in self.decomposing_modes_dict.keys():
            raise ValueError(
                "decomposing_mode must be one of {}, but got decomposing_mode='{}'".format(
                    self.decomposing_modes_dict.keys(), decomposing_mode
                )
            )
        W = self.weight.view(self.decomposing_modes_dict[decomposing_mode])
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        self.U = Parameter(U)
        self.S = Parameter(S)
        self.Vh = Parameter(Vh)
        self.register_parameter('weight', None)
        self.decomposing = True

    def compose(self) -> None:
        """Compose the weight matrix from singular value decomposition.
        Replaces U, S, Vh matrices with weights such that weights = U * S * Vh.
        """

        W = self.U @ torch.diag(self.S) @ self.Vh
        self.weight = Parameter(
            W.view(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size
            )
        )

        self.register_parameter('U', None)
        self.register_parameter('S', None)
        self.register_parameter('Vh', None)
        self.decomposing = False

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

    def set_U_S_Vh(self, u: Tensor, s: Tensor, vh: Tensor) -> None:
        """Update U, S, Vh matrices.
        Raises:
            Assertion Error: If ``self.decomposing`` is False.
        """
        assert self.decomposing, "for setting U, S and Vh, the model must be decomposed"
        self.U = Parameter(u)
        self.S = Parameter(s)
        self.Vh = Parameter(vh)