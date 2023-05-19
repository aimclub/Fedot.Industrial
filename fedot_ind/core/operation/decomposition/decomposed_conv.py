import torch
from torch import Tensor
from torch.nn import Conv2d, Parameter

from fedot_ind.core.architecture.abstraction.сheckers import parameter_value_check


class DecomposedConv2d(Conv2d):
    """Extends the Conv2d layer by implementing the singular value decomposition of
    the weight matrix.

    Args:
        base_conv:  The convolutional layer whose parameters will be copied
        decomposing: If ``True``, decomposes weights after initialization.
            Default: ``True``
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
            Default: ``'channel'``
    """

    def __init__(
        self,
        base_conv: Conv2d,
        decomposing: bool = True,
        decomposing_mode: str = 'channel',
        device=None,
        dtype=None,
    ) -> None:

        super().__init__(
            base_conv.in_channels,
            base_conv.out_channels,
            base_conv.kernel_size,
            base_conv.stride,
            base_conv.padding,
            base_conv.dilation,
            base_conv.groups,
            (base_conv.bias is not None),
            base_conv.padding_mode,
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
        parameter_value_check(
            parameter='decomposing_mode',
            value=decomposing_mode,
            valid_values=set(self.decomposing_modes_dict.keys())
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