from typing import Callable, Dict
from torch.nn.modules.conv import Conv2d
from torch.nn.modules import Module

from core.models.cnn.decomposed_conv import DecomposedConv2d


class StructureOptimizer:
    """Base class for structure optimizers."""
    def __init__(self) -> None:
        super(StructureOptimizer, self).__init__()


class EnergyThresholdPruning(StructureOptimizer):
    """Prune the weight matrices to the energy threshold."""

    def __init__(self, energy_threshold: float) -> None:
        super(StructureOptimizer, self).__init__()
        self.energy_threshold = energy_threshold

    def optimize(self, conv: DecomposedConv2d) -> float:
        """Prune the weight matrices to the self.energy_threshold.
        Returns the compression ratio.
        """

        assert conv.decomposing, "for pruning, the model must be decomposed"
        len_S = conv.S.numel()
        S, indices = conv.S.sort()
        U = conv.U[:, indices]
        Vh = conv.Vh[indices, :]
        sum = (S**2).sum()
        threshold = self.energy_threshold * sum
        for i, s in enumerate(S):
            sum -= s**2
            if sum < threshold:
                conv.set_U_S_Vh(U[:, i:], S[i:], Vh[i:, :])
                break
        return 1 - conv.S.numel() / len_S


def decompose_module(model: Module, decomposing_mode: str = "channel") -> None:
    """Replace Conv2d layers with DecomposedConv2d layers in module (in-place)."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            decompose_module(module, decomposing_mode="channel")

        if isinstance(module, Conv2d):
            new_module = DecomposedConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
                decomposing=False,
            )
            new_module.load_state_dict(module.state_dict())
            new_module.decompose(decomposing_mode=decomposing_mode)
            setattr(model, name, new_module)


def prune_model(model: Module, optimizer: StructureOptimizer) -> float:
    """Prune DecomposedConv2d layers of the model with pruning_fn.
    Returns the average by layers compression ratio.
    """
    compression = 0
    n = 0
    for module in model.modules():
        if isinstance(module, DecomposedConv2d):
            n += 1
            compression += optimizer.optimize(module)
    return compression / n


