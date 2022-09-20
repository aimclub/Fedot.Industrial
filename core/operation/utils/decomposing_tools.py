import torch
from torch.nn.modules.conv import Conv2d
from torch.nn.modules import Module
from torch.linalg import vector_norm

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

    def optimize(self, conv: DecomposedConv2d) -> (int, int):
        """Prune the weight matrices to the self.energy_threshold.
        Return the number of parameters before and after pruning.
        """

        assert conv.decomposing, "for pruning, the model must be decomposed"
        num_old_params = conv.S.numel() + conv.U.numel() + conv.Vh.numel()
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
        num_new_params = conv.S.numel() + conv.U.numel() + conv.Vh.numel()
        return num_old_params, num_new_params


class SoftFilterPruning(StructureOptimizer):
    """Zerolize filters of convolutional layer to the pruning ratio"""

    def __init__(self, pruning_ratio: float) -> None:
        super(StructureOptimizer, self).__init__()
        self.pruning_ratio = pruning_ratio

    def optimize(self, conv: DecomposedConv2d) -> (int, int):
        """Zerolize filters of convolutional layer to the self.pruning_percent.
        Return the number of zero parameters before and after pruning.
        """
        num_old_params = torch.count_nonzero(conv.weight)
        filter_pruned_num = int(conv.weight.size()[0] * self.pruning_ratio)
        filter_norms = vector_norm(conv.weight, dim=(1, 2, 3))
        _, indices = filter_norms.sort()
        with torch.no_grad():
            conv.weight[indices[:filter_pruned_num]] = 0
        num_new_params = torch.count_nonzero(conv.weight)
        return num_old_params, num_new_params


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


def prune_model(model: Module, optimizer: StructureOptimizer) -> (int, int):
    """Prune Conv2d layers of the model with pruning_fn.
    Return the number of parameters before and after pruning.
    """
    old_params_counter = 0
    new_params_counter = 0
    for module in model.modules():
        if isinstance(module, Conv2d):
            num_old_params, num_new_params = optimizer.optimize(module)
            old_params_counter += num_old_params
            new_params_counter += num_new_params
    return old_params_counter, new_params_counter
