from torch.nn.modules import Module
from torch.nn.modules.conv import Conv2d

from core.models.cnn.decomposed_conv import DecomposedConv2d


def energy_threshold_pruning(conv: DecomposedConv2d, energy_threshold: float) -> None:
    """Prune the weight matrices to the energy_threshold (in-place)."""
    assert conv.decomposing, "for pruning, the model must be decomposed"
    S, indices = conv.S.sort()
    U = conv.U[:, indices]
    Vh = conv.Vh[indices, :]
    sum = (S ** 2).sum()
    threshold = energy_threshold * sum
    for i, s in enumerate(S):
        sum -= s ** 2
        if sum < threshold:
            conv.set_U_S_Vh(U[:, i:].clone(), S[i:].clone(), Vh[i:, :].clone())
            break


def decompose_module(model: Module, decomposing_mode: str = "channel") -> None:
    """Replace Conv2d layers with DecomposedConv2d layers in module (in-place)."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            decompose_module(module, decomposing_mode=decomposing_mode)

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
