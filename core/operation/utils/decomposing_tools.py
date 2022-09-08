from torch.nn.modules.conv import Conv2d
from torch.nn.modules import Module
from core.models.cnn.decomposed_conv import DecomposedConv2d


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


def prune_model(model: Module, e: float) -> float:
    """Prune DecomposedConv2d layers of the model to the energy threshold.
    Returns the average by layers compression ratio.
    """
    compression = 0
    n = 0
    for module in model.modules():
        if isinstance(module, DecomposedConv2d):
            n += 1
            compression += module.pruning(e)
    return compression / n
