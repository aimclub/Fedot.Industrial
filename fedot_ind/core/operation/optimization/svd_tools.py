from typing import Optional

import torch
from torch.nn.modules import Module
from torch.nn.modules.conv import Conv2d

from fedot_ind.core.models.cnn.decomposed_conv import DecomposedConv2d


def energy_threshold_pruning(conv: DecomposedConv2d, energy_threshold: float) -> None:
    """Prune the weight matrices to the energy_threshold (in-place).

    Args:
        conv: The optimizable layer.
        energy_threshold: pruning hyperparameter must be in the range (0, 1].
        the lower the threshold, the more singular values will be pruned.

    Raises:
        Assertion Error: If ``conv.decomposing`` is False.
    """
    assert conv.decomposing, "for pruning, the model must be decomposed"
    assert 0 < energy_threshold <= 1, "energy_threshold must be in the range (0, 1]"
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


def decompose_module(model: Module, decomposing_mode: Optional[str] = None) -> None:
    """Replace Conv2d layers with DecomposedConv2d layers in module (in-place).

    Args:
        model: Decomposable module.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
            If ``None`` replace layers without decomposition. Default: ``None``
    """
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
            if decomposing_mode is not None:
                new_module.decompose(decomposing_mode=decomposing_mode)
            setattr(model, name, new_module)


def _load_svd_params(model, state_dict, prefix='') -> None:
    """Loads state_dict to DecomposedConv2d layers in model."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            _load_svd_params(module, state_dict, prefix=f'{prefix}{name}.')

        if isinstance(module, DecomposedConv2d):
            module.set_U_S_Vh(
                u=state_dict[f'{prefix}{name}.U'],
                s=state_dict[f'{prefix}{name}.S'],
                vh=state_dict[f'{prefix}{name}.Vh']
            )


def load_svd_state_dict(
        model: Module,
        decomposing_mode: str,
        state_dict_path: str
) -> None:
    """Loads SVD state_dict to model.

    Args:
        model: An instance of the base model.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        state_dict_path: Path to state_dict file.
    """
    state_dict = torch.load(state_dict_path, map_location='cpu')
    decompose_module(model=model, decomposing_mode=decomposing_mode)
    _load_svd_params(model, state_dict)
    model.load_state_dict(state_dict)
