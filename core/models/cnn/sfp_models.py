from typing import Any, List, Optional, Type, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        planes: int,
        input_size: int,
        output_size: int,
        pruning_ratio: float,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        planes = planes - int(pruning_ratio * planes)
        self.conv1 = conv3x3(input_size, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, output_size)
        self.bn2 = norm_layer(output_size)
        self.downsample = downsample
        self.stride = stride
        if downsample is not None:
            self.register_buffer('indexes', torch.zeros(planes, dtype=torch.int))
        else:
            self.register_buffer('indexes', torch.zeros(input_size, dtype=torch.int))

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.index_add_(1, self.indexes, identity)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        planes: int,
        input_size: int,
        output_size: int,
        pruning_ratio: float,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if downsample is not None:
            s = planes * self.expansion - int(planes * self.expansion * pruning_ratio)
            self.register_buffer('indexes', torch.zeros(s, dtype=torch.int))
        else:
            self.register_buffer('indexes', torch.zeros(input_size, dtype=torch.int))
        norm_layer = nn.BatchNorm2d
        planes = planes - int(pruning_ratio * planes)
        self.conv1 = conv1x1(input_size, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, output_size)
        self.bn3 = norm_layer(output_size)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.index_add_(1, self.indexes, identity)
        out = self.relu(out)

        return out


class ResNetSFP(nn.Module):
    """Pruned ResNet for soft filter pruning optimization.

    Args:
        block: ``'BasicBlock'`` or ``'Bottleneck'``.
        layers: Number of blocks on each layer.
        input_size: Input size of each block.
        output_size: Output size of each block.
        pruning_ratio: Pruning hyperparameter, percentage of pruned filters.
        num_classes: Number of classes.
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        input_size: Dict[str, List[int]],
        output_size: Dict[str, List[int]],
        pruning_ratio: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.pr = pruning_ratio
        self.inplanes = 64
        in_size = self.inplanes - int(self.inplanes * self.pr)
        self.conv1 = nn.Conv2d(3, in_size, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block=block,
            planes=64,
            blocks=layers[0],
            input_size=input_size['layer1'],
            output_size=output_size['layer1'],
        )
        self.layer2 = self._make_layer(
            block=block,
            planes=128,
            blocks=layers[1],
            input_size=input_size['layer2'],
            output_size=output_size['layer2'],
            stride=2)
        self.layer3 = self._make_layer(
            block=block,
            planes=256,
            blocks=layers[2],
            input_size=input_size['layer3'],
            output_size=output_size['layer3'],
            stride=2)
        self.layer4 = self._make_layer(
            block=block,
            planes=512,
            blocks=layers[3],
            input_size=input_size['layer4'],
            output_size=output_size['layer4'],
            stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_size['layer4'][-1], num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        input_size: List[int],
        output_size: List[int],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            out_size = planes * block.expansion
            out_size = out_size - int(out_size * self.pr)
            downsample = nn.Sequential(
                conv1x1(input_size[0], out_size, stride),
                nn.BatchNorm2d(out_size),
            )
        layers = []
        layers.append(
            block(
                planes=planes,
                input_size=input_size[0],
                output_size=output_size[0],
                pruning_ratio=self.pr,
                stride=stride,
                downsample=downsample
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                planes=planes,
                input_size=input_size[i],
                output_size=output_size[i],
                pruning_ratio=self.pr
            ))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def sfp_resnet18(**kwargs: Any) -> ResNetSFP:
    """Pruned ResNet-18."""
    return ResNetSFP(BasicBlock, [2, 2, 2, 2], **kwargs)


def sfp_resnet34(**kwargs: Any) -> ResNetSFP:
    """Pruned ResNet-34."""
    return ResNetSFP(BasicBlock, [3, 4, 6, 3], **kwargs)


def sfp_resnet50(**kwargs: Any) -> ResNetSFP:
    """Pruned ResNet-50."""
    return ResNetSFP(Bottleneck, [3, 4, 6, 3], **kwargs)


def sfp_resnet101(**kwargs: Any) -> ResNetSFP:
    """Pruned ResNet-101."""
    return ResNetSFP(Bottleneck, [3, 4, 23, 3], **kwargs)


def sfp_resnet152(**kwargs: Any) -> ResNetSFP:
    """Pruned ResNet-152."""
    return ResNetSFP(Bottleneck, [3, 8, 36, 3], **kwargs)


SFP_MODELS = {
    "ResNet18": sfp_resnet18,
    "ResNet34": sfp_resnet34,
    "ResNet50": sfp_resnet50,
    "ResNet101": sfp_resnet101,
    "ResNet152": sfp_resnet152,
}
