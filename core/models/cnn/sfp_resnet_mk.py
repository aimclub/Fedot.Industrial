from typing import Any, List, Optional, Type, Union

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
        inplanes: int,
        planes: int,
        pruning_ratio: float,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        pruned_planes = planes - int(pruning_ratio * planes)
        self.conv1 = conv3x3(inplanes, pruned_planes, stride)
        self.bn1 = norm_layer(pruned_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(pruned_planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        pruning_ratio: float,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        pruned_planes = planes - int(pruning_ratio * planes)
        self.conv1 = conv1x1(inplanes, pruned_planes)
        self.bn1 = norm_layer(pruned_planes)
        self.conv2 = conv3x3(pruned_planes, pruned_planes, stride)
        self.bn2 = norm_layer(pruned_planes)
        self.conv3 = conv1x1(pruned_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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

        out += identity
        out = self.relu(out)

        return out


class ResNetSFP(nn.Module):
    """Pruned ResNet for soft filter pruning optimization.

    Args:
        block: ``'BasicBlock'`` or ``'Bottleneck'``.
        layers: Number of blocks on each layer.
        pruning_ratio: Pruning hyperparameter, percentage of pruned filters.
        num_classes: Number of classes.
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pruning_ratio: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.pr = pruning_ratio
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, self.pr, stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.pr))
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


SFP_MODELS_FOR_MK = {
    "ResNet18": sfp_resnet18,
    "ResNet34": sfp_resnet34,
    "ResNet50": sfp_resnet50,
    "ResNet101": sfp_resnet101,
    "ResNet152": sfp_resnet152,
}
