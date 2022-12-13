from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet


def resnet18_one_channel(**kwargs) -> ResNet:
    """ResNet18 for one input channel"""
    model = resnet18(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def resnet34_one_channel(**kwargs) -> ResNet:
    """ResNet34 for one input channel"""
    model = resnet34(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def resnet50_one_channel(**kwargs) -> ResNet:
    """ResNet50 for one input channel"""
    model = resnet50(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def resnet101_one_channel(**kwargs) -> ResNet:
    """ResNet101 for one input channel"""
    model = resnet101(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def resnet152_one_channel(**kwargs) -> ResNet:
    """ResNet152 for one input channel"""
    model = resnet152(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


CLF_MODELS = {
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'ResNet101': resnet101,
    'ResNet152': resnet152,
    'ResNet18one': resnet18_one_channel,
    'ResNet34one': resnet34_one_channel,
    'ResNet50one': resnet50_one_channel,
    'ResNet101one': resnet101_one_channel,
    'ResNet152one': resnet152_one_channel,
}

CLF_MODELS_ONE_CHANNEL = {
    'ResNet18one': resnet18_one_channel,
    'ResNet34one': resnet34_one_channel,
    'ResNet50one': resnet50_one_channel,
    'ResNet101one': resnet101_one_channel,
    'ResNet152one': resnet152_one_channel,
}
