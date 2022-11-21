from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet


class SimpleConvNet2(nn.Module):
    """Convolutional neural network with two convolutional layers

    Args:
        num_classes: number of classes.
    """

    def __init__(self, num_classes: int):
        super(SimpleConvNet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


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


MODELS = {
    'SimpleConvNet2': SimpleConvNet2,
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
