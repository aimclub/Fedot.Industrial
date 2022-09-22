from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class SimpleConvNet2(nn.Module):
    """Convolutional neural network with two convolutional layers"""

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


MODELS = {
    "SimpleConvNet2": SimpleConvNet2,
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "ResNet152": resnet152,
}
