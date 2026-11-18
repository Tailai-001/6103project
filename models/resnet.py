
import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes=100):
    """Four-stage CIFAR ResNet-18."""

    model = resnet18(weights=None)

    # CIFAR-100: 32x32 images
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # Remove ImageNet initial max pooling
    model.maxpool = nn.Identity()

    # CIFAR-100 has 100 classes
    model.fc = nn.Linear(512, num_classes)

    return model


if __name__ == "__main__":
    model = build_resnet18()

    x = torch.randn(4, 3, 32, 32)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == (4, 100)
    print("ResNet-18 forward test passed!")
