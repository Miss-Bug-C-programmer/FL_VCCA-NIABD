from __future__ import annotations

import torch.nn as nn
from torchvision.models import mobilenet_v2, resnet18


def build_resnet18(
    num_classes: int,
    *,
    in_channels: int = 3,
    small_input: bool = True,
) -> nn.Module:
    """Build the ResNet-18 used by both server and client roles.

    CIFAR-sized inputs use the common 3x3, stride-1 stem and no max-pooling.
    This preserves spatial information that the ImageNet stem would discard on
    32x32 images while retaining the ResNet-18 block structure described in the
    reference system.
    """

    model = resnet18(weights=None, num_classes=int(num_classes))
    if bool(small_input):
        model.conv1 = nn.Conv2d(
            int(in_channels),
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
    elif int(in_channels) != 3:
        model.conv1 = nn.Conv2d(
            int(in_channels),
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
    return model


class SmallCNN(nn.Module):
    """Compact non-ResNet classifier for heterogeneous FD experiments."""

    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, int(num_classes))

    def forward(self, inputs):
        return self.classifier(self.features(inputs).flatten(1))


def build_mobilenet_v2(
    num_classes: int,
    *,
    in_channels: int = 3,
    small_input: bool = True,
) -> nn.Module:
    model = mobilenet_v2(weights=None, num_classes=int(num_classes))
    if bool(small_input):
        model.features[0][0] = nn.Conv2d(
            int(in_channels),
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
    elif int(in_channels) != 3:
        model.features[0][0] = nn.Conv2d(
            int(in_channels),
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
    return model
