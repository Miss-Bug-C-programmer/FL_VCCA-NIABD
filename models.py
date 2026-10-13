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


class PreActBasicBlock(nn.Module):
    """Full pre-activation residual block used by ResNet-18 v2."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(int(in_channels))
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            stride=int(stride),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(int(out_channels))
        self.conv2 = nn.Conv2d(
            int(out_channels),
            int(out_channels),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.shortcut = (
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=1,
                stride=int(stride),
                bias=False,
            )
            if int(stride) != 1 or int(in_channels) != int(out_channels)
            else None
        )

    def forward(self, inputs):
        preactivated = self.relu(self.bn1(inputs))
        shortcut = (
            self.shortcut(preactivated) if self.shortcut is not None else inputs
        )
        output = self.conv1(preactivated)
        output = self.conv2(self.relu(self.bn2(output)))
        return output + shortcut


class PreActResNet18(nn.Module):
    """ResNet-18 v2 with full pre-activation and a small-image stem."""

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 3,
        small_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = 64
        if bool(small_input):
            self.conv1 = nn.Conv2d(
                int(in_channels), 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                int(in_channels), 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, int(num_classes))
        self._initialize_parameters()

    def _make_layer(self, out_channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        strides = [int(stride)] + [1] * (int(blocks) - 1)
        layers = []
        for block_stride in strides:
            layers.append(
                PreActBasicBlock(
                    self.in_channels,
                    int(out_channels),
                    stride=block_stride,
                )
            )
            self.in_channels = int(out_channels)
        return nn.Sequential(*layers)

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        output = self.maxpool(self.conv1(inputs))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.relu(self.bn(output))
        output = self.avgpool(output)
        return self.fc(output.flatten(1))


def build_resnet18_v2(
    num_classes: int,
    *,
    in_channels: int = 3,
    small_input: bool = True,
) -> nn.Module:
    return PreActResNet18(
        int(num_classes),
        in_channels=int(in_channels),
        small_input=bool(small_input),
    )


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
