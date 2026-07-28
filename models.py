from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


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
