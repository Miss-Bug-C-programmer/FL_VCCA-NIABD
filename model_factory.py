from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from models import build_resnet18


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int
    in_channels: int = 3
    small_input: bool = True


DATASET_SPECS = {
    "cifar10": DatasetSpec(num_classes=10),
    "cifar100": DatasetSpec(num_classes=100),
    "femnist": DatasetSpec(num_classes=62),
    "cinic10": DatasetSpec(num_classes=10),
    "tiny-imagenet-200": DatasetSpec(num_classes=200),
}


def dataset_spec(dataset_name: str) -> DatasetSpec:
    name = str(dataset_name).lower()
    if name in {"tiny_imagenet_200", "tinyimagenet200"}:
        name = "tiny-imagenet-200"
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r}; expected one of: {supported}"
        ) from exc


def build_model(
    architecture: str,
    *,
    dataset_name: str,
    device,
) -> nn.Module:
    architecture = str(architecture).lower()
    if architecture != "resnet18":
        raise ValueError(
            f"Unsupported architecture={architecture!r}; this iteration supports resnet18."
        )

    spec = dataset_spec(dataset_name)
    return build_resnet18(
        spec.num_classes,
        in_channels=spec.in_channels,
        small_input=spec.small_input,
    ).to(torch.device(device))


def build_models(
    dataset_name: str,
    num_clients: int,
    device,
    *,
    server_architecture: str = "resnet18",
    client_architecture: str = "resnet18",
) -> tuple[List[nn.Module], nn.Module]:
    """Build one global student and one local teacher per client."""

    if int(num_clients) <= 0:
        raise ValueError("num_clients must be positive")

    clients = [
        build_model(
            client_architecture,
            dataset_name=dataset_name,
            device=device,
        )
        for _ in range(int(num_clients))
    ]
    server = build_model(
        server_architecture,
        dataset_name=dataset_name,
        device=device,
    )
    return clients, server
