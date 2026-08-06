from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import List, Sequence

import torch
import torch.nn as nn

from models import build_mobilenet_v2, build_resnet18, SmallCNN


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
    spec = dataset_spec(dataset_name)
    if architecture == "resnet18":
        model = build_resnet18(
            spec.num_classes,
            in_channels=spec.in_channels,
            small_input=spec.small_input,
        )
    elif architecture in {"small_cnn", "smallcnn"}:
        model = SmallCNN(spec.num_classes, in_channels=spec.in_channels)
    elif architecture in {"mobilenet_v2", "mobilenet-v2", "mobilenet"}:
        model = build_mobilenet_v2(
            spec.num_classes,
            in_channels=spec.in_channels,
            small_input=spec.small_input,
        )
    else:
        raise ValueError(
            f"Unsupported architecture={architecture!r}; expected resnet18, small_cnn, or mobilenet_v2."
        )
    return model.to(torch.device(device))


def build_models(
    dataset_name: str,
    num_clients: int,
    device,
    *,
    server_architecture: str = "resnet18",
    client_architecture: str = "resnet18",
    client_architectures: Sequence[str] | None = None,
) -> tuple[List[nn.Module], nn.Module]:
    """Build one global student and one local teacher per client."""

    if int(num_clients) <= 0:
        raise ValueError("num_clients must be positive")

    assignments = (
        [str(value) for value in client_architectures]
        if client_architectures is not None
        else [str(client_architecture)] * int(num_clients)
    )
    if len(assignments) != int(num_clients):
        raise ValueError("client_architectures must contain one value per client.")
    clients = [
        build_model(
            assignments[client_id],
            dataset_name=dataset_name,
            device=device,
        )
        for client_id in range(int(num_clients))
    ]
    server = build_model(
        server_architecture,
        dataset_name=dataset_name,
        device=device,
    )
    return clients, server


def architecture_assignment_hash(
    *,
    server_architecture: str,
    client_architectures: Sequence[str],
) -> str:
    payload = {
        "server": str(server_architecture).lower(),
        "clients": [str(value).lower() for value in client_architectures],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
