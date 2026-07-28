import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch

from federated_client import FederatedClient
from federated_server import FederatedServer


def _loader():
    dataset = TensorDataset(
        torch.randn(4, 2),
        torch.tensor([0, 1, 0, 1]),
    )
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_server_client_runtime_has_a_strict_model_ownership_boundary():
    client_model = nn.Linear(2, 2)
    server_model = nn.Linear(2, 2)
    client = FederatedClient(
        client_id=0,
        model=client_model,
        train_loader=_loader(),
        device="cpu",
    )
    server = FederatedServer(
        model=server_model,
        proxy_loader=_loader(),
        device="cpu",
    )

    assert client.model is client_model
    assert server.model is server_model
    assert all(
        "client" not in attribute.lower()
        for attribute in vars(server)
    )
