import torch
from torch.utils.data import DataLoader, TensorDataset

from federated_client import FederatedClient
from federated_server import FederatedServer
from model_factory import build_models


def quick_validate() -> None:
    clients, server = build_models(
        dataset_name="cifar10",
        num_clients=2,
        device="cpu",
    )
    sample = torch.randn(2, 3, 32, 32)
    server_logits = server(sample)
    client_logits = [client(sample) for client in clients]

    assert server_logits.shape == (2, 10)
    assert all(logits.shape == (2, 10) for logits in client_logits)

    labels = torch.tensor([0, 1])
    proxy_loader = DataLoader(
        TensorDataset(sample, labels),
        batch_size=2,
        shuffle=False,
    )
    client_runtimes = [
        FederatedClient(
            client_id=client_id,
            model=model,
            train_loader=proxy_loader,
            device="cpu",
        )
        for client_id, model in enumerate(clients)
    ]
    server_runtime = FederatedServer(
        model=server,
        proxy_loader=proxy_loader,
        device="cpu",
    )
    packets = [
        client.upload_proxy_logits(
            proxy_loader,
            query_id="example-query",
        )
        for client in client_runtimes
    ]
    knowledge = server_runtime.receive_client_uploads(
        packets,
        query_id="example-query",
        expected_client_ids=[0, 1],
    )
    aggregate = server_runtime.aggregate_admitted_logits(
        knowledge,
        [0, 1],
    )

    assert aggregate is not None
    assert aggregate.shape == (2, 10)
    assert all(not hasattr(packet, "model") for packet in packets)
    print(
        "real client logits upload ready: "
        f"server={type(server).__name__}, clients={len(clients)}, "
        f"logits={tuple(server_logits.shape)}, "
        f"upload_bytes={sum(packet.payload_bytes for packet in packets)}"
    )


if __name__ == "__main__":
    quick_validate()
