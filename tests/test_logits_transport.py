import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from federated_client import FederatedClient
from federated_server import FederatedServer
from admission import TeacherKnowledge, TeacherMetadata
from logits_transport import ClientLogitsPacket
from trainer import predict_logits


def _proxy_loader():
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
    )
    labels = torch.tensor([0, 1, 1, 0])
    return DataLoader(
        TensorDataset(inputs, labels),
        batch_size=2,
        shuffle=False,
    )


def test_client_packet_roundtrip_is_a_detached_byte_upload():
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    packet = ClientLogitsPacket.from_logits(
        client_id=3,
        model_round=7,
        generated_at_s=12.0,
        query_id="query-7",
        logits=logits,
    )
    logits[0, 0] = -99.0

    decoded = packet.decode_logits()
    assert decoded.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert packet.payload_bytes == 4 * 4
    assert not hasattr(packet, "model")
    assert not hasattr(packet, "train_loader")


def test_real_client_upload_matches_its_local_model_prediction():
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0], [-1.0, 2.0]]))
        model.bias.zero_()
    loader = _proxy_loader()
    client = FederatedClient(
        client_id=0,
        model=model,
        train_loader=loader,
        device="cpu",
    )
    expected = predict_logits(model, loader, device="cpu")

    packet = client.upload_proxy_logits(
        loader,
        query_id="round-1",
    )

    assert torch.equal(packet.decode_logits(), expected)
    assert packet.model_round == 0
    assert packet.generated_at_s > 0.0


def test_server_updates_from_uploaded_logits_without_a_client_model_reference():
    proxy_loader = _proxy_loader()
    server_model = nn.Linear(2, 2)
    server = FederatedServer(
        model=server_model,
        proxy_loader=proxy_loader,
        device="cpu",
        strict_numeric_checks=True,
    )
    packet = ClientLogitsPacket.from_logits(
        client_id=0,
        model_round=1,
        generated_at_s=10.0,
        query_id="round-1",
        logits=torch.tensor(
            [[8.0, -8.0], [-8.0, 8.0], [-8.0, 8.0], [8.0, -8.0]]
        ),
    )
    before = copy.deepcopy(server_model.state_dict())

    knowledge = server.receive_client_uploads(
        [packet],
        query_id="round-1",
        expected_client_ids=[0],
    )
    aggregate = server.aggregate_admitted_logits(knowledge, [0])
    updated = server.train_from_uploaded_logits(
        aggregate,
        learning_rate=0.01,
        temperature=2.0,
    )

    assert updated is True
    assert all(
        "client" not in attribute.lower()
        for attribute in vars(server)
    )
    assert any(
        not torch.equal(before[key], server_model.state_dict()[key])
        for key in before
    )


def test_server_rejects_a_packet_for_a_different_proxy_query():
    server = FederatedServer(
        model=nn.Linear(2, 2),
        proxy_loader=_proxy_loader(),
        device="cpu",
    )
    packet = ClientLogitsPacket.from_logits(
        client_id=0,
        model_round=1,
        generated_at_s=10.0,
        query_id="old-query",
        logits=torch.zeros(4, 2),
    )

    try:
        server.receive_client_uploads(
            [packet],
            query_id="new-query",
            expected_client_ids=[0],
        )
    except ValueError as exc:
        assert "stale query" in str(exc)
    else:
        raise AssertionError("Expected stale proxy-query upload rejection.")


def test_server_aggregates_temperature_softened_teacher_probabilities():
    knowledge = {
        0: TeacherKnowledge(
            TeacherMetadata(0, 1, 1.0),
            torch.tensor([[4.0, 0.0]]),
        ),
        1: TeacherKnowledge(
            TeacherMetadata(1, 1, 1.0),
            torch.tensor([[0.0, 0.0]]),
        ),
    }
    aggregate = FederatedServer.aggregate_admitted_probabilities(
        knowledge,
        [0, 1],
        temperature=2.0,
    )
    expected = (
        torch.softmax(knowledge[0].logits / 2.0, dim=1)
        + torch.softmax(knowledge[1].logits / 2.0, dim=1)
    ) / 2.0

    assert aggregate is not None
    assert torch.allclose(aggregate, expected)
