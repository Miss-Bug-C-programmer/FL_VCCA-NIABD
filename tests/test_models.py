import torch
import torch.nn as nn

from model_factory import build_models, dataset_spec
from models import build_resnet18


def test_resnet18_uses_cifar_stem_and_requested_classes():
    model = build_resnet18(num_classes=10, small_input=True)

    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)
    assert model.fc.out_features == 10
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_factory_builds_one_server_and_direct_client_teachers():
    clients, server = build_models(
        dataset_name="cifar10",
        num_clients=2,
        device="cpu",
    )

    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert clients[0] is not server
    assert clients[0].fc.out_features == 10
    assert server.fc.out_features == 10
    assert next(clients[0].parameters()).data_ptr() != next(server.parameters()).data_ptr()


def test_femnist_keeps_existing_62_class_scope():
    assert dataset_spec("femnist").num_classes == 62
    clients, server = build_models(
        dataset_name="femnist",
        num_clients=1,
        device="cpu",
    )
    assert clients[0].fc.out_features == 62
    assert server.fc.out_features == 62
