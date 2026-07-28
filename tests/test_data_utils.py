import torch

from data_utils import cleanup_dataloaders, get_dataloaders


def _write_fake_femnist(root):
    generator = torch.Generator().manual_seed(3)
    train_x = torch.randint(
        0,
        256,
        (60, 28, 28),
        dtype=torch.uint8,
        generator=generator,
    )
    train_y = torch.randint(0, 62, (60,), generator=generator)
    test_x = torch.randint(
        0,
        256,
        (20, 28, 28),
        dtype=torch.uint8,
        generator=generator,
    )
    test_y = torch.randint(0, 62, (20,), generator=generator)
    torch.save({"x": train_x, "y": train_y}, root / "femnist_train.pt")
    torch.save({"x": test_x, "y": test_y}, root / "femnist_test.pt")


def test_dataloaders_expose_client_server_contract_without_legacy_tiers(tmp_path):
    _write_fake_femnist(tmp_path)
    dataloaders = get_dataloaders(
        dataset_path=str(tmp_path),
        dataset_name="femnist",
        num_clients=3,
        batch_size=4,
        seed=0,
        partition_scheme="iid",
        num_workers=0,
        auxiliary_num_workers=0,
    )
    try:
        assert len(dataloaders["client"]) == 3
        assert {"proxy", "val", "test"}.issubset(dataloaders)
        assert not {"end", "edge", "cloud", "bridge"}.intersection(dataloaders)
        images, labels = next(iter(dataloaders["client"][0]))
        assert images.shape[1:] == (3, 32, 32)
        assert labels.ndim == 1
        assert dataloaders["loader_config"]["client_num_workers"] == 0
    finally:
        cleanup_dataloaders(dataloaders)


def test_client_partitions_cover_the_private_training_split(tmp_path):
    _write_fake_femnist(tmp_path)
    dataloaders = get_dataloaders(
        dataset_path=str(tmp_path),
        dataset_name="femnist",
        num_clients=4,
        batch_size=4,
        seed=11,
        partition_scheme="quantity-skew",
        num_workers=0,
        auxiliary_num_workers=0,
    )
    try:
        partition_total = sum(
            item["num_samples"] for item in dataloaders["partition_stats"]
        )
        assert partition_total == dataloaders["split_sizes"]["train"]
        assert all(
            item["num_samples"] > 0 for item in dataloaders["partition_stats"]
        )
    finally:
        cleanup_dataloaders(dataloaders)
