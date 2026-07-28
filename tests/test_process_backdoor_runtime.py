from __future__ import annotations

import torch

from attacks import AttackConfig, AttackPlan
from data_utils import (
    build_federated_data_plan,
    build_server_dataloaders_from_plan,
    cleanup_dataloaders,
)
from model_factory import build_model
from process_runtime import ProcessRuntimeConfig, run_fedagg_server_client_process_async
from runtime_trace import generate_runtime_trace


def _write_femnist(root):
    generator = torch.Generator().manual_seed(812)
    train_x = torch.randint(
        0, 256, (48, 28, 28), dtype=torch.uint8, generator=generator
    )
    train_y = torch.tensor([0, 1, 2, 3, 4, 5] * 8)
    test_x = torch.randint(
        0, 256, (18, 28, 28), dtype=torch.uint8, generator=generator
    )
    test_y = torch.tensor([0, 1, 2, 3, 4, 5] * 3)
    torch.save({"x": train_x, "y": train_y}, root / "femnist_train.pt")
    torch.save({"x": test_x, "y": test_y}, root / "femnist_test.pt")


def _profile():
    return {
        "name": "backdoor-homogeneous-test",
        "slow_client_fraction": 0.0,
        "normal_compute_slowdown_factor": 1.0,
        "slow_compute_slowdown_factor": 1.0,
        "normal_upload_delay_s": 0.0,
        "slow_upload_delay_s": 0.0,
        "availability_probability": 1.0,
        "upload_attempt_drop_probability": 0.0,
        "ack_delay_probability": 0.0,
        "ack_delay_s": 0.0,
        "events": {},
    }


def test_process_runtime_uses_client_local_poisoning_without_packet_oracle(tmp_path):
    _write_femnist(tmp_path)
    plan = build_federated_data_plan(
        dataset_path=str(tmp_path),
        dataset_name="femnist",
        num_clients=2,
        batch_size=8,
        seed=0,
        partition_scheme="iid",
        label_skew_classes=2,
        quantity_skew_alpha=0.5,
        dirichlet_alpha=0.5,
        val_ratio=0.1,
        proxy_ratio=0.1,
        proxy_dataset_size=6,
    )
    loaders = build_server_dataloaders_from_plan(plan)
    trace = generate_runtime_trace(
        profile=_profile(),
        seed=0,
        num_clients=2,
        rounds=2,
        warmup_rounds=1,
        participation_rate=1.0,
    )
    attack_plan = AttackPlan.build(
        seed=0,
        num_clients=2,
        config=AttackConfig(
            attack_type="badnets",
            target_label=0,
            malicious_fraction=0.5,
            poison_ratio=0.5,
            attack_start_round=1,
            attack_end_round=2,
            trigger_size=3,
        ),
    )
    config = ProcessRuntimeConfig(
        quorum_fraction=1.0,
        warmup_rounds=1,
        soft_deadline_override_s=5.0,
        hard_deadline_override_s=10.0,
        rpc_timeout_s=1.0,
        max_retries=1,
        registration_timeout_s=30.0,
        shutdown_timeout_s=30.0,
        server_device="cpu",
        client_device="cpu",
        strict_numeric_checks=True,
    )
    server = build_model("resnet18", dataset_name="femnist", device="cpu")
    try:
        metrics = run_fedagg_server_client_process_async(
            server_model=server,
            server_dataloaders=loaders,
            data_plan=plan,
            trace=trace,
            config=config,
            local_epochs=1,
            rounds=2,
            learning_rate=0.01,
            distill_temperature=2.0,
            attack_plan=attack_plan,
        )
    finally:
        cleanup_dataloaders(loaders)

    assert metrics["knowledge_interface"] == "localhost-tcp-serialized-proxy-logits"
    assert metrics["server_has_client_model_refs"] is False
    assert metrics["attack_type"] == "badnets"
    assert sum(metrics["poisoned_samples"]) > 0
    assert len(metrics["basr_global"]) == 2
    assert all(0.0 <= value <= 1.0 for value in metrics["basr_global"])
    assert all(
        "is_malicious" not in str(event.get("packet_id", "")).lower()
        for event in metrics["runtime_events"]
    )
