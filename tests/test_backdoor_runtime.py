from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from attacks import AttackConfig, AttackPlan
from federated_runtime import run_fedagg_server_client
from niabd import NIABDConfig, NeuroInspiredAdaptiveBackdoorDefense


class _TinyVisionModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def _loaders(num_clients=2):
    generator = torch.Generator().manual_seed(91)
    x = torch.rand(24, 3, 8, 8, generator=generator) * 2.0 - 1.0
    y = torch.tensor([0, 1, 2] * 8)
    dataset = TensorDataset(x, y)
    return {
        "client": [
            DataLoader(dataset, batch_size=6, shuffle=False)
            for _ in range(num_clients)
        ],
        "proxy": DataLoader(dataset, batch_size=6, shuffle=False),
        "test": DataLoader(dataset, batch_size=6, shuffle=False),
    }


def test_real_batch_poisoning_flows_through_serialized_logits_and_niabd():
    plan = AttackPlan.build(
        seed=4,
        num_clients=2,
        config=AttackConfig(
            attack_type="badnets",
            target_label=0,
            malicious_fraction=0.5,
            poison_ratio=0.5,
            attack_start_round=2,
            attack_end_round=3,
            trigger_size=2,
        ),
    )
    metrics = run_fedagg_server_client(
        [_TinyVisionModel(), _TinyVisionModel()],
        _TinyVisionModel(),
        _loaders(),
        device="cpu",
        rounds=3,
        local_epochs=1,
        learning_rate=0.01,
        defense_controller=NeuroInspiredAdaptiveBackdoorDefense(
            NIABDConfig(warmup_rounds=1)
        ),
        attack_plan=plan,
    )
    assert metrics["knowledge_interface"] == "serialized-proxy-logits"
    assert metrics["attack_type"] == "badnets"
    assert metrics["poisoned_samples"][0] == 0
    assert metrics["poisoned_samples"][1] > 0
    assert metrics["poisoned_samples"][2] > 0
    assert len(metrics["basr_global"]) == 3
    assert all(0.0 <= value <= 1.0 for value in metrics["basr_global"])
    assert len(metrics["backdoor_client_records"]) == 3
    assert any(
        row["is_malicious"]
        for row in metrics["backdoor_client_records"][1]
    )
    assert metrics["niabd_enabled"] == 1


def test_triggered_no_poison_keeps_basr_and_reports_zero_poisoning():
    plan = AttackPlan.build(
        seed=7,
        num_clients=2,
        config=AttackConfig(
            attack_type="badnets",
            target_label=0,
            malicious_fraction=0.5,
            poison_ratio=0.0,
            attack_start_round=1,
            attack_end_round=1,
            trigger_size=2,
        ),
    )
    metrics = run_fedagg_server_client(
        [_TinyVisionModel(), _TinyVisionModel()],
        _TinyVisionModel(),
        _loaders(),
        device="cpu",
        rounds=1,
        local_epochs=1,
        learning_rate=0.01,
        attack_plan=plan,
    )

    assert metrics["poisoned_samples"] == [0]
    assert metrics["basr_global_denominator"][0] > 0
    assert 0.0 <= metrics["basr_global"][0] <= 1.0
