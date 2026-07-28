from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from attacks import (
    AttackConfig,
    AttackPlan,
    BackdoorBatchPoisoner,
    evaluate_basr,
)
from attacks.trigger import (
    apply_badnets,
    apply_blend,
    apply_dba,
    apply_dynamic,
)


def test_attack_plan_is_deterministic_and_strategy_independent():
    cfg = AttackConfig(
        attack_type="dba",
        malicious_fraction=0.2,
        attack_start_round=3,
    )
    first = AttackPlan.build(seed=7, num_clients=20, config=cfg)
    second = AttackPlan.build(seed=7, num_clients=20, config=cfg)
    assert first == second
    assert len(first.malicious_client_ids) == 4
    assert sorted(part for _, part in first.dba_trigger_assignments) == [0, 1, 2, 3]


def test_badnets_trigger_changes_only_expected_patch():
    images = torch.zeros(2, 3, 32, 32)
    out = apply_badnets(images, size=4, value=1.0)
    assert out.shape == images.shape
    changed = (out != images).sum().item()
    assert changed == 2 * 3 * 4 * 4


def test_dba_global_trigger_is_union_of_four_local_triggers():
    images = torch.zeros(1, 3, 32, 32)
    global_trigger = apply_dba(images, size=4, part=None)
    local_union = images.clone()
    for part in range(4):
        local = apply_dba(images, size=4, part=part)
        local_union = torch.maximum(local_union, local)
    assert torch.equal(global_trigger, local_union)


def test_blend_and_dynamic_preserve_tensor_shape_and_range():
    images = torch.zeros(4, 3, 32, 32)
    blend = apply_blend(images, alpha=0.2)
    dynamic_1 = apply_dynamic(
        images,
        size=4,
        round_number=5,
        attack_start_round=5,
        period=2,
    )
    dynamic_2 = apply_dynamic(
        images,
        size=4,
        round_number=7,
        attack_start_round=5,
        period=2,
    )
    assert blend.shape == images.shape
    assert dynamic_1.shape == images.shape
    assert float(blend.min()) >= -1.0
    assert float(blend.max()) <= 1.0
    assert not torch.equal(dynamic_1, dynamic_2)


def test_poisoner_excludes_original_target_class_and_is_deterministic():
    config = AttackConfig(
        attack_type="badnets",
        target_label=0,
        malicious_fraction=0.5,
        poison_ratio=0.5,
        attack_start_round=2,
    )
    plan = AttackPlan.build(seed=11, num_clients=2, config=config)
    malicious_id = plan.malicious_client_ids[0]
    poisoner_a = BackdoorBatchPoisoner(plan=plan, client_id=malicious_id)
    poisoner_b = BackdoorBatchPoisoner(plan=plan, client_id=malicious_id)
    images = torch.zeros(8, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    clean_x, clean_y = poisoner_a(
        images,
        labels,
        round_number=1,
        batch_index=1,
    )
    assert torch.equal(clean_x, images)
    assert torch.equal(clean_y, labels)

    out_a, labels_a = poisoner_a(
        images,
        labels,
        round_number=2,
        batch_index=1,
    )
    out_b, labels_b = poisoner_b(
        images,
        labels,
        round_number=2,
        batch_index=1,
    )
    assert torch.equal(out_a, out_b)
    assert torch.equal(labels_a, labels_b)
    assert labels_a[0].item() == 0
    changed_to_target = int(((labels != 0) & (labels_a == 0)).sum().item())
    assert changed_to_target == 4
    assert poisoner_a.round_stats.poisoned == 4


class _AlwaysTarget(nn.Module):
    def forward(self, x):
        logits = torch.zeros(x.shape[0], 3, device=x.device)
        logits[:, 0] = 10.0
        return logits


def test_basr_excludes_samples_already_in_target_class():
    images = torch.zeros(6, 3, 32, 32)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    plan = AttackPlan.build(
        seed=0,
        num_clients=2,
        config=AttackConfig(
            attack_type="badnets",
            target_label=0,
            malicious_fraction=0.5,
            attack_start_round=1,
        ),
    )
    result = evaluate_basr(
        _AlwaysTarget(),
        loader,
        device="cpu",
        plan=plan,
        round_number=1,
    )
    assert result.denominator == 4
    assert result.numerator == 4
    assert result.basr == 1.0
