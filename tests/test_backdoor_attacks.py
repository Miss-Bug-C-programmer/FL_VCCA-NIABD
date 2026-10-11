from __future__ import annotations

import math

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from attacks import (
    AttackConfig,
    AttackPlan,
    BackdoorBatchPoisoner,
    evaluate_aa,
    evaluate_basr,
    evaluate_backdoor_suite,
    evaluate_ta,
)
from attacks.trigger import (
    apply_badnets,
    apply_blend,
    apply_dba,
    apply_dynamic,
)
from dataset_metadata import dataset_normalization


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


def test_cinic_badnets_white_patch_uses_official_channel_normalization():
    images = torch.zeros(1, 3, 32, 32)
    out = apply_badnets(
        images,
        size=4,
        value=1.0,
        dataset_name="cinic10",
    )
    expected = torch.tensor(
        [2.1528118, 2.2147079, 2.2010806],
        dtype=out.dtype,
    )
    assert torch.allclose(out[0, :, 27, 27], expected, atol=1e-6)
    assert torch.count_nonzero(out[:, :, :27, :]).item() == 0


def test_mnist_attacks_preserve_grayscale_shape_and_raw_pixel_semantics():
    normalization = dataset_normalization("mnist")
    mean = torch.tensor(normalization.mean).view(1, 1, 1, 1)
    std = torch.tensor(normalization.std).view(1, 1, 1, 1)
    raw = torch.full((2, 1, 28, 28), 0.25)
    images = (raw - mean) / std

    badnets = apply_badnets(
        images,
        size=4,
        value=1.0,
        dataset_name="mnist",
    )
    dba = apply_dba(
        images,
        size=4,
        part=None,
        value=1.0,
        dataset_name="mnist",
    )
    blend = apply_blend(images, alpha=0.2, dataset_name="mnist")
    dynamic = apply_dynamic(
        images,
        size=4,
        round_number=10,
        attack_start_round=10,
        period=10,
        dataset_name="mnist",
    )

    for triggered in (badnets, dba, blend, dynamic):
        assert triggered.shape == images.shape
        raw_triggered = triggered * std + mean
        assert float(raw_triggered.min()) >= -1e-6
        assert float(raw_triggered.max()) <= 1.0 + 1e-6
        assert not torch.equal(triggered, images)

    assert torch.allclose(
        (badnets * std + mean)[:, :, -5:-1, -5:-1],
        torch.ones(2, 1, 4, 4),
        atol=1e-6,
    )


def _cinic_normalize(raw: torch.Tensor) -> torch.Tensor:
    normalization = dataset_normalization("cinic10")
    mean = raw.new_tensor(normalization.mean).view(1, 3, 1, 1)
    std = raw.new_tensor(normalization.std).view(1, 3, 1, 1)
    return (raw - mean) / std


def _cinic_denormalize(value: torch.Tensor) -> torch.Tensor:
    normalization = dataset_normalization("cinic10")
    mean = value.new_tensor(normalization.mean).view(1, 3, 1, 1)
    std = value.new_tensor(normalization.std).view(1, 3, 1, 1)
    return value * std + mean


def test_cinic_dba_global_trigger_is_white_union_of_four_local_parts():
    raw = torch.full((1, 3, 32, 32), 0.25)
    images = _cinic_normalize(raw)
    global_trigger = apply_dba(
        images,
        size=4,
        part=None,
        value=1.0,
        dataset_name="cinic10",
    )
    local_union = images.clone()
    for part in range(4):
        local = apply_dba(
            images,
            size=4,
            part=part,
            value=1.0,
            dataset_name="cinic10",
        )
        changed = local != images
        local_union = torch.where(changed, local, local_union)
    assert torch.equal(global_trigger, local_union)
    raw_global = _cinic_denormalize(global_trigger)
    changed = (raw_global - raw).abs() > 1e-6
    assert changed.any()
    assert torch.allclose(
        raw_global[changed],
        torch.ones_like(raw_global[changed]),
        atol=1e-6,
    )


def test_cinic_blend_matches_raw_pixel_checker_mixture():
    raw = torch.full((1, 3, 32, 32), 0.25)
    blended = apply_blend(
        _cinic_normalize(raw),
        alpha=0.2,
        dataset_name="cinic10",
    )
    raw_blended = _cinic_denormalize(blended)
    values = torch.unique(raw_blended.round(decimals=5))
    expected = torch.tensor([0.2, 0.4], dtype=values.dtype)
    assert torch.allclose(values, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("round_number", "expected_value"),
    [(1, 1.0), (11, 0.8), (21, 0.6), (31, 0.9)],
)
def test_cinic_dynamic_trigger_uses_raw_round_intensity(
    round_number,
    expected_value,
):
    raw = torch.full((1, 3, 32, 32), 0.25)
    dynamic = apply_dynamic(
        _cinic_normalize(raw),
        size=4,
        round_number=round_number,
        attack_start_round=1,
        period=10,
        dataset_name="cinic10",
    )
    raw_dynamic = _cinic_denormalize(dynamic)
    changed = (raw_dynamic - raw).abs() > 1e-5
    assert changed.any()
    assert torch.allclose(
        raw_dynamic[changed],
        torch.full_like(raw_dynamic[changed], expected_value),
        atol=1e-5,
    )


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


def test_ta_aa_exports_validity_and_clean_aa_is_not_zero():
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
    ta = evaluate_ta(_AlwaysTarget(), loader, device="cpu")
    aa = evaluate_aa(
        _AlwaysTarget(),
        loader,
        device="cpu",
        plan=plan,
        round_number=1,
    )
    clean = evaluate_aa(
        _AlwaysTarget(),
        loader,
        device="cpu",
        plan=AttackPlan.build(
            seed=0,
            num_clients=2,
            config=AttackConfig(attack_type="none"),
        ),
        round_number=1,
    )
    assert ta.valid is True
    assert ta.numerator == 2
    assert ta.denominator == 6
    assert aa.valid is True
    assert aa.numerator == 4
    assert aa.denominator == 4
    assert math.isnan(clean.value)
    assert clean.denominator == 0
    assert clean.reason == "not_applicable_clean_run"


def test_strict_ta_aa_do_not_sanitize_nonfinite_logits():
    class Nonfinite(nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 3), float("nan"))

    images = torch.zeros(2, 3, 32, 32)
    labels = torch.tensor([1, 2])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    plan = AttackPlan.build(
        seed=0,
        num_clients=1,
        config=AttackConfig(
            attack_type="badnets",
            target_label=0,
            malicious_fraction=1.0,
            attack_start_round=1,
        ),
    )
    ta = evaluate_ta(
        Nonfinite(), loader, device="cpu", strict_numeric_checks=True
    )
    aa = evaluate_aa(
        Nonfinite(),
        loader,
        device="cpu",
        plan=plan,
        round_number=1,
        strict_numeric_checks=True,
    )
    assert ta.valid is False and math.isnan(ta.value)
    assert aa.valid is False and math.isnan(aa.value)
    assert ta.reason == aa.reason == "nonfinite_logits"


def test_dba_backdoor_suite_exports_all_local_aa_values():
    images = torch.zeros(8, 3, 32, 32)
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    plan = AttackPlan.build(
        seed=0,
        num_clients=4,
        config=AttackConfig(
            attack_type="dba",
            target_label=0,
            malicious_fraction=1.0,
            attack_start_round=1,
        ),
    )
    result = evaluate_backdoor_suite(
        _AlwaysTarget(),
        loader,
        device="cpu",
        plan=plan,
        round_number=1,
        strict_numeric_checks=True,
    )
    assert result["aa_valid"] is True
    assert result["basr_global"] == 1.0
    assert all(result[f"basr_local_{part}"] == 1.0 for part in range(1, 5))
