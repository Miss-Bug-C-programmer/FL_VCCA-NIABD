from attacks import AttackConfig, AttackPlan
import pytest


def test_attack_plan_is_reproducible():
    cfg = AttackConfig(attack_type="dba", malicious_fraction=.2)
    a = AttackPlan.build(seed=3, num_clients=20, config=cfg)
    b = AttackPlan.build(seed=3, num_clients=20, config=cfg)
    assert a == b
    assert len(a.malicious_client_ids) == 4
    assert sorted(part for _, part in a.dba_trigger_assignments) == [0,1,2,3]


def test_data_balanced_attack_plan_matches_sample_mass_without_labels():
    counts = [96, 68, 141, 140, 91, 73, 144, 124, 121, 122,
              160, 63, 93, 82, 67, 43, 110, 118, 69, 75]
    cfg = AttackConfig(
        attack_type="badnets",
        malicious_fraction=0.2,
        malicious_selection="data-balanced",
    )
    plan = AttackPlan.build(
        seed=2,
        num_clients=20,
        config=cfg,
        client_sample_counts=counts,
    )
    selected_mass = sum(counts[index] for index in plan.malicious_client_ids)
    assert len(plan.malicious_client_ids) == 4
    assert selected_mass == 400
    assert plan.to_dict()["malicious_sample_fraction"] == 0.2


def test_attack_plan_persists_non_cifar_dataset_identity(tmp_path):
    cfg = AttackConfig(attack_type="badnets", attack_start_round=1)
    plan = AttackPlan.build(
        seed=2,
        num_clients=4,
        config=cfg,
        dataset_name="cinic10",
    )
    path = tmp_path / "plan.json"
    plan.save(path)
    assert AttackPlan.load(path) == plan
    assert plan.to_dict()["dataset_name"] == "cinic10"


def test_cifar_attack_plan_serialization_keeps_legacy_shape():
    cfg = AttackConfig(attack_type="badnets", attack_start_round=1)
    plan = AttackPlan.build(seed=2, num_clients=4, config=cfg)
    assert plan.dataset_name == "cifar10"
    assert "dataset_name" not in plan.to_dict()


def test_cinic_attack_plan_rejects_raw_trigger_outside_pixel_range():
    cfg = AttackConfig(
        attack_type="badnets",
        attack_start_round=1,
        trigger_value=-0.5,
    )
    with pytest.raises(ValueError, match="raw pixel intensity"):
        AttackPlan.build(
            seed=2,
            num_clients=4,
            config=cfg,
            dataset_name="cinic10",
        )

    # The CIFAR path intentionally retains its established normalized range.
    cifar = AttackPlan.build(seed=2, num_clients=4, config=cfg)
    assert cifar.config.trigger_value == -0.5
