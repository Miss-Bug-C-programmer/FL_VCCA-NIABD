import torch

from attacks import AttackConfig, AttackPlan, BackdoorBatchPoisoner
from attacks.trigger import apply_badnets, apply_blend, apply_dba, apply_dynamic


def test_badnets_changes_patch_only():
    x = torch.zeros(2, 3, 32, 32)
    y = apply_badnets(x, size=4)
    assert y.shape == x.shape
    assert int((y != x).sum()) == 2 * 3 * 4 * 4


def test_dba_local_and_global_are_different():
    x = torch.zeros(1, 3, 32, 32)
    local = apply_dba(x, size=4, part=0)
    global_trigger = apply_dba(x, size=4, part=None)
    assert int((global_trigger != x).sum()) == 4 * int((local != x).sum())


def test_poisoner_excludes_target_class():
    cfg = AttackConfig(attack_type="badnets", target_label=0, malicious_fraction=0.25,
                       poison_ratio=1.0, attack_start_round=1)
    plan = AttackPlan.build(seed=0, num_clients=4, config=cfg)
    cid = plan.malicious_client_ids[0]
    poisoner = BackdoorBatchPoisoner(plan=plan, client_id=cid)
    x = torch.zeros(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 0])
    _, out = poisoner(x, labels, round_number=1, batch_index=0)
    assert out.tolist() == [0, 0, 0, 0]
    assert poisoner.last_stats.eligible == 2
    assert poisoner.last_stats.poisoned == 2


def test_dynamic_changes_phase():
    x = torch.zeros(1, 3, 32, 32)
    a = apply_dynamic(x, size=4, round_number=1, attack_start_round=1, period=2)
    b = apply_dynamic(x, size=4, round_number=3, attack_start_round=1, period=2)
    assert not torch.equal(a, b)


def test_blend_is_bounded():
    x = torch.zeros(2, 3, 32, 32)
    y = apply_blend(x, alpha=0.2)
    assert float(y.max()) <= 1.0 and float(y.min()) >= -1.0
