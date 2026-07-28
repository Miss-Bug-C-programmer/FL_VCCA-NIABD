from attacks import AttackConfig, AttackPlan


def test_attack_plan_is_reproducible():
    cfg = AttackConfig(attack_type="dba", malicious_fraction=.2)
    a = AttackPlan.build(seed=3, num_clients=20, config=cfg)
    b = AttackPlan.build(seed=3, num_clients=20, config=cfg)
    assert a == b
    assert len(a.malicious_client_ids) == 4
    assert sorted(part for _, part in a.dba_trigger_assignments) == [0,1,2,3]
