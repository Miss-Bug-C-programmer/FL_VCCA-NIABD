import torch

from admission import TeacherKnowledge, TeacherMetadata
from niabd import NIABDConfig, NeuroInspiredAdaptiveBackdoorDefense


def _knowledge(client_id, logits, round_number):
    return TeacherKnowledge(
        metadata=TeacherMetadata(
            client_id=client_id,
            model_round=round_number,
            generated_at_s=float(round_number),
        ),
        logits=torch.tensor(logits, dtype=torch.float32),
    )


def _normal_round(round_number):
    return [
        _knowledge(
            0,
            [
                [2.0, 0.1, -1.0],
                [1.8, 0.2, -0.8],
                [-0.6, 2.1, 0.0],
                [-0.5, 1.9, 0.1],
            ],
            round_number,
        ),
        _knowledge(
            1,
            [
                [2.1, 0.0, -0.9],
                [1.9, 0.1, -0.7],
                [-0.7, 2.0, 0.1],
                [-0.4, 1.8, 0.0],
            ],
            round_number,
        ),
    ]


def test_niabd_suppresses_an_extreme_target_logit_and_protects_memory():
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(
            initial_threshold=2.0,
            minimum_standard_deviation=0.2,
            benign_deviation_limit=4.0,
            warmup_rounds=1,
            threshold_learning_rate=0.1,
            minimum_consensus_teachers=2,
        )
    )
    labels = torch.tensor([0, 0, 1, 1])
    student = torch.zeros(4, 3)
    warmup = controller.purify(
        teacher_knowledge=_normal_round(1),
        student_logits=student,
        proxy_labels=labels,
        current_round=1,
    )
    threshold_before = controller.thresholds
    malicious_logits = [
        [2.0, 0.1, 20.0],
        [1.8, 0.2, 20.0],
        [-0.6, 2.1, 20.0],
        [-0.5, 1.9, 20.0],
    ]
    result = controller.purify(
        teacher_knowledge=[
            _normal_round(2)[0],
            _knowledge(1, malicious_logits, 2),
        ],
        student_logits=student,
        proxy_labels=labels,
        current_round=2,
    )
    threshold_after = controller.thresholds

    purified_malicious = result.purified_knowledge[1].logits
    assert warmup.metrics["warmup"] == 1.0
    assert result.metrics["anomaly_fraction"] > 0.0
    assert result.records[1].mean_suppression > 0.0
    assert result.records[1].memory_eligible is False
    assert float(purified_malicious[:, 2].max()) < 20.0
    assert threshold_before is not None
    assert threshold_after is not None
    # The abnormal teacher is excluded from threshold exposure.  A malicious
    # outlier must not potentiate the threshold (threshold poisoning).
    assert float(threshold_after[2]) <= float(threshold_before[2])
    assert controller.prototype_mean is not None
    assert float(controller.prototype_mean[:, 2].mean()) < 1.0


def test_niabd_leaves_in_threshold_logits_unchanged():
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1)
    )
    labels = torch.tensor([0, 0, 1, 1])
    student = torch.zeros(4, 3)
    controller.purify(
        teacher_knowledge=_normal_round(1),
        student_logits=student,
        proxy_labels=labels,
        current_round=1,
    )
    normal = _normal_round(2)
    result = controller.purify(
        teacher_knowledge=normal,
        student_logits=student,
        proxy_labels=labels,
        current_round=2,
    )

    for original, purified in zip(normal, result.purified_knowledge):
        assert torch.allclose(
            purified.logits,
            original.logits,
            atol=1e-5,
        )


def test_niabd_warmup_updates_prototype_memory_each_round():
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=2, minimum_consensus_teachers=2)
    )
    labels = torch.tensor([0, 0, 1, 1])
    student = torch.zeros(4, 3)
    first = controller.purify(
        teacher_knowledge=_normal_round(1),
        student_logits=student,
        proxy_labels=labels,
        current_round=1,
    )
    second = controller.purify(
        teacher_knowledge=_normal_round(2),
        student_logits=student,
        proxy_labels=labels,
        current_round=2,
    )

    assert first.metrics["prototype_observations"] == 8.0
    assert second.metrics["prototype_observations"] == 16.0


def test_niabd_configuration_rejects_invalid_threshold_bounds():
    try:
        NIABDConfig(
            initial_threshold=2.0,
            minimum_threshold=3.0,
            maximum_threshold=4.0,
        )
    except ValueError as exc:
        assert "initial_threshold" in str(exc)
    else:
        raise AssertionError("Expected invalid NIABD thresholds to fail.")


def _teachers_from_tensor(values, round_number):
    return [
        _knowledge(index, values[index].tolist(), round_number)
        for index in range(int(values.shape[0]))
    ]


def _run_controller(controller, values, round_number):
    return controller.purify(
        teacher_knowledge=_teachers_from_tensor(values, round_number),
        student_logits=torch.zeros_like(values[0]),
        proxy_labels=torch.zeros(values.shape[1], dtype=torch.long),
        current_round=round_number,
    )


def test_niabd_automatically_protects_memory_and_recovers_without_oracle():
    base = torch.zeros(6, 4, 3)
    config = NIABDConfig(
        warmup_rounds=1,
        minimum_consensus_teachers=4,
        risk_ema_beta=1.0,
        risk_on=0.5,
        risk_off=0.2,
        onset_patience=1,
        recovery_patience=1,
        stable_patience=1,
        prototype_learning_rate=0.01,
        recovery_memory_lr=0.5,
    )
    controller = NeuroInspiredAdaptiveBackdoorDefense(config)
    _run_controller(controller, base, 1)
    trusted_before = controller.trusted_mean
    assert trusted_before is not None
    suspicious = _run_controller(controller, base + 0.5, 2)
    assert controller.phase == "SUSPICIOUS"
    assert suspicious.metrics["niabd_trusted_memory_frozen"] is True
    assert torch.allclose(controller.trusted_mean, trusted_before)
    recovery = _run_controller(controller, base, 3)
    assert controller.phase == "RECOVERY"
    assert recovery.metrics["niabd_threshold_update_mode"] == "recovery_clipped_recalibration"
    normal = _run_controller(controller, base, 4)
    assert controller.phase == "NORMAL"
    assert normal.metrics["niabd_recovery_stable_rounds"] == 0


def test_niabd_checkpoint_restores_controller_state_and_next_output():
    base = torch.zeros(6, 4, 3)
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(
            warmup_rounds=1,
            minimum_consensus_teachers=4,
            risk_ema_beta=1.0,
            risk_on=0.5,
            risk_off=0.2,
            onset_patience=1,
            recovery_patience=1,
            stable_patience=1,
        )
    )
    _run_controller(controller, base, 1)
    _run_controller(controller, base + 0.5, 2)
    for values, round_number in ((base, 3), (base, 4), (base + 0.5, 5)):
        snapshot = controller.snapshot_state()
        restored = NeuroInspiredAdaptiveBackdoorDefense(controller.config)
        restored.restore_state(snapshot)
        left = _run_controller(controller, values, round_number)
        right = _run_controller(restored, values, round_number)
        assert controller.phase == restored.phase
        assert controller.risk_ema == restored.risk_ema
        assert left.metrics == right.metrics
        for left_item, right_item in zip(left.purified_knowledge, right.purified_knowledge):
            assert torch.equal(left_item.logits, right_item.logits)
