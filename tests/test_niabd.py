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
    assert float(threshold_after[2]) > float(threshold_before[2])
    assert controller.prototype_mean is not None
    assert float(controller.prototype_mean[2]) < 1.0


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
        NIABDConfig(warmup_rounds=2)
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
