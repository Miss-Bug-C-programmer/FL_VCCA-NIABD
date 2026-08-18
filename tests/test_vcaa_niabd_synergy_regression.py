import types

import torch

from admission import TeacherKnowledge, TeacherMetadata
from niabd import NIABDConfig, NeuroInspiredAdaptiveBackdoorDefense
from vcaa import VCAAConfig, VersionContentAwareAdmission


def _knowledge(client_id: int, value: float, round_number: int) -> TeacherKnowledge:
    logits = torch.tensor(
        [[value, 0.0, -value], [0.0, value, -value]],
        dtype=torch.float32,
    )
    return TeacherKnowledge(
        metadata=TeacherMetadata(
            client_id=client_id,
            model_round=round_number,
            source_round=round_number,
            generated_at_s=float(round_number),
            received_at_s=float(round_number),
            consumed_at_s=float(round_number),
            proxy_version="p",
        ),
        logits=logits,
    )


def test_vcaa_history_keeps_hard_valid_rejects_to_avoid_survivor_bias():
    controller = VersionContentAwareAdmission(
        VCAAConfig(
            warmup_rounds=0,
            minimum_content_history_size=3,
            history_window_rounds=5,
            content_scale_floor=0.05,
        ),
        clock=lambda: 2.0,
    )
    controller._history.append((1, (0.9, 0.8, 0.7)))
    controlled = [0.90, 0.80, 0.10]

    def fake_stats(self, teacher_knowledge, student_logits, proxy_labels):
        del self, student_logits, proxy_labels
        rows = []
        for score in controlled[: len(teacher_knowledge)]:
            rows.append(
                {
                    "proxy_accuracy": score,
                    "mean_entropy": 0.0,
                    "entropy_deviation": 0.0,
                    "mean_kl": 0.0,
                    "consensus_divergence": 0.0,
                    "num_classes": 3.0,
                    "accuracy_term": score,
                    "entropy_term": score,
                    "divergence_term": score,
                    "sanitized_value_count": 0.0,
                }
            )
        return rows

    controller._content_statistics = types.MethodType(fake_stats, controller)
    teachers = [_knowledge(i, 1.0 + i * 0.1, 2) for i in range(3)]
    decision = controller.evaluate(
        teacher_knowledge=teachers,
        student_logits=torch.zeros(2, 3),
        proxy_labels=torch.tensor([0, 1]),
        current_round=2,
    )
    assert any(record.hard_valid and not record.content_valid for record in decision.records)
    latest_scores = controller.snapshot_state()["history"][-1][1]
    assert len(latest_scores) == 3
    assert min(latest_scores) == 0.10
    assert decision.content_threshold_source == "historical_median_minus_mad_floor"


def _cohort(count: int, round_number: int, shift: float = 0.0):
    base = torch.tensor(
        [[2.0, 0.1, -1.0], [1.8, 0.2, -0.8], [-0.6, 2.1, 0.0]],
        dtype=torch.float32,
    )
    result = []
    for client_id in range(count):
        jitter = (client_id - count / 2) * 0.01
        result.append(
            TeacherKnowledge(
                metadata=TeacherMetadata(
                    client_id=client_id,
                    model_round=round_number,
                    source_round=round_number,
                    generated_at_s=float(round_number),
                    proxy_version="p",
                ),
                logits=base + shift + jitter,
            )
        )
    return result


def test_niabd_uses_reference_cohort_without_authorizing_reference_only_packets():
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1, minimum_consensus_teachers=4)
    )
    reference = _cohort(6, 1)
    action = reference[:2]
    warmup = controller.purify(
        teacher_knowledge=action,
        reference_knowledge=reference,
        student_logits=torch.zeros_like(action[0].logits),
        proxy_labels=torch.tensor([0, 0, 1]),
        current_round=1,
    )
    assert controller.trusted_mean is not None
    assert len(warmup.purified_knowledge) == 2
    assert {x.metadata.client_id for x in warmup.purified_knowledge} == {0, 1}

    shifted_reference = _cohort(6, 2, shift=0.25)
    result = controller.purify(
        teacher_knowledge=shifted_reference[:2],
        reference_knowledge=shifted_reference,
        student_logits=torch.zeros_like(action[0].logits),
        proxy_labels=torch.tensor([0, 0, 1]),
        current_round=2,
    )
    assert result.metrics["niabd_reference_teachers"] == 6
    assert result.metrics["niabd_action_teachers"] == 2
    assert len(result.records) == 2
    assert max(record.teacher_memory_score for record in result.records) <= 12.0


def test_consensus_aware_purification_does_not_suppress_collective_benign_drift():
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(
            warmup_rounds=1,
            minimum_consensus_teachers=4,
            initial_threshold=0.5,
            minimum_threshold=0.5,
            consensus_purification_threshold=1.5,
        )
    )
    first = _cohort(6, 1)
    controller.purify(
        teacher_knowledge=first,
        student_logits=torch.zeros_like(first[0].logits),
        proxy_labels=torch.tensor([0, 0, 1]),
        current_round=1,
    )
    shifted = _cohort(6, 2, shift=0.4)
    result = controller.purify(
        teacher_knowledge=shifted,
        student_logits=torch.zeros_like(first[0].logits),
        proxy_labels=torch.tensor([0, 0, 1]),
        current_round=2,
    )
    assert result.metrics["mean_suppression"] < 1e-4
    for original, purified in zip(shifted, result.purified_knowledge):
        assert torch.allclose(original.logits, purified.logits, atol=1e-4)
