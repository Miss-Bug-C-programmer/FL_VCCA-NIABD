import types

import pytest
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


def _constant_content_statistics(scores):
    def fake_stats(self, teacher_knowledge, student_logits, proxy_labels):
        del self, student_logits, proxy_labels
        rows = []
        for score in scores[: len(teacher_knowledge)]:
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

    return fake_stats


def test_vcaa_history_window_evicts_old_round_but_keeps_low_proxy_scores():
    controller = VersionContentAwareAdmission(
        VCAAConfig(
            warmup_rounds=0,
            minimum_content_history_size=3,
            history_window_rounds=2,
            minimum_content_cohort_size=1,
            content_scale_floor=0.05,
        ),
        clock=lambda: 2.0,
    )
    for round_number, scores in (
        (1, [0.90, 0.90, 0.90]),
        (2, [0.10, 0.10, 0.10]),
        (3, [0.80, 0.80, 0.80]),
    ):
        teachers = [_knowledge(i, 1.0 + i * 0.1, round_number) for i in range(3)]
        controller._content_statistics = types.MethodType(
            _constant_content_statistics(scores), controller
        )
        decision = controller.evaluate(
            teacher_knowledge=teachers,
            student_logits=torch.zeros(2, 3),
            proxy_labels=torch.tensor([0, 1]),
            current_round=round_number,
        )
        if round_number == 2:
            assert any(not record.content_valid for record in decision.records)

    history = controller.snapshot_state()["history"]
    assert [round_number for round_number, _ in history] == [2, 3]
    assert all(score == 0.10 for score in history[0][1])
    assert all(score == 0.80 for score in history[1][1])


def test_vcaa_snapshot_restores_history_with_rejected_low_proxy_score():
    config = VCAAConfig(
        warmup_rounds=0,
        minimum_content_history_size=3,
        history_window_rounds=2,
        minimum_content_cohort_size=1,
        content_scale_floor=0.05,
    )
    controller = VersionContentAwareAdmission(config, clock=lambda: 2.0)
    for round_number, scores in (
        (1, [0.90, 0.90, 0.90]),
        (2, [0.10, 0.10, 0.10]),
    ):
        teachers = [_knowledge(i, 1.0 + i * 0.1, round_number) for i in range(3)]
        controller._content_statistics = types.MethodType(
            _constant_content_statistics(scores), controller
        )
        controller.evaluate(
            teacher_knowledge=teachers,
            student_logits=torch.zeros(2, 3),
            proxy_labels=torch.tensor([0, 1]),
            current_round=round_number,
        )
    snapshot = controller.snapshot_state()

    teachers = [_knowledge(i, 1.0 + i * 0.1, 3) for i in range(3)]
    controller._content_statistics = types.MethodType(
        _constant_content_statistics([0.80, 0.80, 0.80]), controller
    )
    expected = controller.evaluate(
        teacher_knowledge=teachers,
        student_logits=torch.zeros(2, 3),
        proxy_labels=torch.tensor([0, 1]),
        current_round=3,
    )

    restored = VersionContentAwareAdmission(config, clock=lambda: 2.0)
    restored.restore_state(snapshot)
    restored._content_statistics = types.MethodType(
        _constant_content_statistics([0.80, 0.80, 0.80]), restored
    )
    actual = restored.evaluate(
        teacher_knowledge=teachers,
        student_logits=torch.zeros(2, 3),
        proxy_labels=torch.tensor([0, 1]),
        current_round=3,
    )

    assert snapshot["history"][1][1] == (0.10, 0.10, 0.10)
    assert actual.threshold == pytest.approx(expected.threshold)
    assert actual.admitted_client_ids == expected.admitted_client_ids
    assert actual.normalized_aggregation_weights == pytest.approx(
        expected.normalized_aggregation_weights
    )


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


def test_niabd_freezes_memory_and_threshold_when_normal_consensus_is_insufficient():
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(
            warmup_rounds=1,
            minimum_consensus_teachers=4,
            risk_ema_beta=1.0,
            risk_on=100.0,
            risk_off=0.1,
            onset_patience=5,
            recovery_patience=5,
            stable_patience=5,
        )
    )
    warmup = _cohort(6, 1)
    controller.purify(
        teacher_knowledge=warmup,
        student_logits=torch.zeros_like(warmup[0].logits),
        proxy_labels=torch.tensor([0, 0, 1]),
        current_round=1,
    )
    trusted_before = controller.trusted_mean.clone()
    thresholds_before = controller.thresholds.clone()

    insufficient = _cohort(3, 2)
    result = controller.purify(
        teacher_knowledge=insufficient,
        student_logits=torch.zeros_like(insufficient[0].logits),
        proxy_labels=torch.tensor([0, 0, 1]),
        current_round=2,
    )

    assert controller.phase == "NORMAL"
    assert result.metrics["niabd_threshold_update_mode"] == (
        "frozen_insufficient_consensus"
    )
    assert result.metrics["niabd_trusted_memory_updated"] is False
    assert result.metrics["niabd_memory_updated"] is False
    assert torch.equal(controller.trusted_mean, trusted_before)
    assert torch.equal(controller.thresholds, thresholds_before)


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
