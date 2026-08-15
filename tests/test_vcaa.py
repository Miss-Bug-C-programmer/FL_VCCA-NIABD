import torch
import torch.nn as nn

from admission import TeacherKnowledge, TeacherMetadata
from vcaa import VCAAConfig, VersionContentAwareAdmission
from niabd import NIABDConfig, NeuroInspiredAdaptiveBackdoorDefense


class _LookupModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("lookup", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs):
        indices = inputs.view(-1).long()
        return self.lookup[indices]


def _metadata(round_number, timestamp=100.0):
    return [
        TeacherMetadata(
            client_id=0,
            model_round=round_number,
            generated_at_s=timestamp,
        ),
        TeacherMetadata(
            client_id=1,
            model_round=round_number,
            generated_at_s=timestamp,
        ),
    ]


def _knowledge(models, round_number):
    return [
        TeacherKnowledge(metadata=metadata, logits=model.lookup.clone())
        for metadata, model in zip(_metadata(round_number), models)
    ]


def _labels():
    return torch.tensor([0, 1, 0, 1])


def test_vcaa_uses_content_quality_and_adaptive_history_threshold():
    good = _LookupModel([[8, -8], [-8, 8], [8, -8], [-8, 8]])
    equally_good = _LookupModel([[7, -7], [-7, 7], [7, -7], [-7, 7]])
    bad = _LookupModel([[-8, 8], [8, -8], [-8, 8], [8, -8]])
    student = _LookupModel([[0, 0], [0, 0], [0, 0], [0, 0]])
    controller = VersionContentAwareAdmission(
        VCAAConfig(
            version_weight=0.0,
            accuracy_weight=1.0,
            entropy_weight=0.0,
            divergence_weight=0.0,
            warmup_rounds=1,
            history_window_rounds=2,
        ),
        clock=lambda: 100.0,
    )

    warmup = controller.evaluate(
        teacher_knowledge=_knowledge([good, equally_good], 1),
        student_logits=student.lookup,
        proxy_labels=_labels(),
        current_round=1,
    )
    decision = controller.evaluate(
        teacher_knowledge=_knowledge([good, bad], 2),
        student_logits=student.lookup,
        proxy_labels=_labels(),
        current_round=2,
    )

    assert warmup.admitted_client_ids == (0, 1)
    assert decision.threshold == 1.0
    assert decision.admitted_client_ids == (0,)
    assert decision.rejected_client_ids == (1,)
    assert decision.records[0].components["proxy_accuracy"] == 1.0
    assert decision.records[1].components["proxy_accuracy"] == 0.0


def test_vcaa_rejects_future_or_collectively_stale_absolute_versions():
    neutral = _LookupModel([[1, 0], [0, 1], [1, 0], [0, 1]])
    student = _LookupModel([[0, 0], [0, 0], [0, 0], [0, 0]])
    controller = VersionContentAwareAdmission(
        VCAAConfig(
            version_weight=1.0,
            max_version_lag=0,
            warmup_rounds=1,
            history_window_rounds=2,
        ),
        clock=lambda: 100.0,
    )

    controller.evaluate(
        teacher_knowledge=_knowledge([neutral, neutral], 3),
        student_logits=student.lookup,
        proxy_labels=_labels(),
        current_round=1,
    )
    decision = controller.evaluate(
        teacher_knowledge=[
            TeacherKnowledge(
                TeacherMetadata(0, 4, 100.0),
                neutral.lookup.clone(),
            ),
            TeacherKnowledge(
                TeacherMetadata(1, 3, 100.0),
                neutral.lookup.clone(),
            ),
        ],
        student_logits=student.lookup,
        proxy_labels=_labels(),
        current_round=2,
    )

    assert decision.admitted_client_ids == ()
    assert decision.rejected_client_ids == (0, 1)
    assert all(not record.hard_valid for record in decision.records)


def _fresh_knowledge(client_id, source_round, logits, *, generated=10.0, consumed=10.0):
    return TeacherKnowledge(
        metadata=TeacherMetadata(
            client_id=client_id,
            model_round=source_round,
            source_round=source_round,
            generated_at_s=generated,
            received_at_s=generated,
            consumed_at_s=consumed,
        ),
        logits=torch.tensor(logits, dtype=torch.float32),
    )


def test_all_teachers_collectively_stale_are_rejected():
    logits = [[4.0, -4.0], [-4.0, 4.0]]
    controller = VersionContentAwareAdmission(
        VCAAConfig(max_version_lag=1, warmup_rounds=0), clock=lambda: 10.0
    )
    decision = controller.evaluate(
        teacher_knowledge=[
            _fresh_knowledge(0, 6, logits),
            _fresh_knowledge(1, 6, logits),
        ],
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=10,
    )
    assert decision.admitted_client_ids == ()
    assert all(record.components["raw_version_lag"] == 4.0 for record in decision.records)
    assert all(record.hard_rejection_reason == "stale_version" for record in decision.records)


def test_received_now_but_generated_long_ago_is_stale():
    controller = VersionContentAwareAdmission(
        VCAAConfig(max_knowledge_age_s=5.0, age_half_life_s=5.0, warmup_rounds=0),
        clock=lambda: 100.0,
    )
    decision = controller.evaluate(
        teacher_knowledge=[
            _fresh_knowledge(0, 10, [[4.0, -4.0], [-4.0, 4.0]], generated=0.0, consumed=20.0),
        ],
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=10,
    )
    record = decision.records[0]
    assert record.components["knowledge_age_s"] == 20.0
    assert record.hard_rejection_reason == "expired_knowledge_age"
    assert record.admitted is False


def test_future_source_round_is_rejected():
    controller = VersionContentAwareAdmission(VCAAConfig(warmup_rounds=0), clock=lambda: 10.0)
    decision = controller.evaluate(
        teacher_knowledge=[_fresh_knowledge(0, 11, [[1.0, 0.0], [0.0, 1.0]])],
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=10,
    )
    assert decision.records[0].hard_rejection_reason == "future_source_round"
    assert decision.admitted_client_ids == ()


def test_high_content_quality_cannot_rescue_hard_stale_teacher():
    controller = VersionContentAwareAdmission(VCAAConfig(warmup_rounds=0), clock=lambda: 10.0)
    decision = controller.evaluate(
        teacher_knowledge=[_fresh_knowledge(0, 1, [[20.0, -20.0], [-20.0, 20.0]])],
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=10,
    )
    assert decision.records[0].components["content_score"] > 0.0
    assert decision.records[0].admitted is False


def test_high_content_quality_cannot_rescue_expired_age():
    controller = VersionContentAwareAdmission(
        VCAAConfig(max_knowledge_age_s=1.0, age_half_life_s=1.0, warmup_rounds=0),
        clock=lambda: 10.0,
    )
    decision = controller.evaluate(
        teacher_knowledge=[_fresh_knowledge(0, 10, [[20.0, -20.0], [-20.0, 20.0]], generated=0.0, consumed=5.0)],
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=10,
    )
    assert decision.records[0].components["content_score"] > 0.0
    assert decision.records[0].admitted is False


def test_fresh_non_iid_teacher_is_not_hard_rejected_for_student_divergence():
    controller = VersionContentAwareAdmission(VCAAConfig(warmup_rounds=0), clock=lambda: 10.0)
    decision = controller.evaluate(
        teacher_knowledge=[_fresh_knowledge(0, 10, [[-20.0, 20.0], [20.0, -20.0]])],
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=10,
    )
    record = decision.records[0]
    assert record.hard_valid is True
    assert record.admitted is True
    assert record.components["mean_kl"] > 0.0


def test_vcaa_history_threshold_uses_robust_median_and_mad():
    controller = VersionContentAwareAdmission(
        VCAAConfig(
            version_weight=0.0,
            accuracy_weight=1.0,
            entropy_weight=0.0,
            divergence_weight=0.0,
            warmup_rounds=0,
            history_window_rounds=5,
        ),
        clock=lambda: 10.0,
    )
    good = [[8.0, -8.0], [-8.0, 8.0]]
    for round_number in (1, 2, 3):
        controller.evaluate(
            teacher_knowledge=[_fresh_knowledge(0, round_number, good)],
            student_logits=torch.zeros(2, 2),
            proxy_labels=torch.tensor([0, 1]),
            current_round=round_number,
        )
    decision = controller.evaluate(
        teacher_knowledge=[_fresh_knowledge(0, 4, [[0.0, 1.0], [1.0, 0.0]])],
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=4,
    )
    assert decision.threshold >= 0.9


def test_vcaa_niabd_pipeline_passes_all_freshness_valid_teachers_to_niabd():
    valid = [[3.0, -3.0], [-3.0, 3.0]]
    divergent = [[-3.0, 3.0], [3.0, -3.0]]
    controller = VersionContentAwareAdmission(
        VCAAConfig(max_version_lag=1, warmup_rounds=0), clock=lambda: 10.0
    )
    decision = controller.evaluate(
        teacher_knowledge=[
            _fresh_knowledge(0, 10, valid),
            _fresh_knowledge(1, 6, valid),
            _fresh_knowledge(2, 10, divergent),
        ],
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=10,
    )
    assert decision.freshness_valid_client_ids == (0, 2)
    assert decision.admitted_client_ids == (0, 2)
    niabd = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1, minimum_consensus_teachers=2)
    )
    knowledge = [
        _fresh_knowledge(0, 10, valid),
        _fresh_knowledge(2, 10, divergent),
    ]
    result = niabd.purify(
        teacher_knowledge=knowledge,
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=1,
    )
    assert len(result.records) == len(decision.freshness_valid_client_ids)


def test_vcaa_configuration_rejects_invalid_content_weights():
    try:
        VCAAConfig(
            accuracy_weight=0.5,
            entropy_weight=0.5,
            divergence_weight=0.5,
        )
    except ValueError as exc:
        assert "sum to 1" in str(exc)
    else:
        raise AssertionError("Expected invalid VCAA weights to be rejected.")
