import torch
import torch.nn as nn

from admission import TeacherKnowledge, TeacherMetadata
from vcaa import VCAAConfig, VersionContentAwareAdmission


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


def test_vcaa_rejects_a_version_below_the_dynamic_version_floor():
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

    assert decision.admitted_client_ids == (0,)
    assert decision.rejected_client_ids == (1,)
    assert decision.records[0].components["version_score"] == 1.0
    assert decision.records[1].components["version_score"] == 0.0


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
