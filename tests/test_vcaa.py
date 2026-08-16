import torch
import torch.nn as nn

import math
from dataclasses import replace

import pytest

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
            minimum_content_cohort_size=1,
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
    assert decision.admitted_client_ids == (0, 1)
    assert decision.rejected_client_ids == ()
    assert decision.records[0].content_reliability > decision.records[1].content_reliability
    assert decision.records[0].aggregation_weight > decision.records[1].aggregation_weight
    assert decision.vcaa_threshold_used_for_weighting is False
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
    assert math.isnan(decision.records[0].components["content_score"])
    assert decision.records[0].aggregation_weight == 0.0
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
    assert math.isnan(decision.records[0].components["content_score"])
    assert decision.records[0].aggregation_weight == 0.0
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


def _binary_teacher_logits():
    return {
        "good": [[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0], [-8.0, 8.0]],
        "mid": [[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        "bad": [[-8.0, 8.0], [8.0, -8.0], [-8.0, 8.0], [8.0, -8.0]],
    }


def _evaluate_content_batch(
    controller,
    names,
    *,
    current_round=2,
    ages=None,
    source_round=None,
):
    templates = _binary_teacher_logits()
    ages = ages or [0.0] * len(names)
    source_round = current_round if source_round is None else source_round
    knowledge = []
    for client_id, (name, age) in enumerate(zip(names, ages)):
        knowledge.append(
            _fresh_knowledge(
                client_id,
                source_round,
                templates[name],
                generated=0.0,
                consumed=float(age),
            )
        )
    return controller.evaluate(
        teacher_knowledge=knowledge,
        student_logits=torch.zeros(4, 2),
        proxy_labels=_labels(),
        current_round=current_round,
    )


def _records_by_client(decision):
    return {record.client_id: record for record in decision.records}


def _v4_content_config(**kwargs):
    values = {
        "version_weight": 0.0,
        "warmup_rounds": 0,
        "minimum_content_cohort_size": 3,
        "accuracy_weight": 1.0,
        "entropy_weight": 0.0,
        "divergence_weight": 0.0,
        "max_knowledge_age_s": 20.0,
        "age_half_life_s": 10.0,
    }
    values.update(kwargs)
    return VCAAConfig(**values)


def test_content_lower_bound_is_not_used_as_reliability_divisor():
    controller = VersionContentAwareAdmission(
        _v4_content_config(history_window_rounds=3), clock=lambda: 100.0
    )
    _evaluate_content_batch(controller, ["good", "good", "good"], current_round=1)
    decision = _evaluate_content_batch(
        controller, ["good", "good", "good"], current_round=2
    )
    reliabilities = [record.content_reliability for record in decision.records]
    assert all(0.0 < value < 1.0 for value in reliabilities)
    assert decision.content_reliability_saturation_fraction == 0.0
    assert all(
        record.components["vcaa_threshold_used_for_weighting"] is False
        for record in decision.records
    )


def test_post_warmup_content_weights_do_not_artificially_saturate():
    controller = VersionContentAwareAdmission(
        _v4_content_config(minimum_content_cohort_size=3), clock=lambda: 100.0
    )
    decision = _evaluate_content_batch(
        controller, ["good", "mid", "bad"], current_round=2
    )
    reliabilities = [record.content_reliability for record in decision.records]
    assert len(set(reliabilities)) == 3
    assert reliabilities[0] > reliabilities[1] > reliabilities[2]
    assert decision.content_reliability_saturation_fraction < 1.0


def test_warmup_uses_uniform_content_reliability():
    controller = VersionContentAwareAdmission(
        _v4_content_config(warmup_rounds=1), clock=lambda: 100.0
    )
    decision = _evaluate_content_batch(
        controller, ["good", "mid", "bad"], current_round=1
    )
    assert all(record.content_reliability == 1.0 for record in decision.records)
    assert all(record.weighting_mode == "warmup_uniform" for record in decision.records)
    assert decision.content_reliability_saturation_fraction == 1.0


def test_content_quality_never_hard_rejects_fresh_teacher():
    controller = VersionContentAwareAdmission(
        _v4_content_config(minimum_content_cohort_size=1), clock=lambda: 100.0
    )
    decision = _evaluate_content_batch(controller, ["bad"], current_round=2)
    record = decision.records[0]
    assert record.hard_valid is True
    assert record.admitted is True
    assert record.aggregation_weight > 0.0
    assert decision.rejected_client_ids == ()


def test_hard_invalid_teacher_is_excluded_before_consensus():
    config = _v4_content_config()
    base_controller = VersionContentAwareAdmission(config, clock=lambda: 100.0)
    augmented_controller = VersionContentAwareAdmission(config, clock=lambda: 100.0)
    base = _evaluate_content_batch(
        base_controller, ["good", "mid", "bad"], current_round=10
    )
    augmented_knowledge = [
        _fresh_knowledge(0, 10, _binary_teacher_logits()["good"]),
        _fresh_knowledge(1, 10, _binary_teacher_logits()["mid"]),
        _fresh_knowledge(2, 10, _binary_teacher_logits()["bad"]),
        _fresh_knowledge(3, 1, [[1000.0, -1000.0], [-1000.0, 1000.0]]),
    ]
    augmented = augmented_controller.evaluate(
        teacher_knowledge=augmented_knowledge,
        student_logits=torch.zeros(4, 2),
        proxy_labels=_labels(),
        current_round=10,
    )
    left = _records_by_client(base)
    right = _records_by_client(augmented)
    for client_id in (0, 1, 2):
        for key in (
            "consensus_divergence",
            "entropy_deviation",
            "content_score",
            "vcaa_content_reliability",
            "vcaa_aggregation_weight",
        ):
            assert right[client_id].components[key] == pytest.approx(
                left[client_id].components[key], abs=1e-6
            )
    assert not right[3].hard_valid
    assert math.isnan(right[3].components["content_score"])


def test_expired_teacher_is_excluded_before_consensus():
    config = _v4_content_config(max_knowledge_age_s=1.0, age_half_life_s=1.0)
    base_controller = VersionContentAwareAdmission(config, clock=lambda: 100.0)
    augmented_controller = VersionContentAwareAdmission(config, clock=lambda: 100.0)
    base = _evaluate_content_batch(
        base_controller, ["good", "mid", "bad"], current_round=10
    )
    augmented_knowledge = [
        _fresh_knowledge(0, 10, _binary_teacher_logits()["good"]),
        _fresh_knowledge(1, 10, _binary_teacher_logits()["mid"]),
        _fresh_knowledge(2, 10, _binary_teacher_logits()["bad"]),
        _fresh_knowledge(
            3,
            10,
            [[1000.0, -1000.0], [-1000.0, 1000.0]],
            generated=0.0,
            consumed=10.0,
        ),
    ]
    augmented = augmented_controller.evaluate(
        teacher_knowledge=augmented_knowledge,
        student_logits=torch.zeros(4, 2),
        proxy_labels=_labels(),
        current_round=10,
    )
    left = _records_by_client(base)
    right = _records_by_client(augmented)
    assert right[3].hard_rejection_reason == "expired_knowledge_age"
    for client_id in (0, 1, 2):
        assert right[client_id].components["content_score"] == pytest.approx(
            left[client_id].components["content_score"], abs=1e-6
        )


def test_freshness_score_affects_actual_aggregation_weight():
    controller = VersionContentAwareAdmission(
        _v4_content_config(minimum_content_cohort_size=2), clock=lambda: 100.0
    )
    decision = _evaluate_content_batch(
        controller, ["good", "good"], current_round=10, ages=[0.0, 5.0]
    )
    first, second = decision.records
    assert first.content_reliability == pytest.approx(second.content_reliability)
    assert first.freshness_score > second.freshness_score
    assert first.aggregation_weight > second.aggregation_weight


def test_equal_reliability_different_age_changes_aggregated_output():
    from robust_aggregation import aggregate_probabilities

    controller = VersionContentAwareAdmission(
        _v4_content_config(minimum_content_cohort_size=2), clock=lambda: 100.0
    )
    logits = [
        torch.tensor([[8.0, -8.0], [8.0, -8.0]]),
        torch.tensor([[-8.0, 8.0], [-8.0, 8.0]]),
    ]
    decision = controller.evaluate(
        teacher_knowledge=[
            _fresh_knowledge(0, 10, logits[0].tolist(), generated=0.0, consumed=0.0),
            _fresh_knowledge(1, 10, logits[1].tolist(), generated=0.0, consumed=5.0),
        ],
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=10,
    )
    assert decision.records[0].content_reliability == pytest.approx(
        decision.records[1].content_reliability
    )
    weighted = aggregate_probabilities(
        logits, method="mean-soft-probabilities", temperature=1.0,
        weights=[decision.aggregation_weights[index] for index in (0, 1)],
    )
    uniform = aggregate_probabilities(
        logits, method="mean-soft-probabilities", temperature=1.0,
        weights=[1.0, 1.0],
    )
    assert weighted[0, 0] > uniform[0, 0]


def test_near_identical_content_does_not_get_noise_amplified():
    controller = VersionContentAwareAdmission(
        _v4_content_config(minimum_content_cohort_size=3), clock=lambda: 100.0
    )
    base = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    knowledge = [
        _fresh_knowledge(index, 10, (base + index * 1e-4).tolist())
        for index in range(3)
    ]
    decision = controller.evaluate(
        teacher_knowledge=knowledge,
        student_logits=torch.zeros(2, 2),
        proxy_labels=torch.tensor([0, 1]),
        current_round=10,
    )
    weights = list(decision.normalized_aggregation_weights.values())
    assert max(weights) - min(weights) < 0.05


def test_content_weighting_is_permutation_invariant():
    config = _v4_content_config()
    names = ["good", "mid", "bad"]
    first = _evaluate_content_batch(
        VersionContentAwareAdmission(config, clock=lambda: 100.0), names
    )
    templates = _binary_teacher_logits()
    reversed_knowledge = [
        _fresh_knowledge(client_id, 2, templates[name])
        for client_id, name in zip((2, 1, 0), reversed(names))
    ]
    second = VersionContentAwareAdmission(config, clock=lambda: 100.0).evaluate(
        teacher_knowledge=reversed_knowledge,
        student_logits=torch.zeros(4, 2),
        proxy_labels=_labels(),
        current_round=2,
    )
    left = _records_by_client(first)
    right = _records_by_client(second)
    for client_id in left:
        assert right[client_id].components["content_score"] == pytest.approx(
            left[client_id].components["content_score"], abs=1e-6
        )
        assert right[client_id].content_reliability == pytest.approx(
            left[client_id].content_reliability, abs=1e-6
        )
        assert right[client_id].aggregation_weight == pytest.approx(
            left[client_id].aggregation_weight, abs=1e-6
        )
        assert right[client_id].normalized_aggregation_weight == pytest.approx(
            left[client_id].normalized_aggregation_weight, abs=1e-6
        )


def test_no_fresh_teacher_round_is_handled_without_fake_content():
    controller = VersionContentAwareAdmission(
        _v4_content_config(), clock=lambda: 100.0
    )
    decision = controller.evaluate(
        teacher_knowledge=[
            _fresh_knowledge(0, 1, _binary_teacher_logits()["good"]),
            _fresh_knowledge(1, 1, _binary_teacher_logits()["mid"]),
        ],
        student_logits=torch.zeros(4, 2),
        proxy_labels=_labels(),
        current_round=10,
    )
    assert decision.freshness_valid_client_ids == ()
    assert decision.aggregation_weights == {}
    assert decision.history_size == 0
    assert all(math.isnan(record.components["content_score"]) for record in decision.records)


def test_single_or_small_valid_cohort_uses_safe_uniform_content_weighting():
    controller = VersionContentAwareAdmission(
        _v4_content_config(minimum_content_cohort_size=3), clock=lambda: 100.0
    )
    decision = _evaluate_content_batch(
        controller, ["good", "bad"], current_round=2
    )
    assert all(record.content_reliability == 1.0 for record in decision.records)
    assert decision.records[0].aggregation_weight == pytest.approx(
        decision.records[1].aggregation_weight
    )


def test_effective_teacher_count_matches_uniform_case():
    controller = VersionContentAwareAdmission(
        _v4_content_config(minimum_content_cohort_size=3), clock=lambda: 100.0
    )
    decision = _evaluate_content_batch(
        controller, ["good", "good", "good"], current_round=2
    )
    assert decision.effective_teacher_count == pytest.approx(3.0)
    assert decision.weight_cv == pytest.approx(0.0)


def test_effective_teacher_count_decreases_for_skewed_weights():
    controller = VersionContentAwareAdmission(
        _v4_content_config(minimum_content_cohort_size=3), clock=lambda: 100.0
    )
    decision = _evaluate_content_batch(
        controller, ["good", "mid", "bad"], current_round=2
    )
    assert 1.0 <= decision.effective_teacher_count < 3.0
    assert decision.weight_cv > 0.0


def test_normalized_vcaa_weights_sum_to_one():
    decision = _evaluate_content_batch(
        VersionContentAwareAdmission(_v4_content_config(), clock=lambda: 100.0),
        ["good", "mid", "bad"],
    )
    assert sum(decision.normalized_aggregation_weights.values()) == pytest.approx(1.0)
    assert all(
        0.0 < value <= 1.0
        for value in decision.normalized_aggregation_weights.values()
    )


def test_legacy_threshold_is_diagnostic_only():
    common = _v4_content_config(history_window_rounds=3)
    low_beta = VersionContentAwareAdmission(
        replace(common, threshold_beta=0.0), clock=lambda: 100.0
    )
    high_beta = VersionContentAwareAdmission(
        replace(common, threshold_beta=3.0), clock=lambda: 100.0
    )
    for controller in (low_beta, high_beta):
        _evaluate_content_batch(controller, ["good", "mid", "bad"], current_round=1)
    left = _evaluate_content_batch(low_beta, ["good", "mid", "bad"], current_round=2)
    right = _evaluate_content_batch(high_beta, ["good", "mid", "bad"], current_round=2)
    assert left.threshold != right.threshold
    assert [record.content_reliability for record in left.records] == pytest.approx(
        [record.content_reliability for record in right.records]
    )
    assert list(left.aggregation_weights.values()) == pytest.approx(
        list(right.aggregation_weights.values())
    )
    assert left.vcaa_threshold_used_for_weighting is False


def test_vcaa_snapshot_restore_reproduces_next_round_calibration():
    config = _v4_content_config(history_window_rounds=3)
    controller = VersionContentAwareAdmission(config, clock=lambda: 100.0)
    _evaluate_content_batch(controller, ["good", "mid", "bad"], current_round=1)
    snapshot = controller.snapshot_state()
    expected = _evaluate_content_batch(
        controller, ["good", "mid", "bad"], current_round=2
    )

    restored = VersionContentAwareAdmission(config, clock=lambda: 100.0)
    restored.restore_state(snapshot)
    actual = _evaluate_content_batch(
        restored, ["good", "mid", "bad"], current_round=2
    )
    assert actual.threshold == pytest.approx(expected.threshold)
    assert actual.content_score_center == pytest.approx(expected.content_score_center)
    assert actual.content_score_scale == pytest.approx(expected.content_score_scale)
    assert actual.effective_teacher_count == pytest.approx(
        expected.effective_teacher_count
    )
    assert actual.normalized_aggregation_weights == pytest.approx(
        expected.normalized_aggregation_weights
    )
    assert actual.aggregation_weights == pytest.approx(expected.aggregation_weights)
