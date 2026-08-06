import math

import pytest
import torch

from admission import TeacherKnowledge, TeacherMetadata
from niabd import NIABDConfig, NeuroInspiredAdaptiveBackdoorDefense
from numeric_integrity import NumericIntegrityError


def _knowledge(client_id: int, logits: torch.Tensor, round_number: int) -> TeacherKnowledge:
    return TeacherKnowledge(
        metadata=TeacherMetadata(
            client_id=client_id,
            model_round=round_number,
            generated_at_s=float(round_number),
        ),
        logits=logits.clone(),
    )


def _teachers(logits: torch.Tensor, round_number: int) -> list[TeacherKnowledge]:
    return [
        _knowledge(index, logits[index], round_number)
        for index in range(int(logits.shape[0]))
    ]


def _stable_logits(
    *,
    teachers: int = 20,
    samples: int = 32,
    classes: int = 10,
    seed: int = 11,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        teachers,
        samples,
        classes,
        generator=generator,
    ) * 0.02


def _purify(
    controller: NeuroInspiredAdaptiveBackdoorDefense,
    logits: torch.Tensor,
    round_number: int,
    labels: torch.Tensor | None = None,
):
    labels = labels if labels is not None else torch.zeros(logits.shape[1], dtype=torch.long)
    return controller.purify(
        teacher_knowledge=_teachers(logits, round_number),
        student_logits=torch.zeros_like(logits[0]),
        proxy_labels=labels,
        current_round=round_number,
    )


def test_single_finite_extreme_does_not_freeze_teacher_memory():
    logits = _stable_logits(samples=100, classes=10)
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1, minimum_standard_deviation=0.1)
    )
    _purify(controller, logits, 1)
    before = controller.prototype_mean
    assert before is not None

    outlier = logits.clone()
    outlier[0, 7, 3] = 25.0
    result = _purify(controller, outlier, 2)

    assert result.metrics["prototype_updated"] == 1.0
    assert result.metrics["niabd_prototype_update_reason"] == "normal_eligible_update"
    assert result.records[0].memory_eligible is False
    assert result.records[0].mean_suppression > 0.0
    purified = result.purified_knowledge[0].logits
    assert float(purified[7, 3]) < 25.0
    assert torch.allclose(
        purified[0, :],
        outlier[0, 0, :],
        atol=1e-5,
    )
    after = controller.prototype_mean
    assert after is not None
    assert torch.allclose(after, before, atol=0.1)


def test_large_benign_proxy_set_keeps_updating_memory():
    logits = _stable_logits(samples=5000, classes=10)
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1)
    )
    first = _purify(controller, logits, 1)
    second = _purify(controller, logits + 0.001, 2)

    assert first.metrics["niabd_eligible_teacher_observations"] == 100000
    assert second.metrics["prototype_updated"] == 1.0
    assert second.metrics["memory_eligible_teachers"] == 20
    assert second.metrics["prototype_observations"] == 200000
    assert second.metrics["niabd_all_ineligible_round"] == 0.0
    assert all(
        math.isfinite(float(value))
        for value in second.metrics.values()
        if isinstance(value, (float, int))
        and not (isinstance(value, float) and math.isnan(value))
    )


def test_abnormal_minority_cannot_move_robust_prototype():
    first = _stable_logits(teachers=20, samples=40, classes=3)
    second = first.clone()
    second[16:] += 8.0
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1)
    )
    _purify(controller, first, 1)
    result = _purify(controller, second, 2)

    assert result.metrics["prototype_updated"] == 1.0
    assert result.metrics["memory_eligible_teachers"] >= 15
    assert sum(not record.memory_eligible for record in result.records) >= 4
    assert sum(
        record.mean_suppression > 0.0 for record in result.records[16:]
    ) == 4
    prototype = controller.prototype_mean
    assert prototype is not None
    assert float(prototype.mean()) < 1.0


def test_consensus_drift_can_recover_without_absorbing_split_clusters():
    first = _stable_logits(teachers=8, samples=16, classes=3)
    drifted = first + 0.5
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1, minimum_standard_deviation=0.1)
    )
    _purify(controller, first, 1)
    drift = _purify(controller, drifted, 2)
    assert drift.metrics["niabd_prototype_update_reason"] == "consensus_drift_update"
    assert drift.metrics["prototype_updated"] == 1.0

    split = first.clone()
    split[:4] += 10.0
    split[4:] -= 10.0
    observations_before = controller.observation_count
    frozen = _purify(controller, split, 3)
    assert frozen.metrics["prototype_updated"] == 0.0
    assert frozen.metrics["niabd_prototype_update_reason"] == "freeze_no_safe_candidate"
    assert controller.observation_count == observations_before


def test_proxy_conditioned_prototype_preserves_sample_semantics():
    base = torch.full((6, 4, 3), -2.0)
    for sample in range(4):
        base[:, sample, sample % 3] = 4.0
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1)
    )
    _purify(controller, base, 1)
    result = _purify(controller, base, 2)

    prototype = controller.prototype_mean
    variance = controller.prototype_variance
    assert prototype is not None and variance is not None
    assert tuple(prototype.shape) == (4, 3)
    assert tuple(variance.shape) == (4, 3)
    assert prototype.argmax(dim=1).tolist() == [0, 1, 2, 0]
    assert all(
        item.logits.argmax(dim=1).tolist() == [0, 1, 2, 0]
        for item in result.purified_knowledge
    )


def test_student_reference_is_aligned_per_proxy_sample():
    logits = _stable_logits(teachers=6, samples=12, classes=3)
    controller = NeuroInspiredAdaptiveBackdoorDefense(
        NIABDConfig(warmup_rounds=1, reference_source="student")
    )
    _purify(controller, logits, 1)
    anomalous = logits.clone()
    anomalous[0, :, 2] += 20.0
    student = torch.ones_like(anomalous[0]) * 0.25
    result = controller.purify(
        teacher_knowledge=_teachers(anomalous, 2),
        student_logits=student,
        proxy_labels=torch.arange(12) % 3,
        current_round=2,
    )
    assert result.purified_knowledge[0].logits.shape == student.shape
    assert float(result.purified_knowledge[0].logits[:, 2].mean()) < 1.0
    assert all(torch.isfinite(item.logits).all() for item in result.purified_knowledge)


def test_proxy_labels_do_not_change_niabd_output_or_metrics():
    logits = _stable_logits(teachers=6, samples=12, classes=3)
    first = NeuroInspiredAdaptiveBackdoorDefense(NIABDConfig(warmup_rounds=1))
    second = NeuroInspiredAdaptiveBackdoorDefense(NIABDConfig(warmup_rounds=1))
    _purify(first, logits, 1, torch.zeros(12, dtype=torch.long))
    _purify(second, logits, 1, torch.ones(12, dtype=torch.long))
    result_a = _purify(first, logits, 2, torch.zeros(12, dtype=torch.long))
    result_b = _purify(second, logits, 2, torch.arange(12) % 3)

    assert result_a.metrics == result_b.metrics
    for left, right in zip(result_a.purified_knowledge, result_b.purified_knowledge):
        assert torch.equal(left.logits, right.logits)


def test_memory_shape_change_requires_explicit_reset_and_nonfinite_is_rejected():
    controller = NeuroInspiredAdaptiveBackdoorDefense(NIABDConfig(warmup_rounds=1))
    _purify(controller, _stable_logits(samples=8, classes=3), 1)
    with pytest.raises(RuntimeError, match=r"call reset\(\)"):
        _purify(controller, _stable_logits(samples=9, classes=3), 2)
    with pytest.raises(NumericIntegrityError):
        _purify(controller, torch.full((20, 8, 3), float("nan")), 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"memory_quantile": 0.0},
        {"maximum_memory_anomaly_fraction": 1.1},
        {"teacher_score_beta": 0.0},
        {"teacher_score_scale_floor": 0.0},
        {"minimum_consensus_teachers": 1},
        {"consensus_recovery_fraction": 0.5},
        {"threshold_exposure_quantile": 1.0},
    ],
)
def test_new_niabd_configuration_ranges_are_validated(kwargs):
    with pytest.raises(ValueError):
        NIABDConfig(**kwargs)
