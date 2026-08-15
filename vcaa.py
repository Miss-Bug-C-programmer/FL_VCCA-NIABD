from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch

from admission import (
    AdmissionDecision,
    TeacherAdmissionRecord,
    TeacherKnowledge,
    TeacherMetadata,
)
from numeric_integrity import require_finite_tensor


VCAA_ALGORITHM_VERSION = "vcaa-v3-absolute-freshness-robust-content"
RESULT_SCHEMA_VERSION = "fedagg-results-v3"
_ROBUST_MAD_SCALE = 1.4826


@dataclass(frozen=True)
class VCAAConfig:
    """Server-side VCAA configuration.

    VCAA has two deliberately separate stages.  Lineage and age are a hard
    validity gate.  Content is a continuous reliability/aggregation weight;
    it is never allowed to make an invalid packet valid.  The old time decay
    fields remain accepted so old CLI/config files still load.  They map to
    ``age_half_life_s`` when the new field is omitted.
    """

    version_weight: float = 0.5
    time_decay_gamma: float = 0.99
    time_unit_s: float = 60.0
    max_version_lag: int = 1
    accuracy_weight: float = 0.5
    entropy_weight: float = 0.25
    divergence_weight: float = 0.25
    accuracy_scale: float = 1.0
    entropy_scale: Optional[float] = None
    divergence_scale: float = 1.0
    history_window_rounds: int = 5
    threshold_beta: float = 1.0
    warmup_rounds: int = 1
    epsilon: float = 1e-8
    nonfinite_policy: str = "fail_closed"
    max_knowledge_age_s: Optional[float] = None
    age_half_life_s: Optional[float] = None
    content_threshold_beta: Optional[float] = None
    consensus_divergence_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.version_weight) <= 1.0:
            raise ValueError("version_weight must be in [0, 1].")
        if not 0.0 < float(self.time_decay_gamma) < 1.0:
            raise ValueError("time_decay_gamma must be in (0, 1).")
        if float(self.time_unit_s) <= 0.0:
            raise ValueError("time_unit_s must be positive.")
        if int(self.max_version_lag) < 0:
            raise ValueError("max_version_lag must be non-negative.")
        weights = (
            float(self.accuracy_weight),
            float(self.entropy_weight),
            float(self.divergence_weight),
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("VCAA content weights must be non-negative.")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("VCAA content weights must sum to 1.")
        if float(self.accuracy_scale) <= 0.0:
            raise ValueError("accuracy_scale must be positive.")
        if self.entropy_scale is not None and float(self.entropy_scale) <= 0.0:
            raise ValueError("entropy_scale must be positive when provided.")
        if float(self.divergence_scale) <= 0.0:
            raise ValueError("divergence_scale must be positive.")
        if self.consensus_divergence_scale is not None and float(
            self.consensus_divergence_scale
        ) <= 0.0:
            raise ValueError("consensus_divergence_scale must be positive.")
        if int(self.history_window_rounds) <= 0:
            raise ValueError("history_window_rounds must be positive.")
        if float(self.threshold_beta) < 0.0:
            raise ValueError("threshold_beta must be non-negative.")
        if self.content_threshold_beta is not None and float(
            self.content_threshold_beta
        ) < 0.0:
            raise ValueError("content_threshold_beta must be non-negative.")
        if int(self.warmup_rounds) < 0:
            raise ValueError("warmup_rounds must be non-negative.")
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.nonfinite_policy not in {"fail_closed", "sanitize_and_record"}:
            raise ValueError(
                "nonfinite_policy must be 'fail_closed' or 'sanitize_and_record'."
            )
        half_life = self.age_half_life_s
        if half_life is None:
            half_life = -float(self.time_unit_s) * math.log(2.0) / math.log(
                float(self.time_decay_gamma)
            )
            object.__setattr__(self, "age_half_life_s", half_life)
        if not math.isfinite(float(half_life)) or float(half_life) <= 0.0:
            raise ValueError("age_half_life_s must be finite and positive.")
        max_age = self.max_knowledge_age_s
        if max_age is None:
            # Backward-compatible decay defaults to a conservative four
            # half-lives hard limit.  Formal runs can set this explicitly.
            max_age = 4.0 * float(half_life)
            object.__setattr__(self, "max_knowledge_age_s", max_age)
        if not math.isfinite(float(max_age)) or float(max_age) < 0.0:
            raise ValueError("max_knowledge_age_s must be finite and non-negative.")

    @property
    def effective_content_threshold_beta(self) -> float:
        return float(
            self.threshold_beta
            if self.content_threshold_beta is None
            else self.content_threshold_beta
        )

    @property
    def effective_divergence_scale(self) -> float:
        return float(
            self.divergence_scale
            if self.consensus_divergence_scale is None
            else self.consensus_divergence_scale
        )


class VersionContentAwareAdmission:
    """VCAA with an absolute lineage gate and robust soft content scoring."""

    name = "vcaa"
    algorithm_version = VCAA_ALGORITHM_VERSION
    result_schema_version = RESULT_SCHEMA_VERSION

    def __init__(
        self,
        config: Optional[VCAAConfig] = None,
        *,
        clock=time.monotonic,
    ) -> None:
        self.config = config or VCAAConfig()
        self._clock = clock
        self._history: Deque[Tuple[int, Tuple[float, ...]]] = deque(
            maxlen=int(self.config.history_window_rounds)
        )

    def reset(self) -> None:
        self._history.clear()

    def snapshot_state(self) -> dict:
        return {
            "history": [
                (int(round_number), tuple(float(score) for score in scores))
                for round_number, scores in self._history
            ],
            "algorithm_version": self.algorithm_version,
            "result_schema_version": self.result_schema_version,
        }

    def restore_state(self, state: dict) -> None:
        if str(state.get("algorithm_version")) != self.algorithm_version:
            raise ValueError("VCAA snapshot algorithm version mismatch.")
        history = state.get("history")
        if not isinstance(history, list):
            raise ValueError("VCAA snapshot history is invalid.")
        self._history.clear()
        for item in history:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("VCAA snapshot history entry is invalid.")
            round_number, scores = item
            if not isinstance(scores, (list, tuple)):
                raise ValueError("VCAA snapshot scores are invalid.")
            self._history.append(
                (int(round_number), tuple(float(score) for score in scores))
            )

    def _lineage_stats(
        self,
        metadata: Sequence[TeacherMetadata],
        *,
        current_round: int,
    ) -> List[Dict[str, float | bool | str]]:
        now = float(self._clock())
        results: List[Dict[str, float | bool | str]] = []
        for item in metadata:
            source_round = int(item.source_round)
            if source_round < 0:
                source_round = int(item.model_round)
            raw_lag = int(current_round) - source_round
            absolute_valid = raw_lag >= 0 and raw_lag <= int(
                self.config.max_version_lag
            )
            reason = ""
            if raw_lag < 0:
                reason = "future_source_round"
            elif raw_lag > int(self.config.max_version_lag):
                reason = "stale_version"

            generated = float(item.generated_at_s)
            received = float(item.received_at_s)
            consumed = float(item.consumed_at_s)
            generated_valid = math.isfinite(generated)
            timestamp_reason = ""
            transport_age = float("nan")
            queue_age = float("nan")
            knowledge_age = float("nan")
            age_valid = True
            # Direct unit callers from the legacy API often provide only a
            # generated marker unrelated to this process' monotonic clock.
            # Production packets always have consumed_at_s, so unknown legacy
            # age remains admissible while explicit lineage is fail-closed.
            if not generated_valid:
                age_valid = False
                timestamp_reason = "nonfinite_generated_timestamp"
            elif math.isfinite(consumed):
                if math.isfinite(received):
                    transport_age = received - generated
                    queue_age = consumed - received
                knowledge_age = consumed - generated
                if (
                    not math.isfinite(knowledge_age)
                    or knowledge_age < -float(self.config.epsilon)
                    or (
                        math.isfinite(transport_age)
                        and transport_age < -float(self.config.epsilon)
                    )
                    or (
                        math.isfinite(queue_age)
                        and queue_age < -float(self.config.epsilon)
                    )
                ):
                    age_valid = False
                    timestamp_reason = "invalid_timestamp_order"
            elif math.isfinite(received):
                # Required fallback when a consumer timestamp is unavailable.
                transport_age = received - generated
                knowledge_age = now - generated
                if (
                    transport_age < -float(self.config.epsilon)
                    or knowledge_age < -float(self.config.epsilon)
                ):
                    age_valid = False
                    timestamp_reason = "invalid_timestamp_order"
            if math.isfinite(knowledge_age):
                age_valid = age_valid and knowledge_age <= float(
                    self.config.max_knowledge_age_s
                ) + float(self.config.epsilon)
                if not age_valid and not timestamp_reason:
                    timestamp_reason = "expired_knowledge_age"
                age_score = math.exp(
                    -math.log(2.0)
                    * max(0.0, knowledge_age)
                    / float(self.config.age_half_life_s)
                )
            else:
                # Missing optional timestamps are neutral only for legacy
                # direct callers; server/process paths never take this branch.
                age_score = 1.0
            if not absolute_valid and reason == "":
                reason = "invalid_version_lineage"
            if not age_valid and timestamp_reason:
                reason = timestamp_reason
            results.append(
                {
                    "version_score": float(age_score if absolute_valid else 0.0),
                    "freshness_score": float(age_score if absolute_valid and age_valid else 0.0),
                    "age_seconds": float(knowledge_age),
                    "knowledge_age_s": float(knowledge_age),
                    "transport_age_s": float(transport_age),
                    "queue_age_s": float(queue_age),
                    "model_round": float(item.model_round),
                    "source_round": float(source_round),
                    "received_at_s": float(received),
                    "consumed_at_s": float(consumed),
                    "proxy_version": str(item.proxy_version),
                    "version_lag": float(raw_lag),
                    "raw_version_lag": float(raw_lag),
                    "minimum_accepted_round": float(
                        int(current_round) - int(self.config.max_version_lag)
                    ),
                    "absolute_version_valid": bool(absolute_valid),
                    "age_valid": bool(age_valid),
                    "hard_valid": bool(absolute_valid and age_valid),
                    "hard_rejection_reason": reason,
                }
            )
        return results

    @staticmethod
    def _safe_probabilities(logits: torch.Tensor, *, name: str) -> torch.Tensor:
        tensor = logits.detach().cpu().float()
        require_finite_tensor(tensor, phase="vcaa", metric=name)
        if tensor.ndim != 2:
            raise ValueError("VCAA expects two-dimensional classification logits.")
        return torch.softmax(tensor, dim=1)

    @torch.no_grad()
    def _content_statistics(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
        student_logits: torch.Tensor,
        proxy_labels: torch.Tensor,
    ) -> List[Dict[str, float]]:
        student = student_logits.detach().cpu().float()
        require_finite_tensor(student, phase="vcaa", metric="student_logits")
        if student.ndim != 2 or int(student.shape[0]) <= 0:
            raise ValueError("VCAA student logits must have shape [P,C].")
        labels = proxy_labels.detach().cpu().long().view(-1)
        if int(labels.numel()) != int(student.shape[0]):
            raise ValueError("VCAA proxy labels and student logits must have equal length.")
        teacher_probabilities = torch.stack(
            [self._safe_probabilities(item.logits, name="teacher_logits") for item in teacher_knowledge],
            dim=0,
        )
        if any(tuple(item.shape) != tuple(student.shape) for item in teacher_probabilities):
            raise ValueError("Teacher and student logits must share the same proxy shape.")
        consensus = torch.median(teacher_probabilities, dim=0).values
        consensus = consensus / consensus.sum(dim=1, keepdim=True).clamp_min(self.config.epsilon)
        log_consensus = consensus.clamp_min(self.config.epsilon).log()
        student_log_prob = torch.log_softmax(student, dim=1)
        student_prob = student_log_prob.exp()
        entropies = -(
            teacher_probabilities
            * teacher_probabilities.clamp_min(self.config.epsilon).log()
        ).sum(dim=2).mean(dim=1)
        entropy_center = torch.median(entropies)
        entropy_mad = torch.median((entropies - entropy_center).abs())
        entropy_scale = max(
            float(self.config.epsilon),
            float(_ROBUST_MAD_SCALE * entropy_mad.item()),
        )
        consensus_expand = consensus.unsqueeze(0)
        midpoint = 0.5 * (teacher_probabilities + consensus_expand)
        teacher_log = teacher_probabilities.clamp_min(self.config.epsilon).log()
        midpoint_log = midpoint.clamp_min(self.config.epsilon).log()
        teacher_kl = (teacher_probabilities * (teacher_log - midpoint_log)).sum(dim=2)
        consensus_kl = (consensus_expand * (log_consensus.unsqueeze(0) - midpoint_log)).sum(dim=2)
        js = 0.5 * (teacher_kl + consensus_kl).mean(dim=1)
        student_kl = (
            student_prob.unsqueeze(0)
            * (student_log_prob.unsqueeze(0) - teacher_log)
        ).sum(dim=2).clamp_min(0.0).mean(dim=1)
        predictions = teacher_probabilities.argmax(dim=2)
        accuracy = (predictions == labels.view(1, -1)).float().mean(dim=1)
        num_classes = int(student.shape[1])
        entropy_normalized = torch.exp(
            -((entropies - entropy_center).abs() / entropy_scale)
        )
        divergence_scale = float(self.config.effective_divergence_scale)
        divergence_term = torch.exp(-js / divergence_scale)
        results = []
        for index in range(len(teacher_knowledge)):
            results.append(
                {
                    "proxy_accuracy": float(accuracy[index].item()),
                    "mean_entropy": float(entropies[index].item()),
                    "entropy_deviation": float(
                        abs(entropies[index].item() - entropy_center.item())
                        / entropy_scale
                    ),
                    "mean_kl": float(student_kl[index].item()),
                    "consensus_divergence": float(js[index].item()),
                    "num_classes": float(num_classes),
                    "accuracy_term": float(
                        min(1.0, max(0.0, accuracy[index].item() / float(self.config.accuracy_scale)))
                    ),
                    "entropy_term": float(entropy_normalized[index].item()),
                    "divergence_term": float(divergence_term[index].item()),
                    "sanitized_value_count": 0.0,
                }
            )
        return results

    def _historical_threshold(self, current_round: int) -> float:
        if int(current_round) <= int(self.config.warmup_rounds):
            return 0.0
        values = [
            score
            for _, round_scores in self._history
            for score in round_scores
            if math.isfinite(float(score))
        ]
        if not values:
            return 0.0
        center = float(statistics.median(values))
        mad = float(statistics.median(abs(value - center) for value in values))
        return max(
            0.0,
            center - float(self.config.effective_content_threshold_beta) * _ROBUST_MAD_SCALE * mad,
        )

    def evaluate(
        self,
        *,
        teacher_knowledge: Sequence[TeacherKnowledge],
        student_logits: torch.Tensor,
        proxy_labels: torch.Tensor,
        current_round: int,
    ) -> AdmissionDecision:
        if not teacher_knowledge:
            raise ValueError("VCAA requires at least one teacher knowledge packet.")
        if int(current_round) <= 0:
            raise ValueError("current_round must be positive.")
        client_ids = [int(item.metadata.client_id) for item in teacher_knowledge]
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("VCAA client IDs must be unique.")

        lineage = self._lineage_stats(
            [item.metadata for item in teacher_knowledge],
            current_round=int(current_round),
        )
        content = self._content_statistics(
            teacher_knowledge,
            student_logits,
            proxy_labels,
        )
        threshold = self._historical_threshold(int(current_round))
        warmup = int(current_round) <= int(self.config.warmup_rounds)
        records = []
        valid_content_scores = []
        aggregation_weights: Dict[int, float] = {}
        for client_id, version, content_stats in zip(client_ids, lineage, content):
            content_score = (
                float(self.config.accuracy_weight) * content_stats["accuracy_term"]
                + float(self.config.entropy_weight) * content_stats["entropy_term"]
                + float(self.config.divergence_weight) * content_stats["divergence_term"]
            )
            freshness = float(version["freshness_score"])
            final_score = (
                float(self.config.version_weight) * freshness
                + (1.0 - float(self.config.version_weight)) * content_score
            )
            # The historical threshold is retained as a diagnostic and as a
            # soft reliability calibration.  It is intentionally not a hard
            # deletion rule: a fresh non-IID teacher still reaches NIABD.
            reliability = content_score
            if threshold > float(self.config.epsilon):
                reliability = min(1.0, content_score / threshold)
            reliability = max(0.0, min(1.0, reliability))
            hard_valid = bool(version["hard_valid"])
            admitted = bool(hard_valid and (warmup or reliability > float(self.config.epsilon)))
            reason = str(version["hard_rejection_reason"])
            if hard_valid and not admitted:
                reason = "zero_content_reliability"
            if hard_valid:
                valid_content_scores.append(float(content_score))
                aggregation_weights[client_id] = float(reliability)
            components = {
                **version,
                **content_stats,
                "content_score": float(content_score),
                "vcaa_content_reliability": float(reliability),
                "vcaa_aggregation_weight": float(reliability),
                "vcaa_hard_valid": float(hard_valid),
                "vcaa_absolute_version_valid": float(version["absolute_version_valid"]),
                "vcaa_age_valid": float(version["age_valid"]),
            }
            records.append(
                TeacherAdmissionRecord(
                    client_id=client_id,
                    admitted=admitted,
                    score=float(final_score if hard_valid else 0.0),
                    components=components,
                    hard_valid=hard_valid,
                    hard_rejection_reason=reason,
                    absolute_version_valid=bool(version["absolute_version_valid"]),
                    age_valid=bool(version["age_valid"]),
                    freshness_score=freshness,
                    content_reliability=float(reliability),
                    aggregation_weight=float(reliability),
                )
            )

        self._history.append((int(current_round), tuple(valid_content_scores)))
        admitted_ids = tuple(record.client_id for record in records if record.admitted)
        freshness_valid_ids = tuple(
            record.client_id for record in records if record.hard_valid
        )
        rejected_ids = tuple(client_id for client_id in client_ids if client_id not in set(admitted_ids))
        return AdmissionDecision(
            method=self.name,
            threshold=float(threshold),
            admitted_client_ids=admitted_ids,
            rejected_client_ids=rejected_ids,
            records=tuple(records),
            algorithm_version=self.algorithm_version,
            result_schema_version=self.result_schema_version,
            nonfinite_policy=self.config.nonfinite_policy,
            history_size=len(self._history),
            freshness_valid_client_ids=freshness_valid_ids,
            aggregation_weights=aggregation_weights,
        )
