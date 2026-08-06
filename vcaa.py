from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch

from numeric_integrity import require_finite_tensor

from admission import (
    AdmissionDecision,
    TeacherAdmissionRecord,
    TeacherKnowledge,
    TeacherMetadata,
)


VCAA_ALGORITHM_VERSION = "vcaa-v2.1-transactional-time-aware"
RESULT_SCHEMA_VERSION = "fedagg-results-v3"


@dataclass(frozen=True)
class VCAAConfig:
    """Configuration for version-content-aware teacher admission."""

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

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.version_weight) <= 1.0:
            raise ValueError("version_weight must be in [0, 1].")
        if not 0.0 < float(self.time_decay_gamma) < 1.0:
            raise ValueError("time_decay_gamma must be in (0, 1).")
        if float(self.time_unit_s) <= 0.0:
            raise ValueError("time_unit_s must be positive.")
        if int(self.max_version_lag) < 0:
            raise ValueError("max_version_lag must be non-negative.")
        content_weights = (
            float(self.accuracy_weight),
            float(self.entropy_weight),
            float(self.divergence_weight),
        )
        if any(weight < 0.0 for weight in content_weights):
            raise ValueError("VCAA content weights must be non-negative.")
        if not math.isclose(sum(content_weights), 1.0, abs_tol=1e-6):
            raise ValueError("VCAA content weights must sum to 1.")
        if float(self.accuracy_scale) <= 0.0:
            raise ValueError("accuracy_scale must be positive.")
        if self.entropy_scale is not None and float(self.entropy_scale) <= 0.0:
            raise ValueError("entropy_scale must be positive when provided.")
        if float(self.divergence_scale) <= 0.0:
            raise ValueError("divergence_scale must be positive.")
        if int(self.history_window_rounds) <= 0:
            raise ValueError("history_window_rounds must be positive.")
        if float(self.threshold_beta) < 1.0:
            raise ValueError("threshold_beta must be at least 1.")
        if int(self.warmup_rounds) < 0:
            raise ValueError("warmup_rounds must be non-negative.")
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.nonfinite_policy not in {"fail_closed", "sanitize_and_record"}:
            raise ValueError(
                "nonfinite_policy must be 'fail_closed' or "
                "'sanitize_and_record'."
            )


class VersionContentAwareAdmission:
    """VCAA implementation following Eqs. (1)--(7) of the reference design."""

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
        """Return explicit controller state for round transactions/checkpoints."""

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

    def _version_scores(
        self,
        metadata: Sequence[TeacherMetadata],
        *,
        current_round: int,
    ) -> List[Dict[str, float]]:
        versions = [
            int(item.source_round)
            if int(item.source_round) >= 0
            else int(item.model_round)
            for item in metadata
        ]
        median_version = float(statistics.median(versions))
        minimum_round = median_version - int(self.config.max_version_lag)
        now = float(self._clock())
        results = []
        for item in metadata:
            received_at_s = float(item.received_at_s)
            if math.isfinite(received_at_s):
                age_seconds = max(0.0, now - received_at_s)
            else:
                # Direct unit callers from the legacy interface may not carry
                # server receive metadata. Production packet paths always do;
                # retain the legacy generated timestamp only for compatibility.
                generated_at_s = float(item.generated_at_s)
                age_seconds = (
                    max(0.0, now - generated_at_s)
                    if math.isfinite(generated_at_s)
                    else float("nan")
                )
            source_round = (
                int(item.source_round)
                if int(item.source_round) >= 0
                else int(item.model_round)
            )
            version_lag = max(0, int(current_round) - source_round)
            age_units = age_seconds / float(self.config.time_unit_s)
            version_floor = float(int(source_round) >= minimum_round)
            if not math.isfinite(age_seconds):
                if self.config.nonfinite_policy == "fail_closed":
                    raise ValueError(
                        "VCAA received_at_s is nonfinite; fail_closed policy "
                        "rejects unverifiable time semantics."
                    )
                age_seconds = 0.0
                age_units = 0.0
            score = (
                float(self.config.time_decay_gamma) ** age_units
            ) * version_floor
            results.append(
                {
                    "version_score": float(score),
                    "age_seconds": float(age_seconds),
                    "model_round": float(item.model_round),
                    "source_round": float(source_round),
                    "version_lag": float(version_lag),
                    "minimum_accepted_round": float(minimum_round),
                }
            )
        return results

    @torch.no_grad()
    def _content_statistics(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
        student_logits: torch.Tensor,
        proxy_labels: torch.Tensor,
    ) -> List[Dict[str, float]]:
        eps = float(self.config.epsilon)
        student_logits = student_logits.detach().cpu().float()
        nonfinite_student = int((~torch.isfinite(student_logits)).sum().item())
        if nonfinite_student:
            if self.config.nonfinite_policy == "fail_closed":
                raise ValueError("VCAA student logits are nonfinite.")
            student_logits = torch.nan_to_num(
                student_logits, nan=0.0, posinf=30.0, neginf=-30.0
            ).clamp(-30.0, 30.0)
        require_finite_tensor(
            student_logits,
            phase="vcaa",
            metric="student_logits",
        )
        labels = proxy_labels.detach().cpu().long().view(-1)
        if student_logits.ndim != 2:
            raise ValueError(
                "VCAA expects two-dimensional classification logits."
            )
        sample_count, num_classes = student_logits.shape
        if sample_count <= 0:
            raise ValueError("VCAA proxy knowledge contains no samples.")
        if int(labels.numel()) != int(sample_count):
            raise ValueError(
                "VCAA proxy labels and student logits must have equal length."
            )
        if num_classes <= 1:
            raise ValueError("VCAA requires at least two output classes.")

        student_log_prob = torch.log_softmax(student_logits, dim=1)
        student_prob = student_log_prob.exp()
        results = []
        for knowledge in teacher_knowledge:
            teacher_logits = knowledge.logits.detach().cpu().float()
            nonfinite_teacher = int(
                (~torch.isfinite(teacher_logits)).sum().item()
            )
            if nonfinite_teacher:
                if self.config.nonfinite_policy == "fail_closed":
                    raise ValueError("VCAA teacher logits are nonfinite.")
                teacher_logits = torch.nan_to_num(
                    teacher_logits, nan=0.0, posinf=30.0, neginf=-30.0
                ).clamp(-30.0, 30.0)
            require_finite_tensor(
                teacher_logits,
                phase="vcaa",
                metric="teacher_logits",
            )
            if teacher_logits.shape != student_logits.shape:
                raise ValueError(
                    "Teacher and student logits must share the same proxy shape."
                )
            teacher_log_prob = torch.log_softmax(teacher_logits, dim=1)
            teacher_prob = teacher_log_prob.exp()
            correct = int(
                (teacher_prob.argmax(dim=1) == labels).sum().item()
            )
            entropy_sum = float(
                (
                    -teacher_prob
                    * torch.log(teacher_prob.clamp_min(eps))
                )
                .sum(dim=1)
                .sum()
                .item()
            )
            kl_sum = float(
                (
                    student_prob
                    * (student_log_prob - teacher_log_prob)
                )
                .sum(dim=1)
                .clamp_min(0.0)
                .sum()
                .item()
            )
            results.append(
                {
                    "proxy_accuracy": float(correct) / sample_count,
                    "mean_entropy": float(entropy_sum) / sample_count,
                    "mean_kl": float(kl_sum) / sample_count,
                    "num_classes": float(num_classes),
                    "sanitized_value_count": float(
                        nonfinite_student + nonfinite_teacher
                    ),
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
        ]
        if not values:
            return 0.0
        mean_score = float(statistics.fmean(values))
        std_score = (
            float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        )
        return mean_score - float(self.config.threshold_beta) * std_score

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
        metadata = [item.metadata for item in teacher_knowledge]
        client_ids = [int(item.client_id) for item in metadata]
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("VCAA client IDs must be unique.")
        if int(current_round) <= 0:
            raise ValueError("current_round must be positive.")

        version_stats = self._version_scores(
            metadata,
            current_round=int(current_round),
        )
        content_stats = self._content_statistics(
            teacher_knowledge,
            student_logits,
            proxy_labels,
        )

        scores = []
        components = []
        for version, content in zip(version_stats, content_stats):
            entropy_scale = self.config.entropy_scale
            if entropy_scale is None:
                entropy_scale = math.log(
                    max(2, int(content["num_classes"]))
                )
            accuracy_term = min(
                1.0,
                max(
                    0.0,
                    content["proxy_accuracy"]
                    / float(self.config.accuracy_scale),
                ),
            )
            entropy_term = math.exp(
                -content["mean_entropy"] / float(entropy_scale)
            )
            divergence_term = math.exp(
                -content["mean_kl"]
                / float(self.config.divergence_scale)
            )
            content_score = (
                float(self.config.accuracy_weight) * accuracy_term
                + float(self.config.entropy_weight) * entropy_term
                + float(self.config.divergence_weight) * divergence_term
            )
            score = (
                float(self.config.version_weight) * version["version_score"]
                + (1.0 - float(self.config.version_weight)) * content_score
            )
            scores.append(float(score))
            components.append(
                {
                    **version,
                    **content,
                    "accuracy_term": float(accuracy_term),
                    "entropy_term": float(entropy_term),
                    "divergence_term": float(divergence_term),
                    "content_score": float(content_score),
                }
            )

        threshold = self._historical_threshold(int(current_round))
        warmup = int(current_round) <= int(self.config.warmup_rounds)
        admitted_ids = [
            client_id
            for client_id, score in zip(client_ids, scores)
            if warmup or score >= threshold
        ]
        admitted_set = set(admitted_ids)
        rejected_ids = [
            client_id for client_id in client_ids if client_id not in admitted_set
        ]

        records = tuple(
            TeacherAdmissionRecord(
                client_id=client_id,
                admitted=client_id in admitted_set,
                score=score,
                components=component,
            )
            for client_id, score, component in zip(
                client_ids,
                scores,
                components,
            )
        )
        admitted_scores = tuple(
            record.score for record in records if record.admitted
        )
        # Record every finite score, not only survivors.  This prevents a
        # low-score survivor-bias spiral in which all future teachers are
        # rejected against an obsolete high-score history.
        if self.config.nonfinite_policy == "fail_closed" and not all(
            math.isfinite(float(score)) for score in scores
        ):
            raise ValueError("VCAA produced a nonfinite admission score.")
        self._history.append((int(current_round), tuple(scores)))

        return AdmissionDecision(
            method=self.name,
            threshold=float(threshold),
            admitted_client_ids=tuple(admitted_ids),
            rejected_client_ids=tuple(rejected_ids),
            records=records,
            algorithm_version=self.algorithm_version,
            result_schema_version=self.result_schema_version,
            nonfinite_policy=self.config.nonfinite_policy,
            history_size=len(self._history),
        )
