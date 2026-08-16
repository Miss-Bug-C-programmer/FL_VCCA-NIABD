from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch

from admission import (
    AdmissionDecision,
    TeacherAdmissionRecord,
    TeacherKnowledge,
    TeacherMetadata,
)
from numeric_integrity import require_finite_tensor


VCAA_ALGORITHM_VERSION = "vcaa-v5-lineage-content-admission-runtime-age"
RESULT_SCHEMA_VERSION = "fedagg-results-v3"
_ROBUST_MAD_SCALE = 1.4826


@dataclass(frozen=True)
class VCAAConfig:
    """Configuration for the three-stage VCAA decision.

    Lineage, timestamp order, and age are a hard validity gate.  A historical
    robust lower bound is a content admission gate after calibration.  The
    content sigmoid remains a continuous reliability signal for admitted
    teachers.  Runtime-calibrated age uses only server-observable completion
    references and is bounded by explicit floors and ceilings.
    """

    version_weight: float = 0.5
    time_decay_gamma: float = 0.99
    time_unit_s: float = 60.0
    max_version_lag: int = 1
    version_lag_half_life_rounds: float = 1.0
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
    content_scale_floor: float = 0.05
    reliability_temperature: float = 1.0
    reliability_z_cap: float = 6.0
    minimum_content_cohort_size: int = 3
    minimum_content_history_size: int = 3
    age_scale_mode: str = "runtime-calibrated"
    runtime_age_reference_multiplier: float = 4.0
    runtime_age_half_life_floor_s: float = 0.5
    runtime_age_half_life_ceiling_s: float = 60.0
    runtime_max_age_multiplier: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.version_weight) <= 1.0:
            raise ValueError("version_weight must be in [0, 1].")
        if not 0.0 < float(self.time_decay_gamma) < 1.0:
            raise ValueError("time_decay_gamma must be in (0, 1).")
        if float(self.time_unit_s) <= 0.0:
            raise ValueError("time_unit_s must be positive.")
        if int(self.max_version_lag) < 0:
            raise ValueError("max_version_lag must be non-negative.")
        if not math.isfinite(float(self.version_lag_half_life_rounds)) or float(
            self.version_lag_half_life_rounds
        ) <= 0.0:
            raise ValueError("version_lag_half_life_rounds must be positive.")
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
        for name, value in (
            ("content_scale_floor", self.content_scale_floor),
            ("reliability_temperature", self.reliability_temperature),
            ("reliability_z_cap", self.reliability_z_cap),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if int(self.minimum_content_cohort_size) < 1:
            raise ValueError("minimum_content_cohort_size must be at least 1.")
        if int(self.minimum_content_history_size) < 1:
            raise ValueError("minimum_content_history_size must be at least 1.")
        if self.age_scale_mode not in {"fixed", "runtime-calibrated"}:
            raise ValueError(
                "age_scale_mode must be 'fixed' or 'runtime-calibrated'."
            )
        for name, value in (
            ("runtime_age_reference_multiplier", self.runtime_age_reference_multiplier),
            ("runtime_age_half_life_floor_s", self.runtime_age_half_life_floor_s),
            ("runtime_age_half_life_ceiling_s", self.runtime_age_half_life_ceiling_s),
            ("runtime_max_age_multiplier", self.runtime_max_age_multiplier),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if float(self.runtime_age_half_life_ceiling_s) < float(
            self.runtime_age_half_life_floor_s
        ):
            raise ValueError(
                "runtime_age_half_life_ceiling_s must not be below its floor."
            )
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.nonfinite_policy not in {"fail_closed", "sanitize_and_record"}:
            raise ValueError(
                "nonfinite_policy must be 'fail_closed' or 'sanitize_and_record'."
            )
        half_life_was_explicit = self.age_half_life_s is not None
        half_life = self.age_half_life_s
        if half_life is None:
            half_life = -float(self.time_unit_s) * math.log(2.0) / math.log(
                float(self.time_decay_gamma)
            )
            object.__setattr__(self, "age_half_life_s", half_life)
        if not math.isfinite(float(half_life)) or float(half_life) <= 0.0:
            raise ValueError("age_half_life_s must be finite and positive.")
        max_age_was_explicit = self.max_knowledge_age_s is not None
        max_age = self.max_knowledge_age_s
        if max_age is None:
            max_age = 4.0 * float(half_life)
            object.__setattr__(self, "max_knowledge_age_s", max_age)
        if not math.isfinite(float(max_age)) or float(max_age) < 0.0:
            raise ValueError("max_knowledge_age_s must be finite and non-negative.")
        object.__setattr__(self, "_age_half_life_was_explicit", half_life_was_explicit)
        object.__setattr__(self, "_max_age_was_explicit", max_age_was_explicit)

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
    """VCAA with hard lineage, calibrated content admission, and soft weights."""

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
        self._effective_age_half_life_s = float(self.config.age_half_life_s)
        self._effective_max_knowledge_age_s = float(self.config.max_knowledge_age_s)
        self._runtime_reference_s = float("nan")
        self._runtime_reference_history: Deque[float] = deque(
            maxlen=int(self.config.history_window_rounds)
        )

    @property
    def effective_age_half_life_s(self) -> float:
        return float(self._effective_age_half_life_s)

    @property
    def effective_max_knowledge_age_s(self) -> float:
        return float(self._effective_max_knowledge_age_s)

    def reset(self) -> None:
        self._history.clear()
        self._runtime_reference_history.clear()
        self._runtime_reference_s = float("nan")
        self._effective_age_half_life_s = float(self.config.age_half_life_s)
        self._effective_max_knowledge_age_s = float(self.config.max_knowledge_age_s)

    def update_runtime_timing(self, reference_time_s: Optional[float]) -> None:
        """Calibrate age from bounded, server-observable completion timing."""

        if self.config.age_scale_mode == "fixed":
            self._effective_age_half_life_s = float(self.config.age_half_life_s)
            self._effective_max_knowledge_age_s = float(
                self.config.max_knowledge_age_s
            )
            return
        if reference_time_s is None or not math.isfinite(float(reference_time_s)):
            return
        reference = max(float(reference_time_s), float(self.config.epsilon))
        self._runtime_reference_history.append(reference)
        robust_reference = float(statistics.median(self._runtime_reference_history))
        half_life = robust_reference * float(
            self.config.runtime_age_reference_multiplier
        )
        half_life = min(
            float(self.config.runtime_age_half_life_ceiling_s),
            max(float(self.config.runtime_age_half_life_floor_s), half_life),
        )
        self._runtime_reference_s = robust_reference
        self._effective_age_half_life_s = half_life
        if bool(getattr(self.config, "_max_age_was_explicit", False)):
            self._effective_max_knowledge_age_s = float(
                self.config.max_knowledge_age_s
            )
        else:
            self._effective_max_knowledge_age_s = half_life * float(
                self.config.runtime_max_age_multiplier
            )

    def snapshot_state(self) -> dict:
        return {
            "history": [
                (int(round_number), tuple(float(score) for score in scores))
                for round_number, scores in self._history
            ],
            "algorithm_version": self.algorithm_version,
            "result_schema_version": self.result_schema_version,
            "effective_age_half_life_s": float(self._effective_age_half_life_s),
            "effective_max_knowledge_age_s": float(
                self._effective_max_knowledge_age_s
            ),
            "runtime_reference_s": float(self._runtime_reference_s),
            "runtime_reference_history": [
                float(value) for value in self._runtime_reference_history
            ],
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
            if any(not math.isfinite(float(score)) for score in scores):
                raise ValueError("VCAA snapshot scores must be finite.")
            self._history.append(
                (int(round_number), tuple(float(score) for score in scores))
            )
        for key in ("effective_age_half_life_s", "effective_max_knowledge_age_s"):
            if not math.isfinite(float(state.get(key, float("nan")))):
                raise ValueError(f"VCAA snapshot {key} must be finite.")
        reference_history = state.get("runtime_reference_history", [])
        if not isinstance(reference_history, list):
            raise ValueError("VCAA snapshot runtime reference history is invalid.")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in reference_history
        ):
            raise ValueError("VCAA snapshot runtime references must be positive.")
        self._runtime_reference_history.clear()
        self._runtime_reference_history.extend(float(value) for value in reference_history)
        self._effective_age_half_life_s = float(state["effective_age_half_life_s"])
        self._effective_max_knowledge_age_s = float(
            state["effective_max_knowledge_age_s"]
        )
        self._runtime_reference_s = float(
            state.get("runtime_reference_s", float("nan"))
        )
        if not math.isfinite(self._runtime_reference_s):
            self._runtime_reference_s = float("nan")

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
            timestamp_valid = True
            timestamp_reason = ""
            transport_age = float("nan")
            queue_age = float("nan")
            knowledge_age = float("nan")
            if not math.isfinite(generated):
                timestamp_valid = False
                timestamp_reason = "nonfinite_generated_timestamp"
            elif math.isfinite(consumed):
                knowledge_age = consumed - generated
                if math.isfinite(received):
                    transport_age = received - generated
                    queue_age = consumed - received
                if (
                    not math.isfinite(knowledge_age)
                    or knowledge_age < -float(self.config.epsilon)
                    or (
                        math.isfinite(received)
                        and received < generated - float(self.config.epsilon)
                    )
                    or (
                        math.isfinite(queue_age)
                        and queue_age < -float(self.config.epsilon)
                    )
                ):
                    timestamp_valid = False
                    timestamp_reason = "invalid_timestamp_order"
            else:
                if math.isfinite(received):
                    transport_age = received - generated
                knowledge_age = now - generated
                if (
                    not math.isfinite(knowledge_age)
                    or knowledge_age < -float(self.config.epsilon)
                    or (
                        math.isfinite(transport_age)
                        and transport_age < -float(self.config.epsilon)
                    )
                ):
                    timestamp_valid = False
                    timestamp_reason = "invalid_timestamp_order"

            age_valid = bool(timestamp_valid)
            if math.isfinite(knowledge_age):
                if knowledge_age > float(self._effective_max_knowledge_age_s) + float(
                    self.config.epsilon
                ):
                    age_valid = False
                    if not timestamp_reason:
                        timestamp_reason = "expired_knowledge_age"
                age_score = math.exp(
                    -math.log(2.0)
                    * max(0.0, knowledge_age)
                    / float(self._effective_age_half_life_s)
                )
            else:
                age_score = 1.0
            if not absolute_valid and not reason:
                reason = "invalid_version_lineage"
            if not age_valid and timestamp_reason:
                reason = timestamp_reason
            version_lag_score = (
                math.exp(
                    -math.log(2.0)
                    * max(0, raw_lag)
                    / float(self.config.version_lag_half_life_rounds)
                )
                if raw_lag >= 0
                else 0.0
            )
            freshness_score = (
                version_lag_score * age_score
                if absolute_valid and age_valid
                else 0.0
            )
            results.append(
                {
                    "version_score": float(
                        version_lag_score if absolute_valid else 0.0
                    ),
                    "version_lag_score": float(
                        version_lag_score if absolute_valid else 0.0
                    ),
                    "age_score": float(age_score if age_valid else 0.0),
                    "freshness_score": float(freshness_score),
                    "age_seconds": float(knowledge_age),
                    "knowledge_age_s": float(knowledge_age),
                    "transport_age_s": float(transport_age),
                    "queue_age_s": float(queue_age),
                    "model_round": float(item.model_round),
                    "source_round": float(source_round),
                    "base_server_round": float(item.base_server_round),
                    "generated_at_s": float(generated),
                    "received_at_s": float(received),
                    "consumed_at_s": float(consumed),
                    "proxy_version": str(item.proxy_version),
                    "version_lag": float(raw_lag),
                    "raw_version_lag": float(raw_lag),
                    "minimum_accepted_round": float(
                        int(current_round) - int(self.config.max_version_lag)
                    ),
                    "timestamp_valid": bool(timestamp_valid),
                    "absolute_version_valid": bool(absolute_valid),
                    "age_valid": bool(age_valid),
                    "hard_valid": bool(absolute_valid and age_valid),
                    "hard_rejection_reason": reason,
                    "effective_age_half_life_s": float(
                        self._effective_age_half_life_s
                    ),
                    "effective_max_knowledge_age_s": float(
                        self._effective_max_knowledge_age_s
                    ),
                    "age_scale_mode": str(self.config.age_scale_mode),
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
        consensus = consensus / consensus.sum(dim=1, keepdim=True).clamp_min(
            self.config.epsilon
        )
        log_consensus = consensus.clamp_min(self.config.epsilon).log()
        student_log_prob = torch.log_softmax(student, dim=1)
        student_prob = student_log_prob.exp()
        entropies = -(
            teacher_probabilities
            * teacher_probabilities.clamp_min(self.config.epsilon).log()
        ).sum(dim=2).mean(dim=1)
        entropy_center = torch.median(entropies)
        entropy_mad = torch.median((entropies - entropy_center).abs())
        entropy_scale = (
            float(self.config.entropy_scale)
            if self.config.entropy_scale is not None
            else max(
                float(self.config.epsilon),
                float(_ROBUST_MAD_SCALE * entropy_mad.item()),
            )
        )
        consensus_expand = consensus.unsqueeze(0)
        midpoint = 0.5 * (teacher_probabilities + consensus_expand)
        teacher_log = teacher_probabilities.clamp_min(self.config.epsilon).log()
        midpoint_log = midpoint.clamp_min(self.config.epsilon).log()
        teacher_kl = (teacher_probabilities * (teacher_log - midpoint_log)).sum(dim=2)
        consensus_kl = (
            consensus_expand
            * (log_consensus.unsqueeze(0) - midpoint_log)
        ).sum(dim=2)
        js = 0.5 * (teacher_kl + consensus_kl).mean(dim=1)
        student_kl = (
            student_prob.unsqueeze(0)
            * (student_log_prob.unsqueeze(0) - teacher_log)
        ).sum(dim=2).clamp_min(0.0).mean(dim=1)
        predictions = teacher_probabilities.argmax(dim=2)
        accuracy = (predictions == labels.view(1, -1)).float().mean(dim=1)
        entropy_normalized = torch.exp(
            -((entropies - entropy_center).abs() / entropy_scale)
        )
        divergence_term = torch.exp(-js / float(self.config.effective_divergence_scale))
        num_classes = int(student.shape[1])
        return [
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
                    min(
                        1.0,
                        max(
                            0.0,
                            accuracy[index].item()
                            / float(self.config.accuracy_scale),
                        ),
                    )
                ),
                "entropy_term": float(entropy_normalized[index].item()),
                "divergence_term": float(divergence_term[index].item()),
                "sanitized_value_count": 0.0,
            }
            for index in range(len(teacher_knowledge))
        ]

    @staticmethod
    def _content_score(content_stats: Dict[str, float], config: VCAAConfig) -> float:
        score = (
            float(config.accuracy_weight) * content_stats["accuracy_term"]
            + float(config.entropy_weight) * content_stats["entropy_term"]
            + float(config.divergence_weight) * content_stats["divergence_term"]
        )
        if not math.isfinite(float(score)):
            raise ValueError("VCAA content score must be finite.")
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _robust_scale(values: Sequence[float]) -> float:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            return 0.0
        center = float(statistics.median(finite))
        mad = float(statistics.median(abs(value - center) for value in finite))
        return float(_ROBUST_MAD_SCALE * mad)

    def _calibration_statistics(
        self,
        content_scores: Sequence[float],
    ) -> Tuple[float, float]:
        if not content_scores:
            return float("nan"), float("nan")
        center = float(statistics.median(content_scores))
        current_scale = self._robust_scale(content_scores)
        historical_scales = [
            self._robust_scale(round_scores) for _, round_scores in self._history
        ]
        finite_candidates = [
            value
            for value in [current_scale, *historical_scales]
            if math.isfinite(float(value))
        ]
        scale = max(
            float(self.config.content_scale_floor),
            float(statistics.median(finite_candidates))
            if finite_candidates
            else 0.0,
        )
        return center, scale

    def _content_reliability(
        self,
        *,
        score: float,
        center: float,
        scale: float,
        warmup: bool,
        cohort_size: int,
    ) -> Tuple[float, float, str]:
        if not math.isfinite(float(score)):
            raise ValueError("VCAA content score must be finite.")
        if not math.isfinite(float(center)) or not math.isfinite(float(scale)):
            raise ValueError("VCAA content calibration must be finite.")
        z = (float(score) - float(center)) / float(scale)
        z = max(
            -float(self.config.reliability_z_cap),
            min(float(self.config.reliability_z_cap), z),
        )
        if warmup:
            return 1.0, z, "warmup_uniform"
        if cohort_size < int(self.config.minimum_content_cohort_size):
            return 1.0, z, "small_cohort_uniform"
        reliability = 1.0 / (
            1.0 + math.exp(-z / float(self.config.reliability_temperature))
        )
        if not 0.0 < reliability < 1.0:
            raise ValueError("VCAA content reliability must lie in (0, 1).")
        return float(reliability), float(z), "robust_relative_sigmoid"

    def _historical_threshold(
        self,
        current_round: int,
    ) -> Tuple[float, bool, str, int]:
        if int(current_round) <= int(self.config.warmup_rounds):
            return float("nan"), False, "warmup_disabled", 0
        values = [
            score
            for _, round_scores in self._history
            for score in round_scores
            if math.isfinite(float(score))
        ]
        if len(values) < int(self.config.minimum_content_history_size):
            return float("nan"), False, "insufficient_history", len(values)
        center = float(statistics.median(values))
        scale = max(
            float(self.config.content_scale_floor),
            float(_ROBUST_MAD_SCALE)
            * float(statistics.median(abs(value - center) for value in values)),
        )
        threshold = max(
            0.0,
            center - float(self.config.effective_content_threshold_beta) * scale,
        )
        return (
            float(threshold),
            True,
            "historical_median_minus_mad_floor",
            len(values),
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
        hard_valid_indices = [
            index for index, item in enumerate(lineage) if bool(item["hard_valid"])
        ]
        valid_knowledge = [teacher_knowledge[index] for index in hard_valid_indices]
        valid_content = (
            self._content_statistics(valid_knowledge, student_logits, proxy_labels)
            if valid_knowledge
            else []
        )
        content_by_index = {
            index: content_stats
            for index, content_stats in zip(hard_valid_indices, valid_content)
        }
        content_scores = [
            self._content_score(content_stats, self.config)
            for content_stats in valid_content
        ]
        content_center, content_scale = self._calibration_statistics(content_scores)
        threshold, gate_active, threshold_source, history_observations = (
            self._historical_threshold(int(current_round))
        )
        warmup = int(current_round) <= int(self.config.warmup_rounds)
        empty_content = {
            "proxy_accuracy": float("nan"),
            "mean_entropy": float("nan"),
            "entropy_deviation": float("nan"),
            "mean_kl": float("nan"),
            "consensus_divergence": float("nan"),
            "num_classes": float("nan"),
            "accuracy_term": float("nan"),
            "entropy_term": float("nan"),
            "divergence_term": float("nan"),
            "sanitized_value_count": float("nan"),
        }
        records: List[TeacherAdmissionRecord] = []
        raw_weights: Dict[int, float] = {}
        calibration_by_client: Dict[int, Tuple[float, float, float, str]] = {}
        for index, (client_id, version) in enumerate(zip(client_ids, lineage)):
            hard_valid = bool(version["hard_valid"])
            content_stats = content_by_index.get(index, empty_content)
            content_score = (
                self._content_score(content_stats, self.config)
                if hard_valid
                else float("nan")
            )
            if hard_valid:
                reliability, score_z, weighting_mode = self._content_reliability(
                    score=content_score,
                    center=content_center,
                    scale=content_scale,
                    warmup=warmup,
                    cohort_size=len(valid_content),
                )
                if gate_active:
                    content_valid = content_score >= float(threshold) - float(
                        self.config.epsilon
                    )
                    content_rejection_reason = (
                        "" if content_valid else "below_historical_threshold"
                    )
                else:
                    content_valid = True
                    content_rejection_reason = f"gate_inactive_{threshold_source}"
                freshness = float(version["freshness_score"])
                raw_weight = (
                    freshness
                    * (
                        float(self.config.version_weight)
                        + (1.0 - float(self.config.version_weight)) * reliability
                    )
                    if content_valid
                    else 0.0
                )
                raw_weight = max(0.0, float(raw_weight))
                calibration_by_client[client_id] = (
                    float(content_score),
                    float(score_z),
                    float(reliability),
                    str(weighting_mode),
                )
            else:
                reliability = float("nan")
                score_z = float("nan")
                weighting_mode = "hard_invalid"
                content_valid = False
                content_rejection_reason = "hard_invalid"
                raw_weight = 0.0
            admitted = bool(hard_valid and content_valid)
            if admitted:
                raw_weights[client_id] = max(float(self.config.epsilon), raw_weight)
            hard_reason = str(version["hard_rejection_reason"])
            rejection_reason = "" if admitted else (hard_reason or content_rejection_reason)
            record_center = float(content_center) if hard_valid else float("nan")
            record_scale = float(content_scale) if hard_valid else float("nan")
            record_threshold = float(threshold) if hard_valid and gate_active else float("nan")
            components = {
                **version,
                **content_stats,
                "content_score": float(content_score),
                "content_threshold": record_threshold,
                "content_valid": bool(content_valid),
                "content_gate_active": bool(gate_active),
                "content_rejection_reason": str(content_rejection_reason),
                "content_threshold_source": str(threshold_source),
                "content_history_observations": int(history_observations),
                "vcaa_content_reliability": float(reliability),
                "vcaa_aggregation_weight": float(raw_weight),
                "normalized_aggregation_weight": 0.0,
                "effective_weight_ratio_to_uniform": 0.0,
                "content_score_center": record_center,
                "content_score_scale": record_scale,
                "content_score_z": float(score_z),
                "weighting_mode": str(weighting_mode),
                "content_threshold_role": "admission_gate",
                "vcaa_threshold_used_for_weighting": bool(gate_active),
                "vcaa_final_score_used_for_weighting": bool(admitted),
                "vcaa_hard_valid": float(hard_valid),
                "vcaa_absolute_version_valid": float(version["absolute_version_valid"]),
                "vcaa_age_valid": float(version["age_valid"]),
                "rejection_reason": str(rejection_reason),
            }
            records.append(
                TeacherAdmissionRecord(
                    client_id=client_id,
                    admitted=admitted,
                    score=float(raw_weight if admitted else 0.0),
                    components=components,
                    hard_valid=hard_valid,
                    hard_rejection_reason=hard_reason,
                    absolute_version_valid=bool(version["absolute_version_valid"]),
                    age_valid=bool(version["age_valid"]),
                    timestamp_valid=bool(version["timestamp_valid"]),
                    version_lag_score=float(version["version_lag_score"]),
                    age_score=float(version["age_score"]),
                    freshness_score=float(version["freshness_score"]),
                    content_valid=content_valid,
                    content_gate_active=gate_active,
                    content_rejection_reason=str(content_rejection_reason),
                    rejection_reason=str(rejection_reason),
                    content_reliability=float(reliability),
                    aggregation_weight=float(raw_weight),
                    content_score_center=record_center,
                    content_score_scale=record_scale,
                    content_score_z=float(score_z),
                    weighting_mode=str(weighting_mode),
                )
            )

        history_scores = [
            float(record.components["content_score"])
            for record in records
            if record.hard_valid and record.content_valid
        ]
        if history_scores:
            self._history.append((int(current_round), tuple(history_scores)))
        total_weight = sum(raw_weights.values())
        normalized_weights: Dict[int, float] = {}
        if raw_weights:
            if total_weight <= float(self.config.epsilon):
                raise ValueError("VCAA admitted teacher weights must have positive sum.")
            normalized_weights = {
                client_id: float(weight / total_weight)
                for client_id, weight in raw_weights.items()
            }
        effective_teacher_count = (
            float(total_weight * total_weight)
            / float(sum(weight * weight for weight in raw_weights.values()))
            if raw_weights
            else float("nan")
        )
        mean_weight = (
            float(total_weight) / float(len(raw_weights)) if raw_weights else float("nan")
        )
        weight_cv = (
            math.sqrt(
                sum((weight - mean_weight) ** 2 for weight in raw_weights.values())
                / float(len(raw_weights))
            )
            / mean_weight
            if raw_weights and mean_weight > 0.0
            else float("nan")
        )
        total_variation = (
            0.5
            * sum(
                abs(weight - (1.0 / len(raw_weights)))
                for weight in normalized_weights.values()
            )
            if raw_weights
            else float("nan")
        )
        saturation_fraction = (
            sum(
                1
                for _, _, reliability, _ in calibration_by_client.values()
                if reliability >= 0.999
            )
            / float(len(calibration_by_client))
            if calibration_by_client
            else float("nan")
        )
        for record_index, record in enumerate(records):
            if record.client_id in normalized_weights:
                normalized = float(normalized_weights[record.client_id])
                ratio = normalized * len(raw_weights)
                record.components["normalized_aggregation_weight"] = normalized
                record.components["effective_weight_ratio_to_uniform"] = ratio
                records[record_index] = replace(
                    record,
                    normalized_aggregation_weight=normalized,
                    effective_weight_ratio_to_uniform=ratio,
                )
            else:
                records[record_index] = replace(
                    record,
                    normalized_aggregation_weight=0.0,
                    effective_weight_ratio_to_uniform=0.0,
                )
        admitted_ids = tuple(record.client_id for record in records if record.admitted)
        freshness_valid_ids = tuple(record.client_id for record in records if record.hard_valid)
        rejected_ids = tuple(
            client_id for client_id in client_ids if client_id not in set(admitted_ids)
        )
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
            aggregation_weights=raw_weights,
            normalized_aggregation_weights=normalized_weights,
            effective_teacher_count=effective_teacher_count,
            weight_cv=weight_cv,
            weight_total_variation_from_uniform=total_variation,
            content_reliability_saturation_fraction=saturation_fraction,
            content_score_center=float(content_center),
            content_score_scale=float(content_scale),
            content_threshold_role="admission_gate",
            content_gate_active=bool(gate_active),
            content_threshold_source=str(threshold_source),
            content_history_observations=int(history_observations),
            vcaa_threshold_used_for_weighting=bool(gate_active),
            effective_age_half_life_s=float(self._effective_age_half_life_s),
            effective_max_knowledge_age_s=float(self._effective_max_knowledge_age_s),
            age_scale_mode=str(self.config.age_scale_mode),
        )
