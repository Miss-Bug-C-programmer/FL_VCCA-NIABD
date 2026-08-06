from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from admission import TeacherKnowledge
from defense import DefenseResult, TeacherDefenseRecord
from numeric_integrity import require_finite_tensor


NIABD_ALGORITHM_VERSION = (
    "niabd-v2.1-proxy-conditioned-transactional-robust-memory"
)
RESULT_SCHEMA_VERSION = "fedagg-results-v3"
_ROBUST_MAD_SCALE = 1.4826


@dataclass(frozen=True)
class NIABDConfig:
    """Configuration for prediction-only NIABD robust memory.

    ``benign_deviation_limit`` is a high-quantile safety bound for teacher
    history/consensus deviations, not a maximum over every proxy/class
    element.  The persistent memory has shape ``[P, C]`` and thresholds have
    shape ``[C]``.  The implementation follows the proxy-conditioned
    prototype, continuous inhibition, EMA, and STDP-like threshold equations
    in Section 5 of the reference design.  A round reads the old state,
    purifies the current logits, and only then updates memory and thresholds.
    The safety assumption is that a sufficiently large, compact consensus is
    not controlled by an adversarial majority; when that cannot be established
    the safe action is to freeze memory.  Per round work is ``O(K*P*C)`` and
    persistent state is ``O(P*C + C)``.
    """

    initial_threshold: float = 2.0
    minimum_threshold: float = 0.5
    maximum_threshold: float = 6.0
    transition_smoothness: float = 1.0
    prototype_learning_rate: float = 0.01
    threshold_learning_rate: float = 0.01
    potentiation_balance: float = 0.5
    threshold_decay: float = 0.01
    benign_deviation_limit: float = 4.0
    warmup_rounds: int = 1
    minimum_standard_deviation: float = 0.1
    reference_source: str = "prototype"
    epsilon: float = 1e-6
    memory_quantile: float = 0.95
    maximum_memory_anomaly_fraction: float = 0.10
    teacher_score_beta: float = 3.0
    teacher_score_scale_floor: float = 1e-3
    minimum_consensus_teachers: int = 4
    consensus_recovery_fraction: float = 0.75
    threshold_exposure_quantile: float = 0.75
    proxy_chunk_size: int = 0

    def __post_init__(self) -> None:
        if float(self.initial_threshold) <= 0.0:
            raise ValueError("initial_threshold must be positive.")
        if float(self.minimum_threshold) <= 0.0:
            raise ValueError("minimum_threshold must be positive.")
        if float(self.maximum_threshold) < float(self.minimum_threshold):
            raise ValueError(
                "maximum_threshold must not be below minimum_threshold."
            )
        if not (
            float(self.minimum_threshold)
            <= float(self.initial_threshold)
            <= float(self.maximum_threshold)
        ):
            raise ValueError(
                "initial_threshold must lie within the threshold bounds."
            )
        if float(self.transition_smoothness) <= 0.0:
            raise ValueError("transition_smoothness must be positive.")
        if not 0.0 < float(self.prototype_learning_rate) <= 1.0:
            raise ValueError("prototype_learning_rate must be in (0, 1].")
        if not 0.0 < float(self.threshold_learning_rate) <= 1.0:
            raise ValueError("threshold_learning_rate must be in (0, 1].")
        if not 0.0 < float(self.potentiation_balance) < 1.0:
            raise ValueError("potentiation_balance must be in (0, 1).")
        if float(self.threshold_decay) <= 0.0:
            raise ValueError("threshold_decay must be positive.")
        if float(self.benign_deviation_limit) <= 0.0:
            raise ValueError("benign_deviation_limit must be positive.")
        if int(self.warmup_rounds) < 1:
            raise ValueError("warmup_rounds must be at least 1.")
        if float(self.minimum_standard_deviation) <= 0.0:
            raise ValueError(
                "minimum_standard_deviation must be positive."
            )
        if self.reference_source not in {"prototype", "student"}:
            raise ValueError(
                "reference_source must be 'prototype' or 'student'."
            )
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")
        if not 0.0 < float(self.memory_quantile) < 1.0:
            raise ValueError("memory_quantile must be in (0, 1).")
        if not 0.0 <= float(self.maximum_memory_anomaly_fraction) <= 1.0:
            raise ValueError(
                "maximum_memory_anomaly_fraction must be in [0, 1]."
            )
        if float(self.teacher_score_beta) <= 0.0:
            raise ValueError("teacher_score_beta must be positive.")
        if float(self.teacher_score_scale_floor) <= 0.0:
            raise ValueError(
                "teacher_score_scale_floor must be positive."
            )
        if int(self.minimum_consensus_teachers) < 2:
            raise ValueError("minimum_consensus_teachers must be at least 2.")
        if not 0.5 < float(self.consensus_recovery_fraction) <= 1.0:
            raise ValueError(
                "consensus_recovery_fraction must be in (0.5, 1]."
            )
        if int(self.proxy_chunk_size) < 0:
            raise ValueError("proxy_chunk_size must be non-negative.")
        if not 0.0 < float(self.threshold_exposure_quantile) < 1.0:
            raise ValueError(
                "threshold_exposure_quantile must be in (0, 1)."
            )


class NeuroInspiredAdaptiveBackdoorDefense:
    """Prediction-only NIABD with proxy-conditioned robust memory.

    Input teacher tensors and the student tensor have shape ``[P, C]``;
    ``proxy_labels`` is intentionally accepted for interface compatibility but
    its values are never read.  Persistent state is ``mu, variance: [P, C]``
    and ``thresholds: [C]``.  Historical deviations and purification use the
    pre-update state, then safe raw teacher logits update memory and thresholds.
    The implementation is shared by synchronous and process-semi-async
    runtimes and never uses attack truth or triggered-test data.
    """

    name = "niabd"
    algorithm_version = NIABD_ALGORITHM_VERSION
    result_schema_version = RESULT_SCHEMA_VERSION

    def __init__(self, config: Optional[NIABDConfig] = None) -> None:
        self.config = config or NIABDConfig()
        self._prototype_mean: Optional[torch.Tensor] = None
        self._prototype_variance: Optional[torch.Tensor] = None
        self._thresholds: Optional[torch.Tensor] = None
        self._observation_count = 0
        self._eligible_teacher_observations = 0
        self._memory_update_rounds = 0
        self._consecutive_frozen_rounds = 0
        self._proxy_shape: Optional[Tuple[int, int]] = None
        self._proxy_version = ""

    def reset(self) -> None:
        """Explicitly reset run-bound memory and all associated counters."""

        self._prototype_mean = None
        self._prototype_variance = None
        self._thresholds = None
        self._observation_count = 0
        self._eligible_teacher_observations = 0
        self._memory_update_rounds = 0
        self._consecutive_frozen_rounds = 0
        self._proxy_shape = None
        self._proxy_version = ""

    def snapshot_state(self) -> dict:
        """Return explicit run-bound state for transaction/checkpoint use."""

        return {
            "prototype_mean": (
                None
                if self._prototype_mean is None
                else self._prototype_mean.clone()
            ),
            "prototype_variance": (
                None
                if self._prototype_variance is None
                else self._prototype_variance.clone()
            ),
            "thresholds": (
                None if self._thresholds is None else self._thresholds.clone()
            ),
            "observation_count": int(self._observation_count),
            "eligible_teacher_observations": int(
                self._eligible_teacher_observations
            ),
            "memory_update_rounds": int(self._memory_update_rounds),
            "consecutive_frozen_rounds": int(self._consecutive_frozen_rounds),
            "proxy_shape": self._proxy_shape,
            "proxy_version": self._proxy_version,
            "algorithm_version": self.algorithm_version,
        }

    def restore_state(self, state: dict) -> None:
        if str(state.get("algorithm_version")) != self.algorithm_version:
            raise ValueError("NIABD snapshot algorithm version mismatch.")
        proxy_shape = state.get("proxy_shape")
        self._proxy_shape = (
            None if proxy_shape is None else tuple(int(item) for item in proxy_shape)
        )
        self._proxy_version = str(state.get("proxy_version", ""))
        self._prototype_mean = (
            None
            if state.get("prototype_mean") is None
            else state["prototype_mean"].clone()
        )
        self._prototype_variance = (
            None
            if state.get("prototype_variance") is None
            else state["prototype_variance"].clone()
        )
        self._thresholds = (
            None if state.get("thresholds") is None else state["thresholds"].clone()
        )
        self._observation_count = int(state.get("observation_count", 0))
        self._eligible_teacher_observations = int(
            state.get("eligible_teacher_observations", 0)
        )
        self._memory_update_rounds = int(state.get("memory_update_rounds", 0))
        self._consecutive_frozen_rounds = int(
            state.get("consecutive_frozen_rounds", 0)
        )

    @property
    def is_initialized(self) -> bool:
        return self._prototype_mean is not None

    @property
    def prototype_mean(self) -> Optional[torch.Tensor]:
        if self._prototype_mean is None:
            return None
        return self._prototype_mean.clone()

    @property
    def prototype_variance(self) -> Optional[torch.Tensor]:
        if self._prototype_variance is None:
            return None
        return self._prototype_variance.clone()

    @property
    def thresholds(self) -> Optional[torch.Tensor]:
        if self._thresholds is None:
            return None
        return self._thresholds.clone()

    @property
    def observation_count(self) -> int:
        return int(self._observation_count)

    @staticmethod
    def _to_cpu_float(logits: torch.Tensor, *, name: str) -> torch.Tensor:
        if not torch.is_tensor(logits):
            raise TypeError(f"NIABD {name} must be a torch.Tensor.")
        if logits.ndim != 2:
            raise ValueError(
                f"NIABD {name} must have shape [samples, classes]."
            )
        if not torch.is_floating_point(logits):
            raise TypeError(f"NIABD {name} must use a floating dtype.")
        require_finite_tensor(
            logits,
            phase="niabd",
            metric=name,
        )
        result = logits.detach().to(device="cpu", dtype=torch.float32)
        require_finite_tensor(
            result,
            phase="niabd",
            metric=f"{name}_float32",
        )
        return result

    def _stack_knowledge(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
    ) -> Tuple[torch.Tensor, Tuple[torch.dtype, ...]]:
        if not teacher_knowledge:
            raise ValueError("NIABD requires at least one admitted teacher.")
        tensors = []
        dtypes = []
        for index, item in enumerate(teacher_knowledge):
            tensor = self._to_cpu_float(
                item.logits,
                name=f"teacher_logits[{index}]",
            )
            tensors.append(tensor)
            dtypes.append(item.logits.dtype)
        reference_shape = tensors[0].shape
        if int(reference_shape[0]) <= 0:
            raise ValueError("NIABD requires P > 0 proxy samples.")
        if int(reference_shape[1]) <= 1:
            raise ValueError("NIABD requires C > 1 output classes.")
        if any(tensor.shape != reference_shape for tensor in tensors):
            raise ValueError(
                "NIABD teacher logits must have equal proxy shapes; "
                "P and C cannot change after memory initialization."
            )
        proxy_versions = {
            str(item.metadata.proxy_version)
            for item in teacher_knowledge
            if str(item.metadata.proxy_version)
        }
        if len(proxy_versions) > 1:
            raise ValueError("NIABD teacher packets have mixed proxy versions.")
        incoming_proxy_version = next(iter(proxy_versions), "")
        if self._proxy_version and incoming_proxy_version != self._proxy_version:
            raise RuntimeError(
                "NIABD proxy_version changed for an initialized run; call reset()."
            )
        if self._proxy_version == "" and incoming_proxy_version:
            self._proxy_version = incoming_proxy_version
        return torch.stack(tensors, dim=0), tuple(dtypes)

    def _validate_memory_shape(self, shape: torch.Size) -> None:
        if not self.is_initialized:
            return
        assert self._prototype_mean is not None
        assert self._prototype_variance is not None
        assert self._thresholds is not None
        expected = (int(shape[0]), int(shape[1]))
        actual = tuple(int(value) for value in self._prototype_mean.shape)
        if actual != expected or tuple(self._prototype_variance.shape) != expected:
            raise RuntimeError(
                "NIABD memory shape is fixed for this run: "
                f"memory={actual}, current={expected}; call reset() explicitly."
            )
        if self._proxy_shape is not None and tuple(self._proxy_shape) != expected:
            raise RuntimeError(
                "NIABD proxy identity shape changed; call reset() explicitly."
            )
        if tuple(int(value) for value in self._thresholds.shape) != (expected[1],):
            raise RuntimeError(
                "NIABD threshold shape is inconsistent with memory; call reset()."
            )

    def _robust_center_variance(
        self,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return teacher-axis median/MAD center and variance with shape [P,C]."""

        center = torch.median(values, dim=0).values
        mad = torch.median(
            (values - center.unsqueeze(0)).abs(),
            dim=0,
        ).values
        minimum_std = float(self.config.minimum_standard_deviation)
        scale = torch.maximum(
            mad * _ROBUST_MAD_SCALE,
            torch.full_like(mad, minimum_std),
        )
        variance = scale.square()
        require_finite_tensor(
            center,
            phase="niabd",
            metric="robust_center",
        )
        require_finite_tensor(
            variance,
            phase="niabd",
            metric="robust_variance",
        )
        return center, variance

    def _initialize_memory(self, stacked_logits: torch.Tensor) -> None:
        center, variance = self._robust_center_variance(stacked_logits)
        self._prototype_mean = center.clone()
        self._prototype_variance = variance.clone()
        self._thresholds = torch.full(
            (int(stacked_logits.shape[-1]),),
            float(self.config.initial_threshold),
            dtype=torch.float32,
        )
        self._observation_count = int(stacked_logits.shape[0] * stacked_logits.shape[1])
        self._eligible_teacher_observations = self._observation_count
        self._memory_update_rounds = 1
        self._proxy_shape = (
            int(stacked_logits.shape[1]),
            int(stacked_logits.shape[2]),
        )
        require_finite_tensor(
            self._thresholds,
            phase="niabd",
            metric="initial_thresholds",
        )

    def _update_memory(
        self,
        stacked_logits: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> float:
        """Apply robust-center EMA and variance EMA to selected raw teachers."""

        assert self._prototype_mean is not None
        assert self._prototype_variance is not None
        selected_count = int(update_mask.sum().item())
        if selected_count <= 0:
            return 0.0
        selected = stacked_logits[update_mask]
        center, batch_variance = self._robust_center_variance(selected)
        eta = float(self.config.prototype_learning_rate)
        previous_mean = self._prototype_mean
        new_mean = (1.0 - eta) * previous_mean + eta * center
        new_variance = (
            (1.0 - eta)
            * (
                self._prototype_variance
                + (previous_mean - new_mean).square()
            )
            + eta
            * (
                batch_variance
                + (center - new_mean).square()
            )
        )
        minimum_variance = float(self.config.minimum_standard_deviation) ** 2
        new_variance = new_variance.clamp_min(minimum_variance)
        require_finite_tensor(
            new_mean,
            phase="niabd",
            metric="prototype_mean_update",
        )
        require_finite_tensor(
            new_variance,
            phase="niabd",
            metric="prototype_variance_update",
        )
        self._prototype_mean = new_mean
        self._prototype_variance = new_variance
        self._observation_count += selected_count * int(stacked_logits.shape[1])
        self._eligible_teacher_observations += selected_count * int(stacked_logits.shape[1])
        self._memory_update_rounds += 1
        return eta

    @staticmethod
    def _upper_robust_z(values: torch.Tensor, floor: float) -> torch.Tensor:
        center = torch.median(values)
        mad = torch.median((values - center).abs())
        q75 = torch.quantile(values, 0.75)
        q25 = torch.quantile(values, 0.25)
        iqr_scale = (q75 - q25) / 1.349
        scale = torch.maximum(
            torch.maximum(
                1.4826 * mad,
                iqr_scale,
            ),
            torch.tensor(float(floor), dtype=values.dtype),
        )
        return torch.relu((values - center) / scale)

    def _teacher_metrics(
        self,
        abs_deviation: torch.Tensor,
        consensus_deviation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        assert self._thresholds is not None
        thresholds = self._thresholds.view(1, 1, -1)
        chunk_size = int(self.config.proxy_chunk_size)
        if chunk_size <= 0:
            chunk_size = int(abs_deviation.shape[0])
        anomaly_parts = []
        high_parts = []
        excess_parts = []
        consensus_parts = []
        # Concatenate scalar summaries before robust scoring. This keeps the
        # result exactly equivalent to the unchunked path while bounding the
        # temporary teacher-axis metric tensors.
        for start in range(0, int(abs_deviation.shape[0]), chunk_size):
            stop = min(start + chunk_size, int(abs_deviation.shape[0]))
            abs_part = abs_deviation[start:stop]
            consensus_part = consensus_deviation[start:stop]
            anomaly_parts.append(
                (abs_part > thresholds).float().mean(dim=(1, 2))
            )
            high_parts.append(
                torch.quantile(
                    abs_part.reshape(abs_part.shape[0], -1),
                    float(self.config.memory_quantile),
                    dim=1,
                )
            )
            excess_parts.append(
                torch.relu(abs_part - thresholds).mean(dim=(1, 2))
            )
            consensus_parts.append(
                torch.quantile(
                    consensus_part.reshape(consensus_part.shape[0], -1),
                    float(self.config.memory_quantile),
                    dim=1,
                )
            )
        anomaly_fraction = torch.cat(anomaly_parts, dim=0)
        high_quantile = torch.cat(high_parts, dim=0)
        mean_excess = torch.cat(excess_parts, dim=0)
        consensus_quantile = torch.cat(consensus_parts, dim=0)
        score = torch.stack(
            [
                self._upper_robust_z(
                    anomaly_fraction,
                    float(self.config.teacher_score_scale_floor),
                ),
                self._upper_robust_z(
                    high_quantile,
                    float(self.config.teacher_score_scale_floor),
                ),
                self._upper_robust_z(
                    mean_excess,
                    float(self.config.teacher_score_scale_floor),
                ),
                self._upper_robust_z(
                    consensus_quantile,
                    float(self.config.teacher_score_scale_floor),
                ),
            ],
            dim=0,
        ).amax(dim=0)
        eligible = (
            (score <= float(self.config.teacher_score_beta))
            & (
                high_quantile
                <= float(self.config.benign_deviation_limit)
            )
            & (
                anomaly_fraction
                <= float(self.config.maximum_memory_anomaly_fraction)
            )
            & (
                consensus_quantile
                <= float(self.config.benign_deviation_limit)
            )
        )
        return {
            "anomaly_fraction": anomaly_fraction,
            "high_quantile_deviation": high_quantile,
            "mean_excess": mean_excess,
            "consensus_deviation": consensus_quantile,
            "teacher_memory_score": score,
            "eligible": eligible,
        }

    def _update_thresholds(
        self,
        abs_deviation: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> None:
        """Apply Eq. (13) using only normal memory-eligible teachers."""

        assert self._thresholds is not None
        if not bool(eligible_mask.any().item()):
            return
        exposure = torch.quantile(
            abs_deviation[eligible_mask].reshape(-1, abs_deviation.shape[-1]),
            float(self.config.threshold_exposure_quantile),
            dim=0,
        )
        thresholds = self._thresholds
        above = exposure > thresholds
        potentiation = (
            float(self.config.potentiation_balance)
            * torch.relu(exposure - thresholds)
        )
        depression = (
            (1.0 - float(self.config.potentiation_balance))
            * float(self.config.threshold_decay)
        )
        delta = torch.where(
            above,
            potentiation,
            torch.full_like(thresholds, -depression),
        )
        updated = (
            thresholds
            + float(self.config.threshold_learning_rate) * delta
        ).clamp(
            min=float(self.config.minimum_threshold),
            max=float(self.config.maximum_threshold),
        )
        require_finite_tensor(
            updated,
            phase="niabd",
            metric="threshold_update",
        )
        self._thresholds = updated

    def _records(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
        *,
        anomaly_fraction: torch.Tensor,
        mean_abs_deviation: torch.Tensor,
        max_abs_deviation: torch.Tensor,
        mean_suppression: torch.Tensor,
        memory_eligible: torch.Tensor,
        high_quantile: torch.Tensor,
        mean_excess: torch.Tensor,
        consensus_deviation: torch.Tensor,
        teacher_memory_score: torch.Tensor,
    ) -> Tuple[TeacherDefenseRecord, ...]:
        records: List[TeacherDefenseRecord] = []
        for index, item in enumerate(teacher_knowledge):
            records.append(
                TeacherDefenseRecord(
                    client_id=int(item.metadata.client_id),
                    anomaly_fraction=float(anomaly_fraction[index].item()),
                    mean_abs_deviation=float(mean_abs_deviation[index].item()),
                    max_abs_deviation=float(max_abs_deviation[index].item()),
                    mean_suppression=float(mean_suppression[index].item()),
                    memory_eligible=bool(memory_eligible[index].item()),
                    teacher_memory_score=float(teacher_memory_score[index].item()),
                    high_quantile_deviation=float(high_quantile[index].item()),
                    mean_excess=float(mean_excess[index].item()),
                    consensus_deviation=float(consensus_deviation[index].item()),
                )
            )
        return tuple(records)

    def _metrics(
        self,
        *,
        warmup: float,
        prototype_updated: bool,
        reason: str,
        candidate_teachers: int,
        memory_eligible_teachers: int,
        anomaly_fraction: float,
        mean_suppression: float,
        teacher_metrics: Optional[Dict[str, torch.Tensor]],
        current_consensus_drift: float,
        effective_memory_weight: float,
    ) -> Dict[str, object]:
        assert self._thresholds is not None
        if teacher_metrics is None:
            score_mean = score_median = score_mad = float("nan")
            high_quantile = mean_excess = consensus = float("nan")
        else:
            scores = teacher_metrics["teacher_memory_score"]
            score_mean = float(scores.mean().item())
            score_median = float(torch.median(scores).item())
            score_mad = float(
                torch.median((scores - torch.median(scores)).abs()).item()
            )
            high_quantile = float(
                teacher_metrics["high_quantile_deviation"].mean().item()
            )
            mean_excess = float(
                teacher_metrics["mean_excess"].mean().item()
            )
            consensus = float(
                teacher_metrics["consensus_deviation"].mean().item()
            )
        return {
            "warmup": float(warmup),
            "prototype_updated": float(prototype_updated),
            "prototype_observations": float(self._observation_count),
            "threshold_mean": float(self._thresholds.mean().item()),
            "threshold_min": float(self._thresholds.min().item()),
            "threshold_max": float(self._thresholds.max().item()),
            "anomaly_fraction": float(anomaly_fraction),
            "mean_suppression": float(mean_suppression),
            "memory_eligible_teachers": int(memory_eligible_teachers),
            "niabd_algorithm_version": self.algorithm_version,
            "result_schema_version": self.result_schema_version,
            "niabd_prototype_update_reason": reason,
            "niabd_memory_candidate_teachers": int(candidate_teachers),
            "niabd_teacher_score_mean": score_mean,
            "niabd_teacher_score_median": score_median,
            "niabd_teacher_score_mad": score_mad,
            "niabd_high_quantile_deviation": high_quantile,
            "niabd_mean_excess": mean_excess,
            "niabd_consensus_deviation": consensus,
            "niabd_current_consensus_drift": float(current_consensus_drift),
            "niabd_all_ineligible_round": float(
                memory_eligible_teachers == 0
            ),
            "niabd_consecutive_frozen_rounds": int(
                self._consecutive_frozen_rounds
            ),
            "niabd_effective_memory_weight": float(effective_memory_weight),
            "niabd_eligible_teacher_observations": int(
                self._eligible_teacher_observations
            ),
            "niabd_memory_update_rounds": int(self._memory_update_rounds),
            "niabd_defense_available": True,
            "niabd_purification_applied": True,
            "niabd_memory_updated": bool(prototype_updated),
        }

    def _warmup_consensus_gate(
        self,
        stacked: torch.Tensor,
    ) -> Tuple[torch.Tensor, str]:
        """Choose a deterministic compact majority for safe initialization."""

        teacher_count = int(stacked.shape[0])
        required = max(
            int(self.config.minimum_consensus_teachers),
            int(
                ceil(
                    float(self.config.consensus_recovery_fraction)
                    * teacher_count
                )
            ),
        )
        if teacher_count < int(self.config.minimum_consensus_teachers):
            return torch.zeros(teacher_count, dtype=torch.bool), (
                "freeze_insufficient_teachers"
            )
        center = torch.median(stacked, dim=0).values
        mad = torch.median(
            (stacked - center.unsqueeze(0)).abs(),
            dim=0,
        ).values
        scale = torch.maximum(
            mad * _ROBUST_MAD_SCALE,
            torch.full_like(
                mad,
                float(self.config.minimum_standard_deviation),
            ),
        )
        normalized = (stacked - center.unsqueeze(0)).abs() / (
            scale.unsqueeze(0) + float(self.config.epsilon)
        )
        teacher_distance = torch.quantile(
            normalized.reshape(teacher_count, -1),
            float(self.config.memory_quantile),
            dim=1,
        )
        candidate_mask = teacher_distance <= float(
            self.config.benign_deviation_limit
        )
        candidate_count = int(candidate_mask.sum().item())
        if candidate_count < required:
            return candidate_mask, "freeze_no_safe_candidate"
        candidate_values = teacher_distance[candidate_mask]
        compact = float(torch.quantile(candidate_values, 0.75).item()) <= float(
            self.config.benign_deviation_limit
        )
        if not compact:
            return candidate_mask, "freeze_no_safe_candidate"

        # A large deterministic gap with two near-equal sides indicates a
        # symmetric or near-symmetric split.  A compact majority plus a small
        # offset minority does not meet this ambiguity condition.
        sorted_distance = torch.sort(teacher_distance).values
        gaps = sorted_distance[1:] - sorted_distance[:-1]
        positive_gaps = gaps[gaps > float(self.config.teacher_score_scale_floor)]
        median_gap = (
            float(torch.median(positive_gaps).item())
            if positive_gaps.numel()
            else 0.0
        )
        max_gap_index = int(torch.argmax(gaps).item()) if gaps.numel() else -1
        ambiguous = False
        if max_gap_index >= 0:
            left = max_gap_index + 1
            right = teacher_count - left
            lower_cluster = max(2, int(ceil(0.35 * teacher_count)))
            gap_floor = max(
                float(self.config.benign_deviation_limit),
                3.0 * max(median_gap, float(self.config.teacher_score_scale_floor)),
            )
            ambiguous = (
                left >= lower_cluster
                and right >= lower_cluster
                and float(gaps[max_gap_index].item()) > gap_floor
            )
        if ambiguous:
            return candidate_mask, "freeze_ambiguous_consensus"
        return candidate_mask, "warmup_robust_update"

    def _warmup_result(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
        candidate_mask: torch.Tensor,
        *,
        reason: str,
    ) -> DefenseResult:
        candidate_count = int(candidate_mask.sum().item())
        records = tuple(
            TeacherDefenseRecord(
                client_id=int(item.metadata.client_id),
                anomaly_fraction=float("nan"),
                mean_abs_deviation=float("nan"),
                max_abs_deviation=float("nan"),
                mean_suppression=float("nan"),
                memory_eligible=bool(candidate_mask[index].item()),
            )
            for index, item in enumerate(teacher_knowledge)
        )
        if self.is_initialized:
            metrics = self._metrics(
                warmup=1.0,
                prototype_updated=bool(candidate_count),
                reason=reason,
                candidate_teachers=candidate_count,
                memory_eligible_teachers=candidate_count,
                anomaly_fraction=float("nan"),
                mean_suppression=float("nan"),
                teacher_metrics=None,
                current_consensus_drift=0.0,
                effective_memory_weight=(
                    1.0 if candidate_count else 0.0
                ),
            )
        else:
            metrics = {
                "warmup": 1.0,
                "prototype_updated": float(bool(candidate_count)),
                "prototype_observations": float(self._observation_count),
                "threshold_mean": float("nan"),
                "threshold_min": float("nan"),
                "threshold_max": float("nan"),
                "anomaly_fraction": float("nan"),
                "mean_suppression": float("nan"),
                "memory_eligible_teachers": candidate_count,
                "niabd_algorithm_version": self.algorithm_version,
                "result_schema_version": self.result_schema_version,
                "niabd_prototype_update_reason": reason,
                "niabd_memory_candidate_teachers": candidate_count,
                "niabd_teacher_score_mean": float("nan"),
                "niabd_teacher_score_median": float("nan"),
                "niabd_teacher_score_mad": float("nan"),
                "niabd_high_quantile_deviation": float("nan"),
                "niabd_mean_excess": float("nan"),
                "niabd_consensus_deviation": float("nan"),
                "niabd_current_consensus_drift": 0.0,
                "niabd_all_ineligible_round": float(candidate_count == 0),
                "niabd_consecutive_frozen_rounds": int(
                    self._consecutive_frozen_rounds
                ),
                "niabd_effective_memory_weight": 0.0,
                "niabd_eligible_teacher_observations": int(
                    self._eligible_teacher_observations
                ),
                "niabd_memory_update_rounds": int(self._memory_update_rounds),
            }
        metrics.update({
            "niabd_defense_available": bool(self.is_initialized),
            "niabd_purification_applied": False,
            "niabd_memory_updated": bool(
                candidate_count and reason == "warmup_robust_update"
            ),
        })
        return DefenseResult(
            method=self.name,
            purified_knowledge=tuple(teacher_knowledge),
            records=records,
            metrics=metrics,
        )

    def _uninitialized_result(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
        *,
        reason: str,
    ) -> DefenseResult:
        records = tuple(
            TeacherDefenseRecord(
                client_id=int(item.metadata.client_id),
                anomaly_fraction=0.0,
                mean_abs_deviation=0.0,
                max_abs_deviation=0.0,
                mean_suppression=0.0,
                memory_eligible=False,
            )
            for item in teacher_knowledge
        )
        return DefenseResult(
            method=self.name,
            purified_knowledge=tuple(teacher_knowledge),
            records=records,
            metrics={
                "warmup": 1.0,
                "prototype_updated": 0.0,
                "prototype_observations": 0.0,
                "threshold_mean": float("nan"),
                "threshold_min": float("nan"),
                "threshold_max": float("nan"),
                "anomaly_fraction": float("nan"),
                "mean_suppression": float("nan"),
                "memory_eligible_teachers": 0,
                "niabd_algorithm_version": self.algorithm_version,
                "result_schema_version": self.result_schema_version,
                "niabd_prototype_update_reason": reason,
                "niabd_memory_candidate_teachers": len(teacher_knowledge),
                "niabd_teacher_score_mean": float("nan"),
                "niabd_teacher_score_median": float("nan"),
                "niabd_teacher_score_mad": float("nan"),
                "niabd_high_quantile_deviation": float("nan"),
                "niabd_mean_excess": float("nan"),
                "niabd_consensus_deviation": float("nan"),
                "niabd_current_consensus_drift": 0.0,
                "niabd_all_ineligible_round": 1.0,
                "niabd_consecutive_frozen_rounds": int(
                    self._consecutive_frozen_rounds
                ),
                "niabd_effective_memory_weight": 0.0,
                "niabd_eligible_teacher_observations": int(
                    self._eligible_teacher_observations
                ),
                "niabd_memory_update_rounds": int(self._memory_update_rounds),
                "niabd_defense_available": False,
                "niabd_purification_applied": False,
                "niabd_memory_updated": False,
            },
        )

    def purify(
        self,
        *,
        teacher_knowledge: Sequence[TeacherKnowledge],
        student_logits: torch.Tensor,
        proxy_labels: torch.Tensor,
        current_round: int,
    ) -> DefenseResult:
        """Purify one admitted teacher batch using the pre-update memory.

        ``teacher_knowledge`` and ``student_logits`` are ``[P,C]`` per teacher
        and student.  ``proxy_labels`` is deliberately not inspected.  The
        returned knowledge preserves each item's metadata and original floating
        dtype.  Memory shape is run-bound and any change raises until ``reset``
        is called explicitly.  Complexity is ``O(K*P*C)`` with ``O(P*C)``
        persistent state.  All derived tensors are checked for finiteness.
        """

        del proxy_labels
        if int(current_round) <= 0:
            raise ValueError("current_round must be positive.")
        stacked, original_dtypes = self._stack_knowledge(teacher_knowledge)
        student = self._to_cpu_float(student_logits, name="student_logits")
        if student.shape != stacked.shape[1:]:
            raise ValueError(
                "NIABD student and teacher logits must share a proxy shape."
            )
        self._validate_memory_shape(student.shape)

        initialized_now = False
        warmup_candidates, warmup_reason = self._warmup_consensus_gate(stacked)
        if not self.is_initialized:
            if warmup_reason != "warmup_robust_update":
                self._consecutive_frozen_rounds += 1
                return self._warmup_result(
                    teacher_knowledge,
                    warmup_candidates,
                    reason=warmup_reason,
                )
            self._initialize_memory(stacked[warmup_candidates])
            initialized_now = True

        assert self._prototype_mean is not None
        assert self._prototype_variance is not None
        assert self._thresholds is not None

        if initialized_now or int(current_round) <= int(self.config.warmup_rounds):
            if not initialized_now:
                if warmup_reason == "warmup_robust_update":
                    self._update_memory(stacked, warmup_candidates)
                else:
                    self._consecutive_frozen_rounds += 1
                    return self._warmup_result(
                        teacher_knowledge,
                        warmup_candidates,
                        reason=warmup_reason,
                    )
            self._consecutive_frozen_rounds = 0
            return self._warmup_result(
                teacher_knowledge,
                warmup_candidates,
                reason="warmup_robust_update",
            )

        previous_mean = self._prototype_mean
        previous_variance = self._prototype_variance
        previous_thresholds = self._thresholds
        standard_deviation = torch.sqrt(previous_variance)
        deviation = (
            stacked - previous_mean.unsqueeze(0)
        ) / (
            standard_deviation.unsqueeze(0)
            + float(self.config.epsilon)
        )
        abs_deviation = deviation.abs()
        current_consensus = torch.median(stacked, dim=0).values
        current_mad = torch.median(
            (stacked - current_consensus.unsqueeze(0)).abs(),
            dim=0,
        ).values
        current_scale = torch.maximum(
            _ROBUST_MAD_SCALE * current_mad,
            torch.full_like(
                current_mad,
                float(self.config.minimum_standard_deviation),
            ),
        )
        consensus_deviation_tensor = (
            stacked - current_consensus.unsqueeze(0)
        ).abs() / (
            current_scale.unsqueeze(0)
            + float(self.config.epsilon)
        )
        require_finite_tensor(
            abs_deviation,
            phase="niabd",
            metric="historical_deviation",
        )
        require_finite_tensor(
            consensus_deviation_tensor,
            phase="niabd",
            metric="consensus_deviation",
        )
        teacher_metrics = self._teacher_metrics(
            abs_deviation,
            consensus_deviation_tensor,
        )
        eligible = teacher_metrics["eligible"]
        normal_count = int(eligible.sum().item())

        threshold_view = previous_thresholds.view(1, 1, -1)
        excess = torch.relu(abs_deviation - threshold_view)
        smoothness = float(self.config.transition_smoothness)
        weights = torch.exp(
            -(excess.square()) / (2.0 * smoothness ** 2)
        )
        if self.config.reference_source == "student":
            reference = student.unsqueeze(0).expand_as(stacked)
        else:
            reference = previous_mean.unsqueeze(0).expand_as(stacked)
        purified = weights * stacked + (1.0 - weights) * reference
        require_finite_tensor(
            weights,
            phase="niabd",
            metric="purification_weights",
        )
        require_finite_tensor(
            purified,
            phase="niabd",
            metric="purified_logits",
        )

        update_mask = eligible.clone()
        reason = "normal_eligible_update"
        current_consensus_drift = 0.0
        if normal_count < int(self.config.minimum_consensus_teachers):
            update_mask = torch.zeros_like(eligible)
            candidate_mask, consensus_reason = self._warmup_consensus_gate(
                stacked
            )
            required = max(
                int(self.config.minimum_consensus_teachers),
                int(
                    ceil(
                        float(self.config.consensus_recovery_fraction)
                        * int(stacked.shape[0])
                    )
                ),
            )
            candidate_count = int(candidate_mask.sum().item())
            compact = False
            if candidate_count > 0:
                candidate_deviation = consensus_deviation_tensor[candidate_mask]
                compact = float(
                    torch.quantile(
                        candidate_deviation.reshape(-1),
                        0.75,
                    ).item()
                ) <= float(self.config.benign_deviation_limit)
            if (
                int(stacked.shape[0]) >= int(self.config.minimum_consensus_teachers)
                and candidate_count >= required
                and compact
                and consensus_reason == "warmup_robust_update"
            ):
                update_mask = candidate_mask
                reason = "consensus_drift_update"
                current_consensus_drift = 1.0
            elif consensus_reason == "freeze_ambiguous_consensus":
                reason = "freeze_ambiguous_consensus"
            elif int(stacked.shape[0]) < int(self.config.minimum_consensus_teachers):
                reason = "freeze_insufficient_teachers"
            else:
                reason = "freeze_no_safe_candidate"

        purified_knowledge = tuple(
            TeacherKnowledge(
                metadata=item.metadata,
                logits=purified[index].to(dtype=original_dtypes[index]).clone(),
            )
            for index, item in enumerate(teacher_knowledge)
        )
        prototype_updated = bool(update_mask.any().item())
        if prototype_updated:
            effective_weight = self._update_memory(stacked, update_mask)
            self._consecutive_frozen_rounds = 0
        else:
            effective_weight = 0.0
            self._consecutive_frozen_rounds += 1

        # Threshold adaptation is deliberately after purification and only
        # uses normal memory-eligible teachers, never drift-only candidates.
        if normal_count >= int(self.config.minimum_consensus_teachers):
            self._update_thresholds(abs_deviation, eligible)

        mean_abs_deviation = abs_deviation.mean(dim=(1, 2))
        max_abs_deviation = abs_deviation.amax(dim=(1, 2))
        anomaly_mask = abs_deviation > threshold_view
        mean_suppression = (1.0 - weights).mean(dim=(1, 2))
        records = self._records(
            teacher_knowledge,
            anomaly_fraction=teacher_metrics["anomaly_fraction"],
            mean_abs_deviation=mean_abs_deviation,
            max_abs_deviation=max_abs_deviation,
            mean_suppression=mean_suppression,
            memory_eligible=eligible,
            high_quantile=teacher_metrics["high_quantile_deviation"],
            mean_excess=teacher_metrics["mean_excess"],
            consensus_deviation=teacher_metrics["consensus_deviation"],
            teacher_memory_score=teacher_metrics["teacher_memory_score"],
        )
        metrics = self._metrics(
            warmup=0.0,
            prototype_updated=prototype_updated,
            reason=reason,
            candidate_teachers=int(update_mask.sum().item()),
            memory_eligible_teachers=normal_count,
            anomaly_fraction=float(anomaly_mask.float().mean().item()),
            mean_suppression=float((1.0 - weights).mean().item()),
            teacher_metrics=teacher_metrics,
            current_consensus_drift=current_consensus_drift,
            effective_memory_weight=effective_weight,
        )
        return DefenseResult(
            method=self.name,
            purified_knowledge=purified_knowledge,
            records=records,
            metrics=metrics,
        )
