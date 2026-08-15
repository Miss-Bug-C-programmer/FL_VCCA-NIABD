from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from admission import TeacherKnowledge
from defense import DefenseResult, TeacherDefenseRecord
from numeric_integrity import require_finite_tensor


NIABD_ALGORITHM_VERSION = "niabd-v3-trusted-memory-recovery-controller"
RESULT_SCHEMA_VERSION = "fedagg-results-v3"
_ROBUST_MAD_SCALE = 1.4826


@dataclass(frozen=True)
class NIABDConfig:
    """Prediction-only NIABD configuration.

    The legacy prototype/threshold names remain available for old experiment
    files.  Trusted memory is updated from a robust current center after
    clipping to the previous trusted state.  The controller phase is derived
    only from server-observable teacher statistics.
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
    risk_ema_beta: float = 0.30
    risk_on: float = 1.25
    risk_off: float = 0.60
    onset_patience: int = 2
    recovery_patience: int = 2
    stable_patience: int = 2
    memory_clip_z: float = 3.0
    reference_clip_z: float = 2.0
    normal_memory_lr: Optional[float] = None
    suspicious_memory_lr: float = 0.0
    recovery_memory_lr: float = 0.20
    clean_ce_weight_normal: float = 0.05
    clean_ce_weight_suspicious: float = 0.10
    clean_ce_weight_recovery: float = 0.20
    threshold_upward_step_limit: float = 0.05

    def __post_init__(self) -> None:
        if float(self.initial_threshold) <= 0.0:
            raise ValueError("initial_threshold must be positive.")
        if float(self.minimum_threshold) <= 0.0:
            raise ValueError("minimum_threshold must be positive.")
        if float(self.maximum_threshold) < float(self.minimum_threshold):
            raise ValueError("maximum_threshold must not be below minimum_threshold.")
        if not self.minimum_threshold <= self.initial_threshold <= self.maximum_threshold:
            raise ValueError("initial_threshold must lie within threshold bounds.")
        if float(self.transition_smoothness) <= 0.0:
            raise ValueError("transition_smoothness must be positive.")
        for name, value in (
            ("prototype_learning_rate", self.prototype_learning_rate),
            ("threshold_learning_rate", self.threshold_learning_rate),
            ("recovery_memory_lr", self.recovery_memory_lr),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1].")
        if not 0.0 <= float(self.suspicious_memory_lr) <= 1.0:
            raise ValueError("suspicious_memory_lr must be in [0, 1].")
        if self.normal_memory_lr is not None and not 0.0 < float(self.normal_memory_lr) <= 1.0:
            raise ValueError("normal_memory_lr must be in (0, 1].")
        if not 0.0 < float(self.potentiation_balance) < 1.0:
            raise ValueError("potentiation_balance must be in (0, 1).")
        if float(self.threshold_decay) <= 0.0:
            raise ValueError("threshold_decay must be positive.")
        if float(self.benign_deviation_limit) <= 0.0:
            raise ValueError("benign_deviation_limit must be positive.")
        if int(self.warmup_rounds) < 1:
            raise ValueError("warmup_rounds must be at least 1.")
        if float(self.minimum_standard_deviation) <= 0.0:
            raise ValueError("minimum_standard_deviation must be positive.")
        if self.reference_source not in {"prototype", "student"}:
            raise ValueError("reference_source must be 'prototype' or 'student'.")
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")
        if not 0.0 < float(self.memory_quantile) < 1.0:
            raise ValueError("memory_quantile must be in (0, 1).")
        if not 0.0 <= float(self.maximum_memory_anomaly_fraction) <= 1.0:
            raise ValueError("maximum_memory_anomaly_fraction must be in [0, 1].")
        if float(self.teacher_score_beta) <= 0.0 or float(self.teacher_score_scale_floor) <= 0.0:
            raise ValueError("teacher score controls must be positive.")
        if int(self.minimum_consensus_teachers) < 2:
            raise ValueError("minimum_consensus_teachers must be at least 2.")
        if not 0.5 < float(self.consensus_recovery_fraction) <= 1.0:
            raise ValueError("consensus_recovery_fraction must be in (0.5, 1].")
        if int(self.proxy_chunk_size) < 0:
            raise ValueError("proxy_chunk_size must be non-negative.")
        if not 0.0 < float(self.threshold_exposure_quantile) < 1.0:
            raise ValueError("threshold_exposure_quantile must be in (0, 1).")
        if not 0.0 < float(self.risk_ema_beta) <= 1.0:
            raise ValueError("risk_ema_beta must be in (0, 1].")
        if not 0.0 < float(self.risk_off) < float(self.risk_on):
            raise ValueError("risk_off must be below risk_on and positive.")
        for name, value in (
            ("onset_patience", self.onset_patience),
            ("recovery_patience", self.recovery_patience),
            ("stable_patience", self.stable_patience),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if float(self.memory_clip_z) <= 0.0 or float(self.reference_clip_z) <= 0.0:
            raise ValueError("memory/reference clip z values must be positive.")
        for value in (
            self.clean_ce_weight_normal,
            self.clean_ce_weight_suspicious,
            self.clean_ce_weight_recovery,
        ):
            if float(value) < 0.0:
                raise ValueError("clean CE weights must be non-negative.")
        if float(self.threshold_upward_step_limit) < 0.0:
            raise ValueError("threshold_upward_step_limit must be non-negative.")

    @property
    def effective_normal_memory_lr(self) -> float:
        return float(
            self.prototype_learning_rate
            if self.normal_memory_lr is None
            else self.normal_memory_lr
        )


class NeuroInspiredAdaptiveBackdoorDefense:
    """NIABD with trusted memory and a non-oracle NORMAL/SUSPICIOUS/RECOVERY controller."""

    name = "niabd"
    algorithm_version = NIABD_ALGORITHM_VERSION
    result_schema_version = RESULT_SCHEMA_VERSION
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    RECOVERY = "RECOVERY"

    def __init__(self, config: Optional[NIABDConfig] = None) -> None:
        self.config = config or NIABDConfig()
        self._trusted_mean: Optional[torch.Tensor] = None
        self._trusted_variance: Optional[torch.Tensor] = None
        self._prototype_mean: Optional[torch.Tensor] = None
        self._prototype_variance: Optional[torch.Tensor] = None
        self._thresholds: Optional[torch.Tensor] = None
        self._phase = self.NORMAL
        self._risk_ema = 0.0
        self._round_risk = 0.0
        self._consensus_shift = 0.0
        self._eligible_ratio = 0.0
        self._suspicious_rounds = 0
        self._recovery_rounds = 0
        self._stable_rounds = 0
        self._observation_count = 0
        self._eligible_teacher_observations = 0
        self._memory_update_rounds = 0
        self._consecutive_frozen_rounds = 0
        self._proxy_shape: Optional[Tuple[int, int]] = None
        self._proxy_version = ""

    def _set_memory(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self._trusted_mean = mean.clone()
        self._trusted_variance = variance.clone()
        # Keep the old names as live aliases for existing callers.
        self._prototype_mean = self._trusted_mean
        self._prototype_variance = self._trusted_variance

    def reset(self) -> None:
        self._trusted_mean = None
        self._trusted_variance = None
        self._prototype_mean = None
        self._prototype_variance = None
        self._thresholds = None
        self._phase = self.NORMAL
        self._risk_ema = 0.0
        self._round_risk = 0.0
        self._consensus_shift = 0.0
        self._eligible_ratio = 0.0
        self._suspicious_rounds = 0
        self._recovery_rounds = 0
        self._stable_rounds = 0
        self._observation_count = 0
        self._eligible_teacher_observations = 0
        self._memory_update_rounds = 0
        self._consecutive_frozen_rounds = 0
        self._proxy_shape = None
        self._proxy_version = ""

    def snapshot_state(self) -> dict:
        return {
            "trusted_mean": None if self._trusted_mean is None else self._trusted_mean.clone(),
            "trusted_variance": None if self._trusted_variance is None else self._trusted_variance.clone(),
            "prototype_mean": None if self._trusted_mean is None else self._trusted_mean.clone(),
            "prototype_variance": None if self._trusted_variance is None else self._trusted_variance.clone(),
            "thresholds": None if self._thresholds is None else self._thresholds.clone(),
            "phase": self._phase,
            "risk_ema": float(self._risk_ema),
            "round_risk": float(self._round_risk),
            "consensus_shift": float(self._consensus_shift),
            "eligible_ratio": float(self._eligible_ratio),
            "suspicious_rounds": int(self._suspicious_rounds),
            "recovery_rounds": int(self._recovery_rounds),
            "stable_rounds": int(self._stable_rounds),
            "observation_count": int(self._observation_count),
            "eligible_teacher_observations": int(self._eligible_teacher_observations),
            "memory_update_rounds": int(self._memory_update_rounds),
            "consecutive_frozen_rounds": int(self._consecutive_frozen_rounds),
            "proxy_shape": self._proxy_shape,
            "proxy_version": self._proxy_version,
            "algorithm_version": self.algorithm_version,
        }

    def restore_state(self, state: dict) -> None:
        if str(state.get("algorithm_version")) != self.algorithm_version:
            raise ValueError("NIABD snapshot algorithm version mismatch.")
        trusted_mean = state.get("trusted_mean", state.get("prototype_mean"))
        trusted_variance = state.get("trusted_variance", state.get("prototype_variance"))
        self._trusted_mean = None if trusted_mean is None else trusted_mean.clone()
        self._trusted_variance = None if trusted_variance is None else trusted_variance.clone()
        self._prototype_mean = self._trusted_mean
        self._prototype_variance = self._trusted_variance
        thresholds = state.get("thresholds")
        self._thresholds = None if thresholds is None else thresholds.clone()
        phase = str(state.get("phase", self.NORMAL))
        if phase not in {self.NORMAL, self.SUSPICIOUS, self.RECOVERY}:
            raise ValueError("NIABD snapshot phase is invalid.")
        self._phase = phase
        self._risk_ema = float(state.get("risk_ema", 0.0))
        self._round_risk = float(state.get("round_risk", self._risk_ema))
        self._consensus_shift = float(state.get("consensus_shift", 0.0))
        self._eligible_ratio = float(state.get("eligible_ratio", 0.0))
        self._suspicious_rounds = int(state.get("suspicious_rounds", 0))
        self._recovery_rounds = int(state.get("recovery_rounds", 0))
        self._stable_rounds = int(state.get("stable_rounds", 0))
        self._observation_count = int(state.get("observation_count", 0))
        self._eligible_teacher_observations = int(state.get("eligible_teacher_observations", 0))
        self._memory_update_rounds = int(state.get("memory_update_rounds", 0))
        self._consecutive_frozen_rounds = int(state.get("consecutive_frozen_rounds", 0))
        shape = state.get("proxy_shape")
        self._proxy_shape = None if shape is None else tuple(int(value) for value in shape)
        self._proxy_version = str(state.get("proxy_version", ""))

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def risk_ema(self) -> float:
        return float(self._risk_ema)

    @property
    def prototype_mean(self) -> Optional[torch.Tensor]:
        return None if self._trusted_mean is None else self._trusted_mean.clone()

    @property
    def trusted_mean(self) -> Optional[torch.Tensor]:
        return self.prototype_mean

    @property
    def prototype_variance(self) -> Optional[torch.Tensor]:
        return None if self._trusted_variance is None else self._trusted_variance.clone()

    @property
    def trusted_variance(self) -> Optional[torch.Tensor]:
        return self.prototype_variance

    @property
    def thresholds(self) -> Optional[torch.Tensor]:
        return None if self._thresholds is None else self._thresholds.clone()

    @property
    def observation_count(self) -> int:
        return int(self._observation_count)

    def clean_ce_weight(self) -> float:
        return float(
            {
                self.NORMAL: self.config.clean_ce_weight_normal,
                self.SUSPICIOUS: self.config.clean_ce_weight_suspicious,
                self.RECOVERY: self.config.clean_ce_weight_recovery,
            }[self._phase]
        )

    @staticmethod
    def _to_cpu_float(logits: torch.Tensor, *, name: str) -> torch.Tensor:
        if not torch.is_tensor(logits) or logits.ndim != 2:
            raise ValueError(f"NIABD {name} must have shape [samples, classes].")
        if not torch.is_floating_point(logits):
            raise TypeError(f"NIABD {name} must use a floating dtype.")
        require_finite_tensor(logits, phase="niabd", metric=name)
        result = logits.detach().to(device="cpu", dtype=torch.float32)
        require_finite_tensor(result, phase="niabd", metric=f"{name}_float32")
        return result

    def _stack_knowledge(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
    ) -> Tuple[torch.Tensor, Tuple[torch.dtype, ...]]:
        if not teacher_knowledge:
            raise ValueError("NIABD requires at least one teacher.")
        tensors = [
            self._to_cpu_float(item.logits, name=f"teacher_logits[{index}]")
            for index, item in enumerate(teacher_knowledge)
        ]
        shape = tensors[0].shape
        if int(shape[0]) <= 0 or int(shape[1]) <= 1:
            raise ValueError("NIABD requires a non-empty [P,C] proxy with C > 1.")
        if any(tensor.shape != shape for tensor in tensors):
            raise ValueError("NIABD teacher logits must share one proxy shape.")
        versions = {str(item.metadata.proxy_version) for item in teacher_knowledge if str(item.metadata.proxy_version)}
        if len(versions) > 1:
            raise ValueError("NIABD teacher packets have mixed proxy versions.")
        incoming = next(iter(versions), "")
        if self._proxy_version and incoming != self._proxy_version:
            raise RuntimeError("NIABD proxy_version changed; call reset().")
        if incoming:
            self._proxy_version = incoming
        return torch.stack(tensors, dim=0), tuple(item.logits.dtype for item in teacher_knowledge)

    def _validate_memory_shape(self, shape: torch.Size) -> None:
        if self._trusted_mean is None:
            return
        expected = tuple(int(value) for value in shape)
        if tuple(self._trusted_mean.shape) != expected or tuple(self._trusted_variance.shape) != expected:
            raise RuntimeError("NIABD memory shape changed; call reset().")
        if self._proxy_shape is not None and tuple(self._proxy_shape) != expected:
            raise RuntimeError("NIABD proxy shape changed; call reset().")
        if tuple(self._thresholds.shape) != (expected[1],):
            raise RuntimeError("NIABD threshold shape is inconsistent with memory.")

    def _robust_center_variance(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        center = torch.median(values, dim=0).values
        mad = torch.median((values - center.unsqueeze(0)).abs(), dim=0).values
        scale = torch.maximum(
            mad * _ROBUST_MAD_SCALE,
            torch.full_like(mad, float(self.config.minimum_standard_deviation)),
        )
        variance = scale.square()
        require_finite_tensor(center, phase="niabd", metric="robust_center")
        require_finite_tensor(variance, phase="niabd", metric="robust_variance")
        return center, variance

    def _initialize_memory(self, values: torch.Tensor) -> None:
        center, variance = self._robust_center_variance(values)
        self._set_memory(center, variance)
        self._thresholds = torch.full(
            (int(values.shape[-1]),),
            float(self.config.initial_threshold),
            dtype=torch.float32,
        )
        self._observation_count = int(values.shape[0] * values.shape[1])
        self._eligible_teacher_observations = self._observation_count
        self._memory_update_rounds = 1
        self._proxy_shape = (int(values.shape[1]), int(values.shape[2]))

    @staticmethod
    def _upper_robust_z(values: torch.Tensor, floor: float) -> torch.Tensor:
        center = torch.median(values)
        mad = torch.median((values - center).abs())
        scale = max(float(_ROBUST_MAD_SCALE * mad.item()), float(floor))
        return torch.relu((values - center) / scale)

    def _teacher_metrics(
        self,
        abs_deviation: torch.Tensor,
        consensus_deviation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        assert self._thresholds is not None
        threshold_view = self._thresholds.view(1, 1, -1)
        anomaly_fraction = (abs_deviation > threshold_view).float().mean(dim=(1, 2))
        high_quantile = torch.quantile(
            abs_deviation.reshape(abs_deviation.shape[0], -1),
            float(self.config.memory_quantile),
            dim=1,
        )
        mean_excess = torch.relu(abs_deviation - threshold_view).mean(dim=(1, 2))
        consensus_quantile = torch.quantile(
            consensus_deviation.reshape(consensus_deviation.shape[0], -1),
            float(self.config.memory_quantile),
            dim=1,
        )
        score = torch.stack(
            [
                self._upper_robust_z(anomaly_fraction, self.config.teacher_score_scale_floor),
                self._upper_robust_z(high_quantile, self.config.teacher_score_scale_floor),
                self._upper_robust_z(mean_excess, self.config.teacher_score_scale_floor),
                self._upper_robust_z(consensus_quantile, self.config.teacher_score_scale_floor),
            ],
            dim=0,
        ).amax(dim=0)
        eligible = (
            (score <= float(self.config.teacher_score_beta))
            & (high_quantile <= float(self.config.benign_deviation_limit))
            & (anomaly_fraction <= float(self.config.maximum_memory_anomaly_fraction))
            & (consensus_quantile <= float(self.config.benign_deviation_limit))
        )
        return {
            "anomaly_fraction": anomaly_fraction,
            "high_quantile_deviation": high_quantile,
            "mean_excess": mean_excess,
            "consensus_deviation": consensus_quantile,
            "teacher_memory_score": score,
            "eligible": eligible,
        }

    def _warmup_candidates(self, stacked: torch.Tensor) -> Tuple[torch.Tensor, str]:
        count = int(stacked.shape[0])
        required = max(
            int(self.config.minimum_consensus_teachers),
            int(ceil(float(self.config.consensus_recovery_fraction) * count)),
        )
        if count < int(self.config.minimum_consensus_teachers):
            return torch.zeros(count, dtype=torch.bool), "freeze_insufficient_teachers"
        center, variance = self._robust_center_variance(stacked)
        z = (stacked - center.unsqueeze(0)).abs() / (torch.sqrt(variance).unsqueeze(0) + self.config.epsilon)
        distance = torch.quantile(z.reshape(count, -1), float(self.config.memory_quantile), dim=1)
        mask = distance <= float(self.config.benign_deviation_limit)
        if int(mask.sum().item()) < required:
            return mask, "freeze_no_safe_candidate"
        return mask, "warmup_robust_update"

    def _transition(self, risk: float) -> None:
        beta = float(self.config.risk_ema_beta)
        self._risk_ema = beta * float(risk) + (1.0 - beta) * self._risk_ema
        if self._phase == self.NORMAL:
            self._recovery_rounds = 0
            self._stable_rounds = 0
            if self._risk_ema >= float(self.config.risk_on):
                self._suspicious_rounds += 1
            else:
                self._suspicious_rounds = 0
            if self._suspicious_rounds >= int(self.config.onset_patience):
                self._phase = self.SUSPICIOUS
                self._suspicious_rounds = 0
        elif self._phase == self.SUSPICIOUS:
            self._stable_rounds = 0
            if self._risk_ema <= float(self.config.risk_off):
                self._recovery_rounds += 1
            else:
                self._recovery_rounds = 0
            if self._recovery_rounds >= int(self.config.recovery_patience):
                self._phase = self.RECOVERY
                self._recovery_rounds = 0
                self._stable_rounds = 0
        else:
            if self._risk_ema >= float(self.config.risk_on):
                self._phase = self.SUSPICIOUS
                self._suspicious_rounds = 0
                self._stable_rounds = 0
            elif self._risk_ema <= float(self.config.risk_off):
                self._stable_rounds += 1
            else:
                self._stable_rounds = 0
            if self._stable_rounds >= int(self.config.stable_patience):
                self._phase = self.NORMAL
                self._stable_rounds = 0

    def _safe_reference(self, current_consensus: torch.Tensor, student: torch.Tensor) -> Tuple[torch.Tensor, float]:
        assert self._trusted_mean is not None and self._trusted_variance is not None
        std = torch.sqrt(self._trusted_variance).clamp_min(float(self.config.minimum_standard_deviation))
        delta = (current_consensus - self._trusted_mean).clamp(
            min=-float(self.config.reference_clip_z) * std,
            max=float(self.config.reference_clip_z) * std,
        )
        safe_consensus = self._trusted_mean + delta
        alpha = {
            self.NORMAL: 0.50,
            self.SUSPICIOUS: 0.85,
            self.RECOVERY: 0.25,
        }[self._phase]
        reference = alpha * self._trusted_mean + (1.0 - alpha) * safe_consensus
        if self.config.reference_source == "student":
            reference = 0.75 * reference + 0.25 * student
        return reference, float(alpha)

    def _update_memory(self, stacked: torch.Tensor, eligible: torch.Tensor) -> bool:
        assert self._trusted_mean is not None and self._trusted_variance is not None
        if not bool(eligible.any().item()) or self._phase == self.SUSPICIOUS:
            return False
        center, variance = self._robust_center_variance(stacked[eligible])
        std = torch.sqrt(self._trusted_variance).clamp_min(float(self.config.minimum_standard_deviation))
        center = center.clamp(
            min=self._trusted_mean - float(self.config.memory_clip_z) * std,
            max=self._trusted_mean + float(self.config.memory_clip_z) * std,
        )
        eta = (
            float(self.config.effective_normal_memory_lr)
            if self._phase == self.NORMAL
            else float(self.config.recovery_memory_lr)
        )
        new_mean = (1.0 - eta) * self._trusted_mean + eta * center
        variance = variance.clamp_min(
            float(self.config.minimum_standard_deviation) ** 2
        )
        variance = torch.minimum(
            variance,
            (std * (1.0 + float(self.config.memory_clip_z))).square(),
        )
        new_variance = (1.0 - eta) * self._trusted_variance + eta * variance
        require_finite_tensor(new_mean, phase="niabd", metric="trusted_mean_update")
        require_finite_tensor(new_variance, phase="niabd", metric="trusted_variance_update")
        self._set_memory(new_mean, new_variance)
        self._observation_count += int(eligible.sum().item()) * int(stacked.shape[1])
        self._eligible_teacher_observations += int(eligible.sum().item()) * int(stacked.shape[1])
        self._memory_update_rounds += 1
        return True

    def _update_thresholds(self, abs_deviation: torch.Tensor, eligible: torch.Tensor) -> str:
        assert self._thresholds is not None
        if not bool(eligible.any().item()):
            return "frozen_no_eligible"
        exposure = torch.quantile(
            abs_deviation[eligible].reshape(-1, abs_deviation.shape[-1]),
            float(self.config.threshold_exposure_quantile),
            dim=0,
        )
        thresholds = self._thresholds
        if self._phase == self.SUSPICIOUS:
            delta = torch.full_like(thresholds, -float(self.config.threshold_decay))
            mode = "suspicious_downward_only"
        elif self._phase == self.RECOVERY:
            target = exposure.clamp(
                min=float(self.config.minimum_threshold),
                max=float(self.config.maximum_threshold),
            )
            delta = (target - thresholds).clamp(
                min=-float(self.config.threshold_decay),
                max=float(self.config.threshold_upward_step_limit),
            )
            mode = "recovery_clipped_recalibration"
        else:
            raw = float(self.config.potentiation_balance) * torch.relu(exposure - thresholds)
            raw = raw.clamp_max(float(self.config.threshold_upward_step_limit))
            depression = (1.0 - float(self.config.potentiation_balance)) * float(self.config.threshold_decay)
            delta = torch.where(exposure > thresholds, raw, torch.full_like(thresholds, -depression))
            mode = "normal_robust_exposure"
        updated = (thresholds + float(self.config.threshold_learning_rate) * delta).clamp(
            min=float(self.config.minimum_threshold), max=float(self.config.maximum_threshold)
        )
        require_finite_tensor(updated, phase="niabd", metric="threshold_update")
        self._thresholds = updated
        return mode

    def _record_tuple(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
        metrics: Dict[str, torch.Tensor],
        abs_deviation: torch.Tensor,
        suppression: torch.Tensor,
        memory_updated: bool,
        threshold_mode: str,
    ) -> Tuple[TeacherDefenseRecord, ...]:
        eligible = metrics["eligible"]
        return tuple(
            TeacherDefenseRecord(
                client_id=int(item.metadata.client_id),
                anomaly_fraction=float(metrics["anomaly_fraction"][index].item()),
                mean_abs_deviation=float(abs_deviation[index].mean().item()),
                max_abs_deviation=float(abs_deviation[index].max().item()),
                mean_suppression=float(suppression[index].mean().item()),
                memory_eligible=bool(eligible[index].item()),
                teacher_memory_score=float(metrics["teacher_memory_score"][index].item()),
                high_quantile_deviation=float(metrics["high_quantile_deviation"][index].item()),
                mean_excess=float(metrics["mean_excess"][index].item()),
                consensus_deviation=float(metrics["consensus_deviation"][index].item()),
                phase=self._phase,
                round_risk=float(self._round_risk),
                risk_ema=float(self._risk_ema),
                consensus_shift=float(self._consensus_shift),
                eligible_ratio=float(self._eligible_ratio),
                trusted_memory_frozen=bool(self._phase == self.SUSPICIOUS),
                trusted_memory_updated=bool(memory_updated),
                threshold_update_mode=threshold_mode,
                reference_trusted_weight=float(self._reference_trusted_weight),
                recovery_stable_rounds=int(self._stable_rounds),
            )
            for index, item in enumerate(teacher_knowledge)
        )

    _reference_trusted_weight = 0.5

    def _metrics(
        self,
        *,
        warmup: float,
        reason: str,
        prototype_updated: bool,
        candidate_count: int,
        memory_eligible: int,
        anomaly_fraction: float,
        mean_suppression: float,
        teacher_metrics: Optional[Dict[str, torch.Tensor]],
        threshold_mode: str,
    ) -> Dict[str, object]:
        assert self._thresholds is not None
        if teacher_metrics is None:
            score_mean = score_median = score_mad = high = excess = consensus = float("nan")
        else:
            scores = teacher_metrics["teacher_memory_score"]
            score_mean = float(scores.mean().item())
            score_median = float(torch.median(scores).item())
            score_mad = float(torch.median((scores - torch.median(scores)).abs()).item())
            high = float(teacher_metrics["high_quantile_deviation"].mean().item())
            excess = float(teacher_metrics["mean_excess"].mean().item())
            consensus = float(teacher_metrics["consensus_deviation"].mean().item())
        return {
            "warmup": float(warmup),
            "prototype_updated": float(prototype_updated),
            "prototype_observations": float(self._observation_count),
            "threshold_mean": float(self._thresholds.mean().item()),
            "threshold_min": float(self._thresholds.min().item()),
            "threshold_max": float(self._thresholds.max().item()),
            "anomaly_fraction": float(anomaly_fraction),
            "mean_suppression": float(mean_suppression),
            "memory_eligible_teachers": int(memory_eligible),
            "niabd_algorithm_version": self.algorithm_version,
            "result_schema_version": self.result_schema_version,
            "niabd_prototype_update_reason": reason,
            "niabd_memory_candidate_teachers": int(candidate_count),
            "niabd_teacher_score_mean": score_mean,
            "niabd_teacher_score_median": score_median,
            "niabd_teacher_score_mad": score_mad,
            "niabd_high_quantile_deviation": high,
            "niabd_mean_excess": excess,
            "niabd_consensus_deviation": consensus,
            "niabd_current_consensus_drift": float(self._consensus_shift),
            "niabd_all_ineligible_round": float(memory_eligible == 0),
            "niabd_consecutive_frozen_rounds": int(self._consecutive_frozen_rounds),
            "niabd_effective_memory_weight": float(
                self.config.effective_normal_memory_lr
                if prototype_updated and self._phase == self.NORMAL
                else self.config.recovery_memory_lr if prototype_updated else 0.0
            ),
            "niabd_eligible_teacher_observations": int(self._eligible_teacher_observations),
            "niabd_memory_update_rounds": int(self._memory_update_rounds),
            "niabd_defense_available": True,
            "niabd_purification_applied": True,
            "niabd_memory_updated": bool(prototype_updated),
            "niabd_phase": self._phase,
            "niabd_round_risk": float(self._round_risk),
            "niabd_risk_ema": float(self._risk_ema),
            "niabd_consensus_shift": float(self._consensus_shift),
            "niabd_eligible_ratio": float(self._eligible_ratio),
            "niabd_trusted_memory_frozen": bool(self._phase == self.SUSPICIOUS),
            "niabd_trusted_memory_updated": bool(prototype_updated),
            "niabd_threshold_update_mode": threshold_mode,
            "niabd_reference_trusted_weight": float(self._reference_trusted_weight),
            "niabd_recovery_stable_rounds": int(self._stable_rounds),
        }

    def purify(
        self,
        *,
        teacher_knowledge: Sequence[TeacherKnowledge],
        student_logits: torch.Tensor,
        proxy_labels: torch.Tensor,
        current_round: int,
    ) -> DefenseResult:
        del proxy_labels
        if int(current_round) <= 0:
            raise ValueError("current_round must be positive.")
        stacked, original_dtypes = self._stack_knowledge(teacher_knowledge)
        student = self._to_cpu_float(student_logits, name="student_logits")
        if student.shape != stacked.shape[1:]:
            raise ValueError("NIABD student and teacher logits must share a proxy shape.")
        self._validate_memory_shape(student.shape)
        candidates, warmup_reason = self._warmup_candidates(stacked)
        if self._trusted_mean is None:
            if warmup_reason != "warmup_robust_update":
                self._consecutive_frozen_rounds += 1
                return self._warmup_result(teacher_knowledge, candidates, warmup_reason)
            self._initialize_memory(stacked[candidates])
            return self._warmup_result(teacher_knowledge, candidates, "warmup_robust_update")
        assert self._trusted_mean is not None and self._trusted_variance is not None and self._thresholds is not None
        if int(current_round) <= int(self.config.warmup_rounds):
            updated = self._update_memory(stacked, candidates)
            self._consecutive_frozen_rounds = 0 if updated else self._consecutive_frozen_rounds + 1
            return self._warmup_result(teacher_knowledge, candidates, "warmup_robust_update" if updated else warmup_reason)

        previous_mean = self._trusted_mean
        previous_variance = self._trusted_variance
        std = torch.sqrt(previous_variance).clamp_min(float(self.config.minimum_standard_deviation))
        abs_deviation = ((stacked - previous_mean.unsqueeze(0)) / (std.unsqueeze(0) + self.config.epsilon)).abs()
        current_consensus = torch.median(stacked, dim=0).values
        current_mad = torch.median((stacked - current_consensus.unsqueeze(0)).abs(), dim=0).values
        current_scale = torch.maximum(
            _ROBUST_MAD_SCALE * current_mad,
            torch.full_like(current_mad, float(self.config.minimum_standard_deviation)),
        )
        consensus_z = (stacked - current_consensus.unsqueeze(0)).abs() / (current_scale.unsqueeze(0) + self.config.epsilon)
        teacher_metrics = self._teacher_metrics(abs_deviation, consensus_z)
        eligible = teacher_metrics["eligible"]
        self._eligible_ratio = float(eligible.float().mean().item())
        consensus_shift_tensor = (current_consensus - previous_mean).abs() / (std + self.config.epsilon)
        self._consensus_shift = float(torch.quantile(consensus_shift_tensor.reshape(-1), 0.95).item())
        anomaly_fraction = float((abs_deviation > self._thresholds.view(1, 1, -1)).float().mean().item())
        risk_components = [
            self._consensus_shift / float(self.config.benign_deviation_limit),
            anomaly_fraction / max(float(self.config.maximum_memory_anomaly_fraction), self.config.epsilon),
            1.0 - self._eligible_ratio,
            float(torch.median(teacher_metrics["teacher_memory_score"]).item()) / float(self.config.teacher_score_beta),
        ]
        self._round_risk = float(max(0.0, max(risk_components)))
        self._transition(self._round_risk)
        reference, alpha = self._safe_reference(current_consensus, student)
        self._reference_trusted_weight = float(alpha)
        threshold_view = self._thresholds.view(1, 1, -1)
        excess = torch.relu(abs_deviation - threshold_view)
        weights = torch.exp(-(excess.square()) / (2.0 * float(self.config.transition_smoothness) ** 2))
        purified = weights * stacked + (1.0 - weights) * reference.unsqueeze(0)
        require_finite_tensor(purified, phase="niabd", metric="purified_logits")
        update_mask = eligible.clone()
        consensus_recovery = False
        if (
            int(eligible.sum().item()) < int(self.config.minimum_consensus_teachers)
            and self._phase != self.SUSPICIOUS
        ):
            recovery_candidates, recovery_reason = self._warmup_candidates(stacked)
            required = max(
                int(self.config.minimum_consensus_teachers),
                int(ceil(float(self.config.consensus_recovery_fraction) * int(stacked.shape[0]))),
            )
            if (
                recovery_reason == "warmup_robust_update"
                and int(recovery_candidates.sum().item()) >= required
            ):
                update_mask = recovery_candidates
                consensus_recovery = True
        memory_updated = self._update_memory(stacked, update_mask)
        if memory_updated:
            self._consecutive_frozen_rounds = 0
        else:
            self._consecutive_frozen_rounds += 1
        threshold_mode = self._update_thresholds(abs_deviation, eligible)
        reason = {
            self.NORMAL: (
                "consensus_drift_update"
                if consensus_recovery and memory_updated
                else "normal_eligible_update"
                if memory_updated
                else "normal_no_safe_candidate"
            ),
            self.SUSPICIOUS: (
                "freeze_no_safe_candidate"
                if not bool(eligible.any().item())
                else "suspicious_memory_frozen"
            ),
            self.RECOVERY: "recovery_clipped_update" if memory_updated else "recovery_no_safe_candidate",
        }[self._phase]
        records = self._record_tuple(
            teacher_knowledge,
            teacher_metrics,
            abs_deviation,
            1.0 - weights,
            memory_updated,
            threshold_mode,
        )
        metrics = self._metrics(
            warmup=0.0,
            reason=reason,
            prototype_updated=memory_updated,
            candidate_count=int(update_mask.sum().item()),
            memory_eligible=int(eligible.sum().item()),
            anomaly_fraction=anomaly_fraction,
            mean_suppression=float((1.0 - weights).mean().item()),
            teacher_metrics=teacher_metrics,
            threshold_mode=threshold_mode,
        )
        return DefenseResult(
            method=self.name,
            purified_knowledge=tuple(
                TeacherKnowledge(
                    metadata=item.metadata,
                    logits=purified[index].to(dtype=original_dtypes[index]).clone(),
                )
                for index, item in enumerate(teacher_knowledge)
            ),
            records=records,
            metrics=metrics,
        )

    def _warmup_result(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
        candidates: torch.Tensor,
        reason: str,
    ) -> DefenseResult:
        records = tuple(
            TeacherDefenseRecord(
                client_id=int(item.metadata.client_id),
                anomaly_fraction=float("nan"),
                mean_abs_deviation=float("nan"),
                max_abs_deviation=float("nan"),
                mean_suppression=0.0,
                memory_eligible=bool(candidates[index].item()),
                phase=self._phase,
                round_risk=float(self._round_risk),
                risk_ema=float(self._risk_ema),
                consensus_shift=float(self._consensus_shift),
                eligible_ratio=float(self._eligible_ratio),
                trusted_memory_frozen=False,
                trusted_memory_updated=bool(
                    self._trusted_mean is not None and candidates.any().item()
                ),
                threshold_update_mode="warmup_frozen",
                reference_trusted_weight=0.5,
                recovery_stable_rounds=int(self._stable_rounds),
            )
            for index, item in enumerate(teacher_knowledge)
        )
        initialized = self._trusted_mean is not None
        memory_updated = bool(initialized and candidates.any().item())
        threshold_mean = float(self._thresholds.mean().item()) if self._thresholds is not None else float("nan")
        metrics = {
            "warmup": 1.0,
            "prototype_updated": float(memory_updated),
            "prototype_observations": float(self._observation_count),
            "threshold_mean": threshold_mean,
            "threshold_min": float(self._thresholds.min().item()) if self._thresholds is not None else float("nan"),
            "threshold_max": float(self._thresholds.max().item()) if self._thresholds is not None else float("nan"),
            "anomaly_fraction": float("nan"),
            "mean_suppression": 0.0,
            "memory_eligible_teachers": int(candidates.sum().item()),
            "niabd_algorithm_version": self.algorithm_version,
            "result_schema_version": self.result_schema_version,
            "niabd_prototype_update_reason": reason,
            "niabd_memory_candidate_teachers": int(candidates.sum().item()),
            "niabd_teacher_score_mean": float("nan"),
            "niabd_teacher_score_median": float("nan"),
            "niabd_teacher_score_mad": float("nan"),
            "niabd_high_quantile_deviation": float("nan"),
            "niabd_mean_excess": float("nan"),
            "niabd_consensus_deviation": float("nan"),
            "niabd_current_consensus_drift": 0.0,
            "niabd_all_ineligible_round": float(not bool(candidates.any().item())),
            "niabd_consecutive_frozen_rounds": int(self._consecutive_frozen_rounds),
            "niabd_effective_memory_weight": 0.0,
            "niabd_eligible_teacher_observations": int(self._eligible_teacher_observations),
            "niabd_memory_update_rounds": int(self._memory_update_rounds),
            "niabd_defense_available": initialized,
            "niabd_purification_applied": False,
            "niabd_memory_updated": memory_updated,
            "niabd_phase": self._phase,
            "niabd_round_risk": float(self._round_risk),
            "niabd_risk_ema": float(self._risk_ema),
            "niabd_consensus_shift": float(self._consensus_shift),
            "niabd_eligible_ratio": float(self._eligible_ratio),
            "niabd_trusted_memory_frozen": False,
            "niabd_trusted_memory_updated": memory_updated,
            "niabd_threshold_update_mode": "warmup_frozen",
            "niabd_reference_trusted_weight": 0.5,
            "niabd_recovery_stable_rounds": int(self._stable_rounds),
        }
        return DefenseResult(
            method=self.name,
            purified_knowledge=tuple(teacher_knowledge),
            records=records,
            metrics=metrics,
        )
