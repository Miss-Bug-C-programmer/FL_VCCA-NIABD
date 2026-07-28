from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from admission import TeacherKnowledge
from defense import DefenseResult, TeacherDefenseRecord


@dataclass(frozen=True)
class NIABDConfig:
    """Configuration for neuro-inspired adaptive backdoor defense."""

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
            raise ValueError(
                "prototype_learning_rate must be in (0, 1]."
            )
        if float(self.threshold_learning_rate) <= 0.0:
            raise ValueError("threshold_learning_rate must be positive.")
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


class NeuroInspiredAdaptiveBackdoorDefense:
    """Prediction-only NIABD with adaptive prototypes and thresholds."""

    name = "niabd"

    def __init__(self, config: Optional[NIABDConfig] = None) -> None:
        self.config = config or NIABDConfig()
        self._prototype_mean: Optional[torch.Tensor] = None
        self._prototype_variance: Optional[torch.Tensor] = None
        self._thresholds: Optional[torch.Tensor] = None
        self._observation_count = 0

    def reset(self) -> None:
        self._prototype_mean = None
        self._prototype_variance = None
        self._thresholds = None
        self._observation_count = 0

    @property
    def is_initialized(self) -> bool:
        return self._prototype_mean is not None

    @property
    def prototype_mean(self) -> Optional[torch.Tensor]:
        if self._prototype_mean is None:
            return None
        return self._prototype_mean.clone()

    @property
    def thresholds(self) -> Optional[torch.Tensor]:
        if self._thresholds is None:
            return None
        return self._thresholds.clone()

    @staticmethod
    def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(
                "NIABD expects logits with shape [samples, classes]."
            )
        return torch.nan_to_num(
            logits.detach().cpu().float(),
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        ).clamp(-30.0, 30.0)

    def _stack_knowledge(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
    ) -> torch.Tensor:
        if not teacher_knowledge:
            raise ValueError(
                "NIABD requires at least one admitted teacher."
            )
        tensors = [
            self._sanitize_logits(item.logits)
            for item in teacher_knowledge
        ]
        reference_shape = tensors[0].shape
        if any(tensor.shape != reference_shape for tensor in tensors):
            raise ValueError(
                "NIABD teacher logits must have equal proxy shapes."
            )
        if int(reference_shape[1]) <= 1:
            raise ValueError("NIABD requires at least two output classes.")
        return torch.stack(tensors, dim=0)

    def _initialize_memory(
        self,
        stacked_logits: torch.Tensor,
    ) -> None:
        flattened = stacked_logits.reshape(-1, stacked_logits.shape[-1])
        minimum_variance = (
            float(self.config.minimum_standard_deviation) ** 2
        )
        self._prototype_mean = flattened.mean(dim=0)
        self._prototype_variance = flattened.var(
            dim=0,
            unbiased=False,
        ).clamp_min(minimum_variance)
        self._thresholds = torch.full(
            (stacked_logits.shape[-1],),
            float(self.config.initial_threshold),
            dtype=torch.float32,
        )
        self._observation_count = int(flattened.shape[0])

    def _warmup_result(
        self,
        teacher_knowledge: Sequence[TeacherKnowledge],
    ) -> DefenseResult:
        assert self._thresholds is not None
        records = tuple(
            TeacherDefenseRecord(
                client_id=int(item.metadata.client_id),
                anomaly_fraction=0.0,
                mean_abs_deviation=0.0,
                max_abs_deviation=0.0,
                mean_suppression=0.0,
                memory_eligible=True,
            )
            for item in teacher_knowledge
        )
        return DefenseResult(
            method=self.name,
            purified_knowledge=tuple(teacher_knowledge),
            records=records,
            metrics={
                "warmup": 1.0,
                "prototype_updated": 1.0,
                "prototype_observations": float(
                    self._observation_count
                ),
                "threshold_mean": float(
                    self._thresholds.mean().item()
                ),
                "threshold_min": float(
                    self._thresholds.min().item()
                ),
                "threshold_max": float(
                    self._thresholds.max().item()
                ),
                "anomaly_fraction": 0.0,
                "mean_suppression": 0.0,
                "memory_eligible_teachers": float(
                    len(teacher_knowledge)
                ),
            },
        )

    def _update_thresholds(
        self,
        abs_deviation: torch.Tensor,
    ) -> None:
        assert self._thresholds is not None
        exposure = abs_deviation.mean(dim=(0, 1))
        above = exposure > self._thresholds
        potentiation = (
            float(self.config.potentiation_balance)
            * torch.relu(exposure - self._thresholds)
        )
        depression = (
            (1.0 - float(self.config.potentiation_balance))
            * float(self.config.threshold_decay)
        )
        delta = torch.where(
            above,
            potentiation,
            torch.full_like(self._thresholds, -depression),
        )
        self._thresholds = (
            self._thresholds
            + float(self.config.threshold_learning_rate) * delta
        ).clamp(
            min=float(self.config.minimum_threshold),
            max=float(self.config.maximum_threshold),
        )

    def _update_prototypes(
        self,
        stacked_logits: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> bool:
        assert self._prototype_mean is not None
        assert self._prototype_variance is not None
        if not bool(eligible_mask.any().item()):
            return False
        values = stacked_logits[eligible_mask].reshape(
            -1,
            stacked_logits.shape[-1],
        )
        eta = float(self.config.prototype_learning_rate)
        batch_mean = values.mean(dim=0)
        new_mean = (1.0 - eta) * self._prototype_mean + eta * batch_mean
        batch_variance = ((values - new_mean) ** 2).mean(dim=0)
        minimum_variance = (
            float(self.config.minimum_standard_deviation) ** 2
        )
        self._prototype_mean = new_mean
        self._prototype_variance = (
            (1.0 - eta) * self._prototype_variance
            + eta * batch_variance
        ).clamp_min(minimum_variance)
        self._observation_count += int(values.shape[0])
        return True

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
        stacked = self._stack_knowledge(teacher_knowledge)
        student = self._sanitize_logits(student_logits)
        if student.shape != stacked.shape[1:]:
            raise ValueError(
                "NIABD student and teacher logits must share a proxy shape."
            )

        initialized_now = False
        if not self.is_initialized:
            self._initialize_memory(stacked)
            initialized_now = True
        if int(current_round) <= int(self.config.warmup_rounds):
            if not initialized_now:
                self._update_prototypes(
                    stacked,
                    torch.ones(
                        stacked.shape[0],
                        dtype=torch.bool,
                    ),
                )
            return self._warmup_result(teacher_knowledge)

        assert self._prototype_mean is not None
        assert self._prototype_variance is not None
        assert self._thresholds is not None
        standard_deviation = torch.sqrt(self._prototype_variance)
        deviation = (
            stacked - self._prototype_mean.view(1, 1, -1)
        ) / (
            standard_deviation.view(1, 1, -1)
            + float(self.config.epsilon)
        )
        abs_deviation = deviation.abs()
        excess = torch.relu(
            abs_deviation - self._thresholds.view(1, 1, -1)
        )
        smoothness = float(self.config.transition_smoothness)
        weights = torch.exp(
            -(excess ** 2) / (2.0 * smoothness ** 2)
        )
        if self.config.reference_source == "student":
            reference = student.unsqueeze(0).expand_as(stacked)
        else:
            reference = self._prototype_mean.view(
                1,
                1,
                -1,
            ).expand_as(stacked)
        purified = weights * stacked + (1.0 - weights) * reference

        anomaly_mask = abs_deviation > self._thresholds.view(1, 1, -1)
        teacher_max_deviation = abs_deviation.amax(dim=(1, 2))
        eligible = (
            teacher_max_deviation
            < float(self.config.benign_deviation_limit)
        )

        records: List[TeacherDefenseRecord] = []
        purified_knowledge = []
        for index, item in enumerate(teacher_knowledge):
            records.append(
                TeacherDefenseRecord(
                    client_id=int(item.metadata.client_id),
                    anomaly_fraction=float(
                        anomaly_mask[index].float().mean().item()
                    ),
                    mean_abs_deviation=float(
                        abs_deviation[index].mean().item()
                    ),
                    max_abs_deviation=float(
                        teacher_max_deviation[index].item()
                    ),
                    mean_suppression=float(
                        (1.0 - weights[index]).mean().item()
                    ),
                    memory_eligible=bool(eligible[index].item()),
                )
            )
            purified_knowledge.append(
                TeacherKnowledge(
                    metadata=item.metadata,
                    logits=purified[index].clone(),
                )
            )

        self._update_thresholds(abs_deviation)
        prototype_updated = self._update_prototypes(stacked, eligible)
        return DefenseResult(
            method=self.name,
            purified_knowledge=tuple(purified_knowledge),
            records=tuple(records),
            metrics={
                "warmup": 0.0,
                "prototype_updated": float(prototype_updated),
                "prototype_observations": float(
                    self._observation_count
                ),
                "threshold_mean": float(
                    self._thresholds.mean().item()
                ),
                "threshold_min": float(
                    self._thresholds.min().item()
                ),
                "threshold_max": float(
                    self._thresholds.max().item()
                ),
                "anomaly_fraction": float(
                    anomaly_mask.float().mean().item()
                ),
                "mean_suppression": float(
                    (1.0 - weights).mean().item()
                ),
                "memory_eligible_teachers": float(
                    eligible.sum().item()
                ),
            },
        )
