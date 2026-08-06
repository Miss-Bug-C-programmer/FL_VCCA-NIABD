from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, Sequence, Tuple, Union

import torch

from admission import TeacherKnowledge


MetricValue = Union[float, int, bool, str, None]


@dataclass(frozen=True)
class TeacherDefenseRecord:
    """Auditable prediction-level defense result for one client teacher."""

    client_id: int
    anomaly_fraction: float
    mean_abs_deviation: float
    max_abs_deviation: float
    mean_suppression: float
    memory_eligible: bool
    teacher_memory_score: float = float("nan")
    high_quantile_deviation: float = float("nan")
    mean_excess: float = float("nan")
    consensus_deviation: float = float("nan")


@dataclass(frozen=True)
class DefenseResult:
    """Method-independent purified knowledge consumed by the server."""

    method: str
    purified_knowledge: Tuple[TeacherKnowledge, ...]
    records: Tuple[TeacherDefenseRecord, ...]
    metrics: Dict[str, MetricValue] = field(default_factory=dict)


class KnowledgeDefenseController(Protocol):
    """Loose interface for NIABD and future logits-space defenses."""

    name: str

    def purify(
        self,
        *,
        teacher_knowledge: Sequence[TeacherKnowledge],
        student_logits: torch.Tensor,
        proxy_labels: torch.Tensor,
        current_round: int,
    ) -> DefenseResult:
        ...

    def reset(self) -> None:
        ...
