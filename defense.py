from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, Sequence, Tuple

import torch

from admission import TeacherKnowledge


@dataclass(frozen=True)
class TeacherDefenseRecord:
    """Auditable prediction-level defense result for one client teacher."""

    client_id: int
    anomaly_fraction: float
    mean_abs_deviation: float
    max_abs_deviation: float
    mean_suppression: float
    memory_eligible: bool


@dataclass(frozen=True)
class DefenseResult:
    """Method-independent purified knowledge consumed by the server."""

    method: str
    purified_knowledge: Tuple[TeacherKnowledge, ...]
    records: Tuple[TeacherDefenseRecord, ...]
    metrics: Dict[str, float] = field(default_factory=dict)


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
