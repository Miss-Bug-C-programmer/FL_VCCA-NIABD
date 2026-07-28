from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, Sequence, Tuple

import torch


@dataclass(frozen=True)
class TeacherMetadata:
    """Server-observable metadata attached to one teacher prediction update."""

    client_id: int
    model_round: int
    generated_at_s: float


@dataclass(frozen=True)
class TeacherKnowledge:
    """Prediction-layer payload uploaded by one client teacher."""

    metadata: TeacherMetadata
    logits: torch.Tensor


@dataclass(frozen=True)
class TeacherAdmissionRecord:
    """Auditable admission result for one client teacher."""

    client_id: int
    admitted: bool
    score: float
    components: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionDecision:
    """Method-independent output consumed by the distillation pipeline."""

    method: str
    threshold: float
    admitted_client_ids: Tuple[int, ...]
    rejected_client_ids: Tuple[int, ...]
    records: Tuple[TeacherAdmissionRecord, ...]


class TeacherAdmissionController(Protocol):
    """Loose interface for VCAA and future teacher-selection baselines."""

    name: str

    def evaluate(
        self,
        *,
        teacher_knowledge: Sequence[TeacherKnowledge],
        student_logits: torch.Tensor,
        proxy_labels: torch.Tensor,
        current_round: int,
    ) -> AdmissionDecision:
        ...

    def reset(self) -> None:
        ...
