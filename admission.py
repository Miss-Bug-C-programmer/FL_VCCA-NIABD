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
    source_round: int = -1
    base_server_round: int = -1
    received_at_s: float = float("nan")
    consumed_at_s: float = float("nan")
    proxy_version: str = ""


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
    components: Dict[str, float | bool | str | None] = field(
        default_factory=dict
    )
    # ``admitted`` is the compatibility name used by the existing result
    # tables.  The fields below make the two VCAA stages explicit: freshness
    # is a hard gate while content is a soft reliability signal.
    hard_valid: bool = True
    hard_rejection_reason: str = ""
    absolute_version_valid: bool = True
    age_valid: bool = True
    freshness_score: float = float("nan")
    content_reliability: float = float("nan")
    aggregation_weight: float = float("nan")


@dataclass(frozen=True)
class AdmissionDecision:
    """Method-independent output consumed by the distillation pipeline."""

    method: str
    threshold: float
    admitted_client_ids: Tuple[int, ...]
    rejected_client_ids: Tuple[int, ...]
    records: Tuple[TeacherAdmissionRecord, ...]
    algorithm_version: str = "none"
    result_schema_version: str = "fedagg-results-v3"
    nonfinite_policy: str = "fail_closed"
    history_size: int = 0
    freshness_valid_client_ids: Tuple[int, ...] = ()
    aggregation_weights: Dict[int, float] = field(default_factory=dict)


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
