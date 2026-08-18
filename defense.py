from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence, Tuple, Union

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
    phase: str = ""
    round_risk: float = float("nan")
    risk_ema: float = float("nan")
    consensus_shift: float = float("nan")
    eligible_ratio: float = float("nan")
    trusted_memory_frozen: bool = False
    trusted_memory_updated: bool = False
    threshold_update_mode: str = ""
    reference_trusted_weight: float = float("nan")
    recovery_stable_rounds: int = 0


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
        reference_knowledge: Optional[Sequence[TeacherKnowledge]] = None,
    ) -> DefenseResult:
        """Purify the action cohort using an optional calibration cohort.

        ``teacher_knowledge`` is the only cohort whose returned packets may be
        consumed by aggregation.  ``reference_knowledge`` is observation-only:
        defenses may use it to estimate robust current statistics, but must not
        return or aggregate a reference-only packet.
        """
        ...

    def reset(self) -> None:
        ...
