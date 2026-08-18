from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from admission import (
    AdmissionDecision,
    TeacherAdmissionController,
    TeacherKnowledge,
    TeacherMetadata,
)
from defense import DefenseResult, KnowledgeDefenseController
from logits_transport import ClientLogitsPacket, ServerLogitsPacket
from trainer import distill_with_logits, predict_logits
from robust_aggregation import aggregate_probabilities


class FederatedServer:
    """Server runtime that only consumes uploaded logits and public metadata."""

    def __init__(
        self,
        *,
        model: nn.Module,
        proxy_loader,
        device,
        amp: bool = False,
        strict_numeric_checks: bool = False,
    ) -> None:
        self.model = model
        self.proxy_loader = proxy_loader
        self.device = device
        self.amp = bool(amp)
        self.strict_numeric_checks = bool(strict_numeric_checks)
        self._proxy_labels = self._collect_proxy_labels(proxy_loader)
        # The admission decision and defense are sequential stages of one
        # server round.  Cache only the hard-valid calibration cohort for that
        # same round; it is never an aggregation authorization.
        self._defense_reference_round: Optional[int] = None
        self._defense_reference_ids: tuple[int, ...] = ()

    @staticmethod
    def _collect_proxy_labels(proxy_loader) -> torch.Tensor:
        labels = []
        for batch in proxy_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "Federated distillation requires labeled proxy batches."
                )
            labels.append(batch[1].detach().cpu().long().view(-1))
        if not labels:
            raise ValueError("Proxy loader contains no samples.")
        return torch.cat(labels, dim=0)

    def receive_client_uploads(
        self,
        packets: Sequence[ClientLogitsPacket],
        *,
        query_id: str,
        expected_client_ids: Sequence[int],
    ) -> Dict[int, TeacherKnowledge]:
        expected = {int(client_id) for client_id in expected_client_ids}
        received: Dict[int, TeacherKnowledge] = {}
        for packet in packets:
            client_id = int(packet.client_id)
            if packet.query_id != str(query_id):
                raise ValueError(
                    f"Client {client_id} uploaded logits for a stale query."
                )
            if client_id in received:
                raise ValueError(
                    f"Duplicate logits upload from client {client_id}."
                )
            logits = packet.decode_logits()
            if int(logits.shape[0]) != int(self._proxy_labels.numel()):
                raise ValueError(
                    f"Client {client_id} logits do not cover the proxy set."
                )
            received_at_s = float(time.monotonic())
            received[client_id] = TeacherKnowledge(
                metadata=TeacherMetadata(
                    client_id=client_id,
                    model_round=int(packet.model_round),
                    generated_at_s=float(packet.generated_at_s),
                    source_round=int(packet.source_round),
                    base_server_round=int(packet.base_server_round),
                    received_at_s=received_at_s,
                    consumed_at_s=float("nan"),
                    proxy_version=str(packet.proxy_version),
                ),
                logits=logits,
            )
        if set(received) != expected:
            raise ValueError(
                "Server must receive exactly one logits upload from every "
                "participating client."
            )
        return received

    @staticmethod
    def mark_knowledge_consumed(
        knowledge_by_client: Dict[int, TeacherKnowledge],
        *,
        consumed_at_s: Optional[float] = None,
    ) -> Dict[int, TeacherKnowledge]:
        """Attach the actual server admission/aggregation consume time."""

        consumed = float(time.monotonic() if consumed_at_s is None else consumed_at_s)
        if not torch.isfinite(torch.tensor(consumed)):
            raise ValueError("consumed_at_s must be finite.")
        return {
            int(client_id): replace(
                knowledge,
                metadata=replace(knowledge.metadata, consumed_at_s=consumed),
            )
            for client_id, knowledge in knowledge_by_client.items()
        }

    def student_proxy_logits(self) -> torch.Tensor:
        return predict_logits(
            self.model,
            self.proxy_loader,
            device=self.device,
            amp=self.amp,
        )

    def apply_admission(
        self,
        knowledge_by_client: Dict[int, TeacherKnowledge],
        *,
        current_round: int,
        controller: Optional[TeacherAdmissionController],
        student_logits: Optional[torch.Tensor] = None,
    ) -> Optional[AdmissionDecision]:
        round_number = int(current_round)
        if controller is None:
            self._defense_reference_round = round_number
            self._defense_reference_ids = tuple(
                sorted(int(client_id) for client_id in knowledge_by_client)
            )
            return None
        decision = controller.evaluate(
            teacher_knowledge=[
                knowledge_by_client[client_id]
                for client_id in sorted(knowledge_by_client)
            ],
            student_logits=(
                self.student_proxy_logits()
                if student_logits is None
                else student_logits
            ),
            proxy_labels=self._proxy_labels,
            current_round=round_number,
        )
        reference_ids = tuple(
            int(client_id)
            for client_id in (
                decision.freshness_valid_client_ids
                or decision.admitted_client_ids
            )
        )
        unknown = set(reference_ids).difference(knowledge_by_client)
        if unknown:
            raise ValueError(
                "Admission returned unknown freshness-valid reference clients: "
                f"{sorted(unknown)}"
            )
        if not set(decision.admitted_client_ids).issubset(reference_ids):
            raise ValueError(
                "Defense reference cohort must contain every admitted teacher."
            )
        self._defense_reference_round = round_number
        self._defense_reference_ids = reference_ids
        return decision

    def apply_defense(
        self,
        knowledge_by_client: Dict[int, TeacherKnowledge],
        *,
        admitted_client_ids: Sequence[int],
        current_round: int,
        controller: Optional[KnowledgeDefenseController],
        student_logits: Optional[torch.Tensor] = None,
    ) -> Optional[DefenseResult]:
        if controller is None or not admitted_client_ids:
            return None
        admitted_ids = tuple(int(client_id) for client_id in admitted_client_ids)
        admitted = [knowledge_by_client[client_id] for client_id in admitted_ids]
        reference_ids = admitted_ids
        if self._defense_reference_round == int(current_round):
            cached = tuple(
                client_id
                for client_id in self._defense_reference_ids
                if client_id in knowledge_by_client
            )
            if cached and set(admitted_ids).issubset(cached):
                reference_ids = cached
        reference = [knowledge_by_client[client_id] for client_id in reference_ids]
        result = controller.purify(
            teacher_knowledge=admitted,
            reference_knowledge=reference,
            student_logits=(
                self.student_proxy_logits()
                if student_logits is None
                else student_logits
            ),
            proxy_labels=self._proxy_labels,
            current_round=int(current_round),
        )
        returned_ids = tuple(
            int(item.metadata.client_id) for item in result.purified_knowledge
        )
        if set(returned_ids) != set(admitted_ids) or len(returned_ids) != len(admitted_ids):
            raise ValueError(
                "Defense may return only, and exactly, the admitted action cohort."
            )
        return result

    @staticmethod
    def aggregate_admitted_logits(
        knowledge_by_client: Dict[int, TeacherKnowledge],
        admitted_client_ids: Sequence[int],
    ) -> Optional[torch.Tensor]:
        admitted = [int(client_id) for client_id in admitted_client_ids]
        if not admitted:
            return None
        logits = [knowledge_by_client[client_id].logits for client_id in admitted]
        reference_shape = logits[0].shape
        if any(item.shape != reference_shape for item in logits):
            raise ValueError("Admitted client logits must have equal shapes.")
        return torch.stack(logits, dim=0).mean(dim=0)

    @staticmethod
    def aggregate_admitted_probabilities(
        knowledge_by_client: Dict[int, TeacherKnowledge],
        admitted_client_ids: Sequence[int],
        *,
        temperature: float,
        aggregation_rule: str = "mean-soft-probabilities",
        trim_fraction: float = 0.1,
        weights: Optional[Sequence[float]] = None,
    ) -> Optional[torch.Tensor]:
        admitted = [int(client_id) for client_id in admitted_client_ids]
        if not admitted:
            return None
        if float(temperature) <= 0.0:
            raise ValueError("Distillation temperature must be positive.")
        logits = [knowledge_by_client[client_id].logits for client_id in admitted]
        reference_shape = logits[0].shape
        if any(item.shape != reference_shape for item in logits):
            raise ValueError("Admitted client logits must have equal shapes.")
        return aggregate_probabilities(
            logits,
            method=str(aggregation_rule),
            temperature=float(temperature),
            trim_fraction=float(trim_fraction),
            weights=weights,
        )

    def train_from_uploaded_logits(
        self,
        target_logits: Optional[torch.Tensor],
        *,
        learning_rate: float,
        temperature: float,
    ) -> bool:
        if target_logits is None:
            return False
        distill_with_logits(
            self.model,
            self.proxy_loader,
            target_logits,
            device=self.device,
            lr=float(learning_rate),
            epochs=1,
            temperature=float(temperature),
            amp=self.amp,
            strict_numeric_checks=self.strict_numeric_checks,
        )
        return True

    def train_from_teacher_probabilities(
        self,
        target_probabilities: Optional[torch.Tensor],
        *,
        learning_rate: float,
        temperature: float,
        clean_ce_weight: float = 0.0,
    ) -> bool:
        if target_probabilities is None:
            return False
        distill_with_logits(
            self.model,
            self.proxy_loader,
            target_probabilities,
            device=self.device,
            lr=float(learning_rate),
            epochs=1,
            temperature=float(temperature),
            amp=self.amp,
            strict_numeric_checks=self.strict_numeric_checks,
            targets_are_probabilities=True,
            clean_ce_weight=float(clean_ce_weight),
        )
        return True

    def build_server_broadcast(
        self,
        *,
        current_round: int,
        query_id: str,
        student_logits: Optional[torch.Tensor] = None,
    ) -> ServerLogitsPacket:
        return ServerLogitsPacket.from_logits(
            model_round=int(current_round),
            query_id=str(query_id),
            logits=(
                self.student_proxy_logits()
                if student_logits is None
                else student_logits
            ),
        )
