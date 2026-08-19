from __future__ import annotations

import copy
import hashlib
import math
import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from attacks import (
    AttackPlan,
    BackdoorBatchPoisoner,
    evaluate_backdoor_suite,
    split_defense_diagnostics,
)
from checkpointing import restore_checkpoint
from admission import (
    AdmissionDecision,
    TeacherAdmissionController,
    TeacherKnowledge,
)
from defense import DefenseResult, KnowledgeDefenseController
from federated_client import FederatedClient
from federated_server import FederatedServer
from result_schema import VCAA_ALGORITHM_VERSION


def _freeze_state_dict(state_dict: dict) -> dict:
    frozen = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            frozen[key] = value.detach().cpu().clone()
        else:
            frozen[key] = copy.deepcopy(value)
    return frozen


def _restore_model(model: nn.Module, state: dict, device) -> None:
    model.load_state_dict(state, strict=True)
    model.to(device)


def _model_is_finite(model: nn.Module) -> bool:
    return all(
        torch.isfinite(value).all()
        for value in model.state_dict().values()
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _finite_int(value, default: int = 0) -> int:
    """Convert an optional metric to int without turning NaN into a crash."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return int(default)
    return int(number) if math.isfinite(number) else int(default)


def _forward_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def evaluate_with_loss(
    model: nn.Module,
    dataloader,
    *,
    device="cpu",
    amp: bool = False,
    strict_numeric_checks: bool = False,
) -> tuple[float, float, int]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    nonfinite_batches = 0
    amp_enabled = bool(amp) and torch.device(device).type == "cuda"

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = _forward_logits(model, images)
            else:
                logits = _forward_logits(model, images)

            if not bool(torch.isfinite(logits).all().item()):
                nonfinite_batches += 1
                if strict_numeric_checks:
                    continue
                logits = torch.nan_to_num(
                    logits,
                    nan=0.0,
                    posinf=30.0,
                    neginf=-30.0,
                ).clamp(-30.0, 30.0)
            loss = loss_fn(logits, labels)
            if not bool(torch.isfinite(loss).item()):
                nonfinite_batches += 1
                continue

            batch_size = int(labels.numel())
            total += batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            loss_sum += float(loss.item()) * batch_size

    return (
        float(correct) / float(max(total, 1)),
        float(loss_sum) / float(max(total, 1)),
        int(nonfinite_batches),
    )


def _decision_metrics(
    decision: Optional[AdmissionDecision],
    *,
    num_clients: int,
) -> Dict[str, object]:
    if decision is None:
        return {
            "method": "none",
            "vcaa_algorithm_version": "none",
            "result_schema_version": "fedagg-results-v3",
            "vcaa_nonfinite_policy": "none",
            "vcaa_history_size": None,
            "threshold": float("nan"),
            "admitted": int(num_clients),
            "rejected": 0,
            "utilization": 1.0,
            "score_mean": 0.0,
            "version_score_mean": 0.0,
            "content_score_mean": 0.0,
            "proxy_accuracy_mean": 0.0,
            "entropy_mean": 0.0,
            "kl_mean": 0.0,
            "hard_rejected": 0,
            "version_rejected": 0,
            "age_rejected": 0,
            "freshness_score_mean": 0.0,
            "content_reliability_mean": 0.0,
            "aggregation_weight_mean": 0.0,
            "freshness_valid_teachers": 0,
            "effective_teacher_count": float("nan"),
            "weight_cv": float("nan"),
            "weight_total_variation_from_uniform": float("nan"),
            "content_reliability_saturation_fraction": float("nan"),
            "content_score_center": float("nan"),
            "content_score_scale": float("nan"),
            "content_threshold_role": "not_applicable",
            "content_gate_active": False,
            "content_threshold_source": "not_applicable",
            "content_history_observations": 0,
            "effective_age_half_life_s": float("nan"),
            "effective_max_knowledge_age_s": float("nan"),
            "age_scale_mode": "not_applicable",
            "vcaa_threshold_used_for_weighting": False,
            "records": [],
        }

    records = list(decision.records)

    def component_mean(key: str) -> float:
        values = [
            float(record.components[key])
            for record in records
            if key in record.components
            and isinstance(record.components[key], (int, float, bool))
            and math.isfinite(float(record.components[key]))
        ]
        return float(sum(values) / len(values)) if values else float("nan")

    def record_mean(attribute: str) -> float:
        values = [
            float(getattr(record, attribute))
            for record in records
            if math.isfinite(float(getattr(record, attribute)))
        ]
        return float(sum(values) / len(values)) if values else float("nan")

    admitted = len(decision.admitted_client_ids)
    return {
        "method": str(decision.method),
        "vcaa_algorithm_version": str(decision.algorithm_version),
        "result_schema_version": str(decision.result_schema_version),
        "vcaa_nonfinite_policy": str(decision.nonfinite_policy),
        "vcaa_history_size": int(decision.history_size),
        "threshold": float(decision.threshold),
        "admitted": int(admitted),
        "rejected": int(len(decision.rejected_client_ids)),
        "utilization": float(admitted) / float(max(num_clients, 1)),
        "score_mean": (
            float(sum(record.score for record in records) / len(records))
            if records
            else float("nan")
        ),
        "version_score_mean": component_mean("version_score"),
        "version_lag_score_mean": component_mean("version_lag_score"),
        "age_score_mean": component_mean("age_score"),
        "content_score_mean": component_mean("content_score"),
        "proxy_accuracy_mean": component_mean("proxy_accuracy"),
        "entropy_mean": component_mean("mean_entropy"),
        "kl_mean": component_mean("mean_kl"),
        "hard_rejected": sum(not record.hard_valid for record in records),
        "version_rejected": sum(
            not record.absolute_version_valid for record in records
        ),
        "age_rejected": sum(not record.age_valid for record in records),
        "freshness_score_mean": record_mean("freshness_score"),
        "content_reliability_mean": record_mean("content_reliability"),
        "aggregation_weight_mean": record_mean("aggregation_weight"),
        "freshness_valid_teachers": len(decision.freshness_valid_client_ids),
        "effective_teacher_count": float(decision.effective_teacher_count),
        "weight_cv": float(decision.weight_cv),
        "weight_total_variation_from_uniform": float(
            decision.weight_total_variation_from_uniform
        ),
        "content_reliability_saturation_fraction": float(
            decision.content_reliability_saturation_fraction
        ),
        "content_score_center": float(decision.content_score_center),
        "content_score_scale": float(decision.content_score_scale),
        "content_threshold_role": str(decision.content_threshold_role),
        "content_gate_active": bool(decision.content_gate_active),
        "content_threshold_source": str(decision.content_threshold_source),
        "content_history_observations": int(
            decision.content_history_observations
        ),
        "content_valid": sum(record.content_valid for record in records),
        "content_rejected": sum(
            record.hard_valid and not record.content_valid for record in records
        ),
        "effective_age_half_life_s": float(decision.effective_age_half_life_s),
        "effective_max_knowledge_age_s": float(
            decision.effective_max_knowledge_age_s
        ),
        "age_scale_mode": str(decision.age_scale_mode),
        "vcaa_threshold_used_for_weighting": bool(
            decision.vcaa_threshold_used_for_weighting
        ),
        "records": [
            {
                "client_id": int(record.client_id),
                "admitted": bool(record.admitted),
                "score": float(record.score),
                "hard_valid": bool(record.hard_valid),
                "hard_rejection_reason": str(record.hard_rejection_reason),
                "absolute_version_valid": bool(record.absolute_version_valid),
                "age_valid": bool(record.age_valid),
                "timestamp_valid": bool(record.timestamp_valid),
                "version_lag_score": float(record.version_lag_score),
                "age_score": float(record.age_score),
                "freshness_score": float(record.freshness_score),
                "content_valid": bool(record.content_valid),
                "content_gate_active": bool(record.content_gate_active),
                "content_rejection_reason": str(
                    record.content_rejection_reason
                ),
                "rejection_reason": str(record.rejection_reason),
                "content_reliability": float(record.content_reliability),
                "aggregation_weight": float(record.aggregation_weight),
                "content_score_center": float(record.content_score_center),
                "content_score_scale": float(record.content_score_scale),
                "content_score_z": float(record.content_score_z),
                "normalized_aggregation_weight": float(
                    record.normalized_aggregation_weight
                ),
                "effective_weight_ratio_to_uniform": float(
                    record.effective_weight_ratio_to_uniform
                ),
                "weighting_mode": str(record.weighting_mode),
                **{
                    str(key): (
                        float(value)
                        if isinstance(value, (int, float, bool))
                        else str(value)
                    )
                    for key, value in record.components.items()
                },
            }
            for record in records
        ],
    }


def _validate_decision(
    decision: AdmissionDecision,
    client_ids: Sequence[int],
) -> None:
    expected = {int(client_id) for client_id in client_ids}
    admitted = set(decision.admitted_client_ids)
    rejected = set(decision.rejected_client_ids)
    if (
        admitted.intersection(rejected)
        or len(admitted) != len(decision.admitted_client_ids)
        or len(rejected) != len(decision.rejected_client_ids)
        or admitted.union(rejected) != expected
    ):
        raise ValueError(
            "Admission decision must classify every client exactly once."
        )
    freshness_valid = set(decision.freshness_valid_client_ids)
    if not freshness_valid.issubset(expected):
        raise ValueError(
            "Admission decision contains an unknown freshness-valid client."
        )
    current_vcaa = (
        str(decision.method).lower() == "vcaa"
        and str(decision.algorithm_version) == VCAA_ALGORITHM_VERSION
    )
    if current_vcaa and not admitted.issubset(freshness_valid):
        raise ValueError(
            "Admitted teachers must be a subset of freshness-valid teachers."
        )
    if (
        current_vcaa
    ):
        record_ids = {int(record.client_id) for record in decision.records}
        if record_ids != expected:
            raise ValueError("VCAA records must cover every received teacher.")
        record_admitted = {
            int(record.client_id)
            for record in decision.records
            if bool(record.admitted)
        }
        if record_admitted != admitted:
            raise ValueError("VCAA record admission flags do not match admitted IDs.")
        if any(
            bool(record.admitted)
            and (not bool(record.hard_valid) or not bool(record.content_valid))
            for record in decision.records
        ):
            raise ValueError("VCAA admitted records must be hard- and content-valid.")
        raw_ids = set(decision.aggregation_weights)
        normalized_ids = set(decision.normalized_aggregation_weights)
        if raw_ids != admitted or normalized_ids != admitted:
            raise ValueError(
                "VCAA aggregation weights must exactly cover admitted IDs."
            )
        raw_weights = list(decision.aggregation_weights.values())
        normalized_weights = list(
            decision.normalized_aggregation_weights.values()
        )
        if any(
            not math.isfinite(float(weight)) or float(weight) <= 0.0
            for weight in raw_weights
        ):
            raise ValueError("VCAA raw aggregation weights must be finite and positive.")
        if any(
            not math.isfinite(float(weight)) or float(weight) < 0.0
            for weight in normalized_weights
        ):
            raise ValueError(
                "VCAA normalized aggregation weights must be finite and non-negative."
            )
        if raw_weights and not math.isclose(
            sum(normalized_weights), 1.0, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise ValueError(
                "VCAA normalized aggregation weights must sum to one."
            )


def _apply_optional_admission(
    *,
    server: FederatedServer,
    knowledge_by_client: Dict[int, TeacherKnowledge],
    current_round: int,
    admission_controller: Optional[TeacherAdmissionController],
    student_logits: torch.Tensor,
) -> tuple[Optional[AdmissionDecision], List[int], List[int]]:
    """Apply admission only when an admission controller is enabled.

    The disabled path is an explicit pass-through: every teacher knowledge
    packet that reached the round candidate cohort is admitted.  NIABD is a
    post-admission defense and therefore cannot reject teachers through this
    path.
    """

    candidate_ids = sorted(int(client_id) for client_id in knowledge_by_client)
    if admission_controller is None:
        return None, candidate_ids, list(candidate_ids)

    decision = server.apply_admission(
        knowledge_by_client,
        current_round=int(current_round),
        controller=admission_controller,
        student_logits=student_logits,
    )
    if decision is None:
        raise RuntimeError(
            "An enabled admission controller must return an AdmissionDecision."
        )
    _validate_decision(decision, candidate_ids)
    admitted_ids = [int(client_id) for client_id in decision.admitted_client_ids]
    freshness_valid_ids = [
        int(client_id)
        for client_id in (
            decision.freshness_valid_client_ids
            or decision.admitted_client_ids
        )
    ]
    return decision, admitted_ids, freshness_valid_ids


def _defense_metrics(
    result: Optional[DefenseResult],
    *,
    method: str,
) -> Dict[str, object]:
    if result is None:
        return {
            "method": str(method),
            "warmup": float("nan"),
            "prototype_updated": float("nan"),
            "prototype_observations": float("nan"),
            "threshold_mean": float("nan"),
            "threshold_min": float("nan"),
            "threshold_max": float("nan"),
            "anomaly_fraction": float("nan"),
            "mean_suppression": float("nan"),
            "memory_eligible_teachers": float("nan"),
            "niabd_algorithm_version": "",
            "result_schema_version": "",
            "niabd_prototype_update_reason": "",
            "niabd_memory_candidate_teachers": float("nan"),
            "niabd_teacher_score_mean": float("nan"),
            "niabd_teacher_score_median": float("nan"),
            "niabd_teacher_score_mad": float("nan"),
            "niabd_high_quantile_deviation": float("nan"),
            "niabd_mean_excess": float("nan"),
            "niabd_consensus_deviation": float("nan"),
            "niabd_current_consensus_drift": float("nan"),
            "niabd_all_ineligible_round": float("nan"),
            "niabd_consecutive_frozen_rounds": float("nan"),
            "niabd_effective_memory_weight": float("nan"),
            "niabd_eligible_teacher_observations": float("nan"),
            "niabd_memory_update_rounds": float("nan"),
            "niabd_defense_available": None,
            "niabd_purification_applied": None,
            "niabd_memory_updated": None,
            "niabd_phase": "",
            "niabd_round_risk": float("nan"),
            "niabd_risk_ema": float("nan"),
            "niabd_consensus_shift": float("nan"),
            "niabd_eligible_ratio": float("nan"),
            "niabd_trusted_memory_frozen": None,
            "niabd_trusted_memory_updated": None,
            "niabd_threshold_update_mode": "",
            "niabd_reference_trusted_weight": float("nan"),
            "niabd_recovery_stable_rounds": float("nan"),
            "teachers_purified": 0,
            "records": [],
        }
    records = [
        {
            "client_id": int(record.client_id),
            "anomaly_fraction": float(record.anomaly_fraction),
            "mean_abs_deviation": float(record.mean_abs_deviation),
            "max_abs_deviation": float(record.max_abs_deviation),
            "mean_suppression": float(record.mean_suppression),
            "memory_eligible": bool(record.memory_eligible),
            "teacher_memory_score": float(record.teacher_memory_score),
            "high_quantile_deviation": float(
                record.high_quantile_deviation
            ),
            "mean_excess": float(record.mean_excess),
            "consensus_deviation": float(record.consensus_deviation),
            "phase": str(record.phase),
            "round_risk": float(record.round_risk),
            "risk_ema": float(record.risk_ema),
            "consensus_shift": float(record.consensus_shift),
            "eligible_ratio": float(record.eligible_ratio),
            "trusted_memory_frozen": bool(record.trusted_memory_frozen),
            "trusted_memory_updated": bool(record.trusted_memory_updated),
            "threshold_update_mode": str(record.threshold_update_mode),
            "reference_trusted_weight": float(record.reference_trusted_weight),
            "recovery_stable_rounds": int(record.recovery_stable_rounds),
        }
        for record in result.records
    ]
    return {
        "method": str(result.method),
        "warmup": float(result.metrics.get("warmup", 0.0)),
        "prototype_updated": float(
            result.metrics.get("prototype_updated", 0.0)
        ),
        "prototype_observations": float(
            result.metrics.get("prototype_observations", 0.0)
        ),
        "threshold_mean": float(
            result.metrics.get("threshold_mean", 0.0)
        ),
        "threshold_min": float(
            result.metrics.get("threshold_min", 0.0)
        ),
        "threshold_max": float(
            result.metrics.get("threshold_max", 0.0)
        ),
        "anomaly_fraction": float(
            result.metrics.get("anomaly_fraction", 0.0)
        ),
        "mean_suppression": float(
            result.metrics.get("mean_suppression", 0.0)
        ),
        "memory_eligible_teachers": int(
            result.metrics.get("memory_eligible_teachers", 0.0)
        ),
        "niabd_algorithm_version": str(
            result.metrics.get("niabd_algorithm_version", "")
        ),
        "result_schema_version": str(
            result.metrics.get("result_schema_version", "")
        ),
        "niabd_prototype_update_reason": str(
            result.metrics.get("niabd_prototype_update_reason", "")
        ),
        "niabd_memory_candidate_teachers": float(
            result.metrics.get("niabd_memory_candidate_teachers", float("nan"))
        ),
        "niabd_teacher_score_mean": float(
            result.metrics.get("niabd_teacher_score_mean", float("nan"))
        ),
        "niabd_teacher_score_median": float(
            result.metrics.get("niabd_teacher_score_median", float("nan"))
        ),
        "niabd_teacher_score_mad": float(
            result.metrics.get("niabd_teacher_score_mad", float("nan"))
        ),
        "niabd_high_quantile_deviation": float(
            result.metrics.get("niabd_high_quantile_deviation", float("nan"))
        ),
        "niabd_mean_excess": float(
            result.metrics.get("niabd_mean_excess", float("nan"))
        ),
        "niabd_consensus_deviation": float(
            result.metrics.get("niabd_consensus_deviation", float("nan"))
        ),
        "niabd_current_consensus_drift": float(
            result.metrics.get("niabd_current_consensus_drift", float("nan"))
        ),
        "niabd_all_ineligible_round": float(
            result.metrics.get("niabd_all_ineligible_round", float("nan"))
        ),
        "niabd_consecutive_frozen_rounds": float(
            result.metrics.get("niabd_consecutive_frozen_rounds", float("nan"))
        ),
        "niabd_effective_memory_weight": float(
            result.metrics.get("niabd_effective_memory_weight", float("nan"))
        ),
        "niabd_eligible_teacher_observations": float(
            result.metrics.get("niabd_eligible_teacher_observations", float("nan"))
        ),
        "niabd_memory_update_rounds": float(
            result.metrics.get("niabd_memory_update_rounds", float("nan"))
        ),
        "niabd_defense_available": result.metrics.get(
            "niabd_defense_available", True
        ),
        "niabd_purification_applied": result.metrics.get(
            "niabd_purification_applied", True
        ),
        "niabd_memory_updated": result.metrics.get(
            "niabd_memory_updated", False
        ),
        "niabd_phase": result.metrics.get("niabd_phase", ""),
        "niabd_round_risk": float(result.metrics.get("niabd_round_risk", float("nan"))),
        "niabd_risk_ema": float(result.metrics.get("niabd_risk_ema", float("nan"))),
        "niabd_consensus_shift": float(result.metrics.get("niabd_consensus_shift", float("nan"))),
        "niabd_eligible_ratio": float(result.metrics.get("niabd_eligible_ratio", float("nan"))),
        "niabd_trusted_memory_frozen": result.metrics.get("niabd_trusted_memory_frozen"),
        "niabd_trusted_memory_updated": result.metrics.get("niabd_trusted_memory_updated"),
        "niabd_threshold_update_mode": str(result.metrics.get("niabd_threshold_update_mode", "")),
        "niabd_reference_trusted_weight": float(result.metrics.get("niabd_reference_trusted_weight", float("nan"))),
        "niabd_recovery_stable_rounds": float(result.metrics.get("niabd_recovery_stable_rounds", float("nan"))),
        "teachers_purified": len(result.purified_knowledge),
        "records": records,
    }


def run_fedagg_server_client(
    client_models: Sequence[nn.Module],
    server_model: nn.Module,
    dataloaders: Dict[str, object],
    *,
    device="cpu",
    local_epochs: int = 1,
    rounds: int = 3,
    learning_rate: float = 0.01,
    distill_temperature: float = 2.0,
    amp: bool = False,
    strict_numeric_checks: bool = False,
    admission_controller: Optional[TeacherAdmissionController] = None,
    defense_controller: Optional[KnowledgeDefenseController] = None,
    enable_client_distillation: bool = True,
    aggregation_rule: str = "mean-soft-probabilities",
    aggregation_trim_fraction: float = 0.1,
    clean_ce_weight: float = 0.05,
    attack_plan: Optional[AttackPlan] = None,
    enable_backdoor_diagnostics: bool = False,
    backdoor_diagnostics_dataset: str = "",
    checkpoint_callback: Optional[Callable[[int, Dict[str, object]], None]] = None,
    resume_payload: Optional[dict] = None,
) -> Dict[str, object]:
    """Train real clients through serialized proxy-logits knowledge exchange.

    Each client owns its model and private loader, performs local optimization,
    evaluates the shared proxy set, and emits a byte-serialized logits packet.
    The server receives only those packets and public metadata. It never
    dereferences a client model during admission, aggregation, or student
    distillation.
    """

    client_loaders = dataloaders.get("client")
    proxy_loader = dataloaders.get("proxy")
    test_loader = dataloaders.get("test")
    if not isinstance(client_loaders, list) or not client_loaders:
        raise ValueError(
            "dataloaders['client'] must contain at least one loader."
        )
    if proxy_loader is None:
        raise ValueError("dataloaders['proxy'] is required.")
    if test_loader is None:
        raise ValueError("dataloaders['test'] is required.")
    if len(client_models) != len(client_loaders):
        raise ValueError(
            "The number of client models must match dataloaders['client']."
        )
    if int(rounds) <= 0:
        raise ValueError("rounds must be positive.")
    if int(local_epochs) <= 0:
        raise ValueError("local_epochs must be positive.")

    device_obj = torch.device(device)
    clients = [
        FederatedClient(
            client_id=client_id,
            model=model,
            train_loader=loader,
            device=device_obj,
            amp=bool(amp),
            strict_numeric_checks=bool(strict_numeric_checks),
        )
        for client_id, (model, loader) in enumerate(
            zip(client_models, client_loaders)
        )
    ]
    server = FederatedServer(
        model=server_model,
        proxy_loader=proxy_loader,
        device=device_obj,
        amp=bool(amp),
        strict_numeric_checks=bool(strict_numeric_checks),
    )
    if attack_plan is not None and int(attack_plan.num_clients) != len(clients):
        raise ValueError(
            "Attack plan client count must match the active client set."
        )
    poisoners = {
        int(client.client_id): BackdoorBatchPoisoner(
            plan=attack_plan,
            client_id=int(client.client_id),
        )
        for client in clients
        if attack_plan is not None and attack_plan.is_malicious(client.client_id)
    }

    metrics: Dict[str, object] = {
        "topology": "server-client",
        "knowledge_interface": "serialized-proxy-logits",
        "aggregation_rule": str(aggregation_rule),
        "server_role": "global-student",
        "client_role": "local-teacher",
        "num_clients": len(clients),
        "admission_method": (
            str(admission_controller.name)
            if admission_controller is not None
            else "none"
        ),
        "vcaa_enabled": int(
            admission_controller is not None
            and str(admission_controller.name).lower() == "vcaa"
        ),
        "defense_method": (
            str(defense_controller.name)
            if defense_controller is not None
            else "none"
        ),
        "niabd_enabled": int(
            defense_controller is not None
            and str(defense_controller.name).lower() == "niabd"
        ),
        "acc_list": [],
        "loss_list": [],
        "local_train_time_s": [],
        "upload_time_s": [],
        "admission_time_s": [],
        "defense_time_s": [],
        "distill_time_s": [],
        "round_time_s": [],
        "wall_clock_time_s": [],
        "clients_trained": [],
        "client_upload_bytes": [],
        "server_broadcast_bytes": [],
        "server_client_distillations": [],
        "server_updates_from_clients": [],
        "client_reverse_distillations": [],
        "server_update_applied": [],
        "teachers_admitted": [],
        "teachers_rejected": [],
        "teacher_utilization": [],
        "admission_threshold": [],
        "admission_score_mean": [],
        "vcaa_version_score_mean": [],
        "vcaa_version_lag_score_mean": [],
        "vcaa_age_score_mean": [],
        "vcaa_content_score_mean": [],
        "vcaa_proxy_accuracy_mean": [],
        "vcaa_entropy_mean": [],
        "vcaa_kl_mean": [],
        "vcaa_hard_rejected": [],
        "vcaa_version_rejected": [],
        "vcaa_age_rejected": [],
        "vcaa_freshness_score_mean": [],
        "vcaa_content_reliability_mean": [],
        "vcaa_aggregation_weight_mean": [],
        "vcaa_freshness_valid_teachers": [],
        "vcaa_effective_teacher_count": [],
        "vcaa_weight_cv": [],
        "vcaa_weight_total_variation_from_uniform": [],
        "vcaa_content_reliability_saturation_fraction": [],
        "vcaa_content_score_center": [],
        "vcaa_content_score_scale": [],
        "vcaa_content_threshold_role": [],
        "vcaa_content_gate_active": [],
        "vcaa_content_threshold_source": [],
        "vcaa_content_history_observations": [],
        "vcaa_content_valid": [],
        "vcaa_content_rejected": [],
        "vcaa_effective_age_half_life_s": [],
        "vcaa_effective_max_knowledge_age_s": [],
        "vcaa_age_scale_mode": [],
        "vcaa_threshold_used_for_weighting": [],
        "vcaa_history_size": [],
        "teacher_admission_records": [],
        "teachers_purified": [],
        "niabd_warmup": [],
        "niabd_anomaly_fraction": [],
        "niabd_mean_suppression": [],
        "niabd_threshold_mean": [],
        "niabd_threshold_min": [],
        "niabd_threshold_max": [],
        "niabd_prototype_updated": [],
        "niabd_prototype_observations": [],
        "niabd_memory_eligible_teachers": [],
        "niabd_algorithm_version": [],
        "result_schema_version": [],
        "niabd_prototype_update_reason": [],
        "niabd_memory_candidate_teachers": [],
        "niabd_teacher_score_mean": [],
        "niabd_teacher_score_median": [],
        "niabd_teacher_score_mad": [],
        "niabd_high_quantile_deviation": [],
        "niabd_mean_excess": [],
        "niabd_consensus_deviation": [],
        "niabd_current_consensus_drift": [],
        "niabd_all_ineligible_round": [],
        "niabd_consecutive_frozen_rounds": [],
        "niabd_effective_memory_weight": [],
        "niabd_eligible_teacher_observations": [],
        "niabd_memory_update_rounds": [],
        "teacher_defense_records": [],
        "nonfinite_eval_batches": [],
        "nonfinite_distill_rollbacks": [],
        "numeric_failure_count": [],
        "student_snapshot_sha256": [],
        "student_snapshot_source_round": [],
        "transaction_id": [],
        "transaction_status": [],
        "rollback_reason": [],
        "received_teachers": [],
        "drift_recovery_candidates": [],
        "memory_update_teachers": [],
        "niabd_defense_available": [],
        "niabd_purification_applied": [],
        "niabd_memory_updated": [],
        "niabd_phase": [],
        "niabd_round_risk": [],
        "niabd_risk_ema": [],
        "niabd_consensus_shift": [],
        "niabd_eligible_ratio": [],
        "niabd_trusted_memory_frozen": [],
        "niabd_trusted_memory_updated": [],
        "niabd_threshold_update_mode": [],
        "niabd_reference_trusted_weight": [],
        "niabd_recovery_stable_rounds": [],
        "attack_type": (
            attack_plan.config.attack_type if attack_plan is not None else "none"
        ),
        "attack_plan_id": (
            attack_plan.identity if attack_plan is not None else ""
        ),
        "target_label": (
            int(attack_plan.config.target_label) if attack_plan is not None else -1
        ),
        "malicious_client_ids": (
            list(attack_plan.malicious_client_ids) if attack_plan is not None else []
        ),
        "malicious_fraction": (
            float(attack_plan.config.malicious_fraction) if attack_plan is not None else 0.0
        ),
        "poison_ratio": (
            float(attack_plan.config.poison_ratio) if attack_plan is not None else 0.0
        ),
        "attack_start_round": (
            int(attack_plan.config.attack_start_round) if attack_plan is not None else -1
        ),
        "attack_active": [],
        "poisoned_samples": [],
        "eligible_poison_samples": [],
        "basr_global": [],
        "basr_global_numerator": [],
        "basr_global_denominator": [],
        "basr_local_1": [],
        "basr_local_2": [],
        "basr_local_3": [],
        "basr_local_4": [],
        "malicious_mean_anomaly_fraction": [],
        "benign_mean_anomaly_fraction": [],
        "malicious_mean_suppression": [],
        "benign_mean_suppression": [],
        "malicious_memory_eligible_rate": [],
        "benign_memory_eligible_rate": [],
        "backdoor_client_records": [],
        "backdoor_diagnostics_enabled": int(enable_backdoor_diagnostics),
    }

    start_round_index = 0
    if resume_payload is not None:
        restore_checkpoint(
            resume_payload,
            server_model=server_model,
            client_models=client_models,
            admission_controller=admission_controller,
            defense_controller=defense_controller,
        )
        saved_metrics = resume_payload.get("metrics_state")
        if not isinstance(saved_metrics, dict):
            raise ValueError("Checkpoint metrics_state is required for resume.")
        metrics.update(saved_metrics)
        start_round_index = int(resume_payload["current_round"])
        if start_round_index >= int(rounds):
            return metrics

    wall_clock = float(
        metrics.get("wall_clock_time_s", [0.0])[-1]
        if metrics.get("wall_clock_time_s")
        else 0.0
    )
    distill_lr = max(float(learning_rate) * 0.2, 1e-4)
    for round_idx in range(start_round_index, int(rounds)):
        round_number = int(round_idx + 1)
        round_start = time.perf_counter()
        round_server_state = _freeze_state_dict(server.model.state_dict())
        round_client_states = [
            _freeze_state_dict(client.model.state_dict()) for client in clients
        ]
        round_admission_state = (
            admission_controller.snapshot_state()
            if admission_controller is not None
            and hasattr(admission_controller, "snapshot_state")
            else None
        )
        round_defense_state = (
            defense_controller.snapshot_state()
            if defense_controller is not None
            and hasattr(defense_controller, "snapshot_state")
            else None
        )

        local_start = time.perf_counter()
        backdoor_records = []
        for client in clients:
            poisoner = poisoners.get(int(client.client_id))
            if poisoner is not None:
                poisoner.start_round(round_number)
            client.train_local(
                epochs=int(local_epochs),
                learning_rate=float(learning_rate),
                batch_transform=poisoner,
                round_number=round_number,
            )
            stats = poisoner.round_stats if poisoner is not None else None
            backdoor_records.append({
                "client_id": int(client.client_id),
                "is_malicious": bool(
                    attack_plan is not None
                    and attack_plan.is_malicious(client.client_id)
                ),
                "attack_active": bool(
                    attack_plan is not None
                    and attack_plan.active_for(client.client_id, round_number)
                ),
                "poisoned_samples": int(stats.poisoned if stats is not None else 0),
                "eligible_poison_samples": int(stats.eligible if stats is not None else 0),
                "poisoned_batches": int(stats.poisoned_batches if stats is not None else 0),
                "dba_trigger_part": (
                    int(attack_plan.dba_part(client.client_id))
                    if attack_plan is not None
                    and attack_plan.config.attack_type == "dba"
                    and attack_plan.is_malicious(client.client_id)
                    else -1
                ),
            })
        local_time = time.perf_counter() - local_start

        upload_start = time.perf_counter()
        query_id = f"proxy-round-{round_number}"
        packets = []
        for client in clients:
            packets.append(
                client.upload_proxy_logits(
                    proxy_loader,
                    query_id=query_id,
                )
            )
        knowledge_by_client = server.receive_client_uploads(
            packets,
            query_id=query_id,
            expected_client_ids=[client.client_id for client in clients],
        )
        upload_time = time.perf_counter() - upload_start
        # One immutable pre-update snapshot is shared by VCAA and NIABD.
        student_snapshot = server.student_proxy_logits().detach().cpu().clone()
        student_snapshot_sha256 = _tensor_sha256(student_snapshot)
        knowledge_by_client = server.mark_knowledge_consumed(
            knowledge_by_client,
            consumed_at_s=time.monotonic(),
        )
        if (
            admission_controller is not None
            and hasattr(admission_controller, "update_runtime_timing")
        ):
            observed_ages = [
                float(item.metadata.consumed_at_s)
                - float(item.metadata.generated_at_s)
                for item in knowledge_by_client.values()
                if math.isfinite(float(item.metadata.consumed_at_s))
                and math.isfinite(float(item.metadata.generated_at_s))
            ]
            if observed_ages:
                admission_controller.update_runtime_timing(
                    statistics.median(observed_ages)
                )
        admission_start = time.perf_counter()
        decision, admitted_ids, freshness_valid_ids = _apply_optional_admission(
            server=server,
            knowledge_by_client=knowledge_by_client,
            current_round=round_number,
            admission_controller=admission_controller,
            student_logits=student_snapshot,
        )
        admission_time = time.perf_counter() - admission_start
        admission_metrics = _decision_metrics(
            decision,
            num_clients=len(clients),
        )
        defense_start = time.perf_counter()
        defense_result = server.apply_defense(
            knowledge_by_client,
            admitted_client_ids=admitted_ids,
            current_round=round_number,
            controller=defense_controller,
            student_logits=student_snapshot,
        )
        if defense_result is not None:
            purified_by_id = {
                int(item.metadata.client_id): item
                for item in defense_result.purified_knowledge
            }
            if (
                len(purified_by_id)
                != len(defense_result.purified_knowledge)
                or set(purified_by_id)
                != {
                    int(client_id)
                    for client_id in (
                        admitted_ids
                    )
                }
            ):
                raise ValueError(
                    "Defense result must return one purified packet for "
                    "every admitted teacher."
                )
            knowledge_by_client = {
                **knowledge_by_client,
                **purified_by_id,
            }
        defense_time = time.perf_counter() - defense_start
        defense_metrics = _defense_metrics(
            defense_result,
            method=(
                str(defense_controller.name)
                if defense_controller is not None
                else "none"
            ),
        )
        defense_split = split_defense_diagnostics(
            defense_metrics["records"],
            malicious_client_ids=(
                attack_plan.malicious_client_ids if attack_plan is not None else ()
            ),
        )

        distill_start = time.perf_counter()
        aggregation_ids = admitted_ids
        aggregation_weights = None
        if (
            decision is not None
            and str(decision.method).lower() == "vcaa"
            and str(decision.algorithm_version) == VCAA_ALGORITHM_VERSION
        ):
            try:
                aggregation_weights = [
                    float(decision.aggregation_weights[int(client_id)])
                    for client_id in aggregation_ids
                ]
            except KeyError as exc:
                raise ValueError(
                    "VCAA aggregation weight is missing for an aggregation ID."
                ) from exc
        aggregated_probabilities = server.aggregate_admitted_probabilities(
            knowledge_by_client,
            aggregation_ids,
            temperature=float(distill_temperature),
            aggregation_rule=str(aggregation_rule),
            trim_fraction=float(aggregation_trim_fraction),
            weights=aggregation_weights,
        )
        server_updated = server.train_from_teacher_probabilities(
            aggregated_probabilities,
            learning_rate=distill_lr,
            temperature=float(distill_temperature),
            clean_ce_weight=float(
                defense_controller.clean_ce_weight()
                if defense_controller is not None
                and hasattr(defense_controller, "clean_ce_weight")
                else clean_ce_weight
            ),
        )
        broadcast_bytes = 0
        reverse_distillations = 0
        if enable_client_distillation:
            broadcast = server.build_server_broadcast(
                current_round=round_number,
                query_id=query_id,
            )
            broadcast_bytes = int(broadcast.payload_bytes) * len(clients)
            for client in clients:
                client.distill_from_server(
                    proxy_loader,
                    broadcast,
                    learning_rate=distill_lr,
                    temperature=float(distill_temperature),
                )
                reverse_distillations += 1
        distill_time = time.perf_counter() - distill_start

        nonfinite_models: List[str] = []
        if not _model_is_finite(server.model):
            nonfinite_models.append("server")
        for client in clients:
            if not _model_is_finite(client.model):
                nonfinite_models.append(f"client_{client.client_id}")

        rollback = int(bool(nonfinite_models))
        if rollback:
            _restore_model(server.model, round_server_state, device_obj)
            for client, state in zip(clients, round_client_states):
                _restore_model(client.model, state, device_obj)
            if round_admission_state is not None:
                admission_controller.restore_state(round_admission_state)
            if round_defense_state is not None:
                defense_controller.restore_state(round_defense_state)

        accuracy, loss, nonfinite_eval = evaluate_with_loss(
            server.model,
            test_loader,
                device=device_obj,
                amp=bool(amp),
                strict_numeric_checks=bool(strict_numeric_checks),
            )
        if attack_plan is not None:
            backdoor_eval = evaluate_backdoor_suite(
                server.model,
                test_loader,
                device=device_obj,
                plan=attack_plan,
                round_number=round_number,
                amp=bool(amp),
            )
        else:
            backdoor_eval = {
                "basr_global": float("nan"),
                "basr_global_numerator": 0,
                "basr_global_denominator": 0,
                "basr_local_1": float("nan"),
                "basr_local_2": float("nan"),
                "basr_local_3": float("nan"),
                "basr_local_4": float("nan"),
            }
        if enable_backdoor_diagnostics:
            if attack_plan is None:
                raise ValueError(
                    "Backdoor diagnostics require an explicit AttackPlan."
                )
            diagnostics_by_client = {
                int(client.client_id): client.compute_backdoor_diagnostics(
                    proxy_loader,
                    plan=attack_plan,
                    dataset_name=str(backdoor_diagnostics_dataset),
                    experiment_seed=int(attack_plan.seed),
                    source_round=round_number,
                )
                for client in clients
            }
            for record in backdoor_records:
                record.update(
                    diagnostics_by_client[int(record["client_id"])]
                )
        round_time = time.perf_counter() - round_start
        wall_clock += round_time

        metrics["acc_list"].append(float(accuracy))
        metrics["loss_list"].append(float(loss))
        metrics["local_train_time_s"].append(float(local_time))
        metrics["upload_time_s"].append(float(upload_time))
        metrics["admission_time_s"].append(float(admission_time))
        metrics["defense_time_s"].append(float(defense_time))
        metrics["distill_time_s"].append(float(distill_time))
        metrics["round_time_s"].append(float(round_time))
        metrics["wall_clock_time_s"].append(float(wall_clock))
        metrics["clients_trained"].append(len(clients))
        metrics["client_upload_bytes"].append(
            int(sum(packet.payload_bytes for packet in packets))
        )
        metrics["server_broadcast_bytes"].append(int(broadcast_bytes))
        metrics["server_client_distillations"].append(len(aggregation_ids))
        metrics["server_updates_from_clients"].append(len(aggregation_ids))
        metrics["client_reverse_distillations"].append(
            int(reverse_distillations)
        )
        metrics["server_update_applied"].append(int(server_updated))
        metrics["teachers_admitted"].append(
            int(admission_metrics["admitted"])
        )
        metrics["teachers_rejected"].append(
            int(admission_metrics["rejected"])
        )
        metrics["teacher_utilization"].append(
            float(admission_metrics["utilization"])
        )
        metrics["admission_threshold"].append(
            float(admission_metrics["threshold"])
        )
        metrics["admission_score_mean"].append(
            float(admission_metrics["score_mean"])
        )
        metrics["vcaa_version_score_mean"].append(
            float(admission_metrics["version_score_mean"])
        )
        metrics["vcaa_version_lag_score_mean"].append(
            float(admission_metrics.get("version_lag_score_mean", float("nan")))
        )
        metrics["vcaa_age_score_mean"].append(
            float(admission_metrics.get("age_score_mean", float("nan")))
        )
        metrics["vcaa_content_score_mean"].append(
            float(admission_metrics["content_score_mean"])
        )
        metrics["vcaa_proxy_accuracy_mean"].append(
            float(admission_metrics["proxy_accuracy_mean"])
        )
        metrics["vcaa_entropy_mean"].append(
            float(admission_metrics["entropy_mean"])
        )
        metrics["vcaa_kl_mean"].append(
            float(admission_metrics["kl_mean"])
        )
        metrics["vcaa_hard_rejected"].append(
            int(admission_metrics.get("hard_rejected", 0))
        )
        metrics["vcaa_version_rejected"].append(
            int(admission_metrics.get("version_rejected", 0))
        )
        metrics["vcaa_age_rejected"].append(
            int(admission_metrics.get("age_rejected", 0))
        )
        metrics["vcaa_freshness_score_mean"].append(
            float(admission_metrics.get("freshness_score_mean", 0.0))
        )
        metrics["vcaa_content_reliability_mean"].append(
            float(admission_metrics.get("content_reliability_mean", 0.0))
        )
        metrics["vcaa_aggregation_weight_mean"].append(
            float(admission_metrics.get("aggregation_weight_mean", 0.0))
        )
        metrics["vcaa_freshness_valid_teachers"].append(
            int(admission_metrics.get("freshness_valid_teachers", 0))
        )
        metrics["vcaa_effective_teacher_count"].append(
            float(admission_metrics.get("effective_teacher_count", float("nan")))
        )
        metrics["vcaa_weight_cv"].append(
            float(admission_metrics.get("weight_cv", float("nan")))
        )
        metrics["vcaa_weight_total_variation_from_uniform"].append(
            float(
                admission_metrics.get(
                    "weight_total_variation_from_uniform", float("nan")
                )
            )
        )
        metrics["vcaa_content_reliability_saturation_fraction"].append(
            float(
                admission_metrics.get(
                    "content_reliability_saturation_fraction", float("nan")
                )
            )
        )
        metrics["vcaa_content_score_center"].append(
            float(admission_metrics.get("content_score_center", float("nan")))
        )
        metrics["vcaa_content_score_scale"].append(
            float(admission_metrics.get("content_score_scale", float("nan")))
        )
        metrics["vcaa_content_threshold_role"].append(
            str(admission_metrics.get("content_threshold_role", "not_applicable"))
        )
        metrics["vcaa_content_gate_active"].append(
            bool(admission_metrics.get("content_gate_active", False))
        )
        metrics["vcaa_content_threshold_source"].append(
            str(admission_metrics.get("content_threshold_source", "not_applicable"))
        )
        metrics["vcaa_content_history_observations"].append(
            int(admission_metrics.get("content_history_observations", 0))
        )
        metrics["vcaa_content_valid"].append(
            int(admission_metrics.get("content_valid", 0))
        )
        metrics["vcaa_content_rejected"].append(
            int(admission_metrics.get("content_rejected", 0))
        )
        metrics["vcaa_effective_age_half_life_s"].append(
            float(admission_metrics.get("effective_age_half_life_s", float("nan")))
        )
        metrics["vcaa_effective_max_knowledge_age_s"].append(
            float(
                admission_metrics.get(
                    "effective_max_knowledge_age_s", float("nan")
                )
            )
        )
        metrics["vcaa_age_scale_mode"].append(
            str(admission_metrics.get("age_scale_mode", "not_applicable"))
        )
        metrics["vcaa_threshold_used_for_weighting"].append(
            bool(admission_metrics.get("vcaa_threshold_used_for_weighting", False))
        )
        metrics["vcaa_history_size"].append(
            admission_metrics.get("vcaa_history_size")
        )
        metrics["teacher_admission_records"].append(
            admission_metrics["records"]
        )
        metrics["teachers_purified"].append(
            int(defense_metrics["teachers_purified"])
        )
        metrics["niabd_warmup"].append(
            float(defense_metrics["warmup"])
        )
        metrics["niabd_anomaly_fraction"].append(
            float(defense_metrics["anomaly_fraction"])
        )
        metrics["niabd_mean_suppression"].append(
            float(defense_metrics["mean_suppression"])
        )
        metrics["niabd_threshold_mean"].append(
            float(defense_metrics["threshold_mean"])
        )
        metrics["niabd_threshold_min"].append(
            float(defense_metrics["threshold_min"])
        )
        metrics["niabd_threshold_max"].append(
            float(defense_metrics["threshold_max"])
        )
        metrics["niabd_prototype_updated"].append(
            float(defense_metrics["prototype_updated"])
        )
        metrics["niabd_prototype_observations"].append(
            float(defense_metrics["prototype_observations"])
        )
        metrics["niabd_memory_eligible_teachers"].append(
            float(defense_metrics["memory_eligible_teachers"])
        )
        for key in (
            "niabd_algorithm_version",
            "result_schema_version",
            "niabd_prototype_update_reason",
            "niabd_memory_candidate_teachers",
            "niabd_teacher_score_mean",
            "niabd_teacher_score_median",
            "niabd_teacher_score_mad",
            "niabd_high_quantile_deviation",
            "niabd_mean_excess",
            "niabd_consensus_deviation",
            "niabd_current_consensus_drift",
            "niabd_all_ineligible_round",
            "niabd_consecutive_frozen_rounds",
            "niabd_effective_memory_weight",
            "niabd_eligible_teacher_observations",
            "niabd_memory_update_rounds",
            "niabd_phase",
            "niabd_round_risk",
            "niabd_risk_ema",
            "niabd_consensus_shift",
            "niabd_eligible_ratio",
            "niabd_trusted_memory_frozen",
            "niabd_trusted_memory_updated",
            "niabd_threshold_update_mode",
            "niabd_reference_trusted_weight",
            "niabd_recovery_stable_rounds",
        ):
            metrics[key].append(defense_metrics.get(key))
        metrics["teacher_defense_records"].append(
            defense_metrics["records"]
        )
        metrics["nonfinite_eval_batches"].append(int(nonfinite_eval))
        metrics["nonfinite_distill_rollbacks"].append(int(rollback))
        metrics["numeric_failure_count"].append(
            float(nonfinite_eval + rollback)
        )
        metrics["student_snapshot_sha256"].append(student_snapshot_sha256)
        metrics["student_snapshot_source_round"].append(round_number - 1)
        metrics["transaction_id"].append(
            f"sync-{round_number}-{student_snapshot_sha256[:16]}"
        )
        metrics["transaction_status"].append("committed")
        metrics["rollback_reason"].append(
            "nonfinite_model" if rollback else None
        )
        metrics["received_teachers"].append(len(packets))
        metrics["drift_recovery_candidates"].append(
            _finite_int(
                defense_metrics.get("niabd_current_consensus_drift", 0.0)
                and defense_metrics.get("niabd_memory_candidate_teachers", 0.0)
                or 0
            )
        )
        metrics["memory_update_teachers"].append(
            _finite_int(
                defense_metrics.get("niabd_memory_candidate_teachers", 0)
                if defense_metrics.get("prototype_updated", 0.0)
                else 0
            )
        )
        metrics["niabd_defense_available"].append(
            defense_metrics.get("niabd_defense_available", None)
        )
        metrics["niabd_purification_applied"].append(
            defense_metrics.get("niabd_purification_applied", None)
        )
        metrics["niabd_memory_updated"].append(
            defense_metrics.get("niabd_memory_updated", None)
        )
        metrics["attack_active"].append(
            int(attack_plan.config.active(round_number))
            if attack_plan is not None else 0
        )
        metrics["poisoned_samples"].append(
            int(sum(record["poisoned_samples"] for record in backdoor_records))
        )
        metrics["eligible_poison_samples"].append(
            int(sum(record["eligible_poison_samples"] for record in backdoor_records))
        )
        for key in (
            "basr_global",
            "basr_local_1",
            "basr_local_2",
            "basr_local_3",
            "basr_local_4",
        ):
            metrics[key].append(float(backdoor_eval[key]))
        metrics["basr_global_numerator"].append(
            int(backdoor_eval["basr_global_numerator"])
        )
        metrics["basr_global_denominator"].append(
            int(backdoor_eval["basr_global_denominator"])
        )
        for key, value in defense_split.items():
            metrics[key].append(float(value))
        metrics["backdoor_client_records"].append(backdoor_records)

        if checkpoint_callback is not None:
            checkpoint_callback(int(round_number), metrics)

        print(
            f"[Round {round_number}] topology=server-client "
            f"interface=serialized-proxy-logits clients={len(clients)} "
            f"admitted={len(admitted_ids)} acc={accuracy:.4f} "
            f"defense={defense_metrics['method']} "
            f"loss={loss:.4f} local={local_time:.2f}s "
            f"upload={upload_time:.2f}s admission={admission_time:.2f}s "
            f"niabd={defense_time:.2f}s distill={distill_time:.2f}s "
            f"attack={metrics['attack_type']} "
            f"poisoned={metrics['poisoned_samples'][-1]} "
            f"basr={metrics['basr_global'][-1]:.4f}"
            f"{' rollback=1' if rollback else ''}"
        )

    return metrics
