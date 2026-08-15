from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

from attacks import AttackConfig, AttackPlan
from checkpointing import (
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint_atomic,
)
from data_utils import (
    build_federated_data_plan,
    build_server_dataloaders_from_plan,
    cleanup_dataloaders,
    get_dataloaders,
)
from device_utils import (
    default_main_device,
    normalize_device,
    supports_pin_memory,
    use_amp_for_device,
)
from model_factory import build_model, build_models, dataset_spec
from model_factory import architecture_assignment_hash, model_parameter_count
from federated_runtime import run_fedagg_server_client
from niabd import NIABDConfig, NeuroInspiredAdaptiveBackdoorDefense
from process_runtime import (
    ProcessRuntimeConfig,
    run_fedagg_server_client_process_async,
)
from runtime_trace import (
    RuntimeTrace,
    generate_runtime_trace,
    load_runtime_profile,
)
from vcaa import VCAAConfig, VersionContentAwareAdmission
from result_schema import (
    AGGREGATION_ALGORITHM_VERSION,
    NIABD_ALGORITHM_VERSION,
    RESULT_SCHEMA_VERSION,
    VCAA_ALGORITHM_VERSION,
    write_schema,
)


PROCESS_RUNTIME = "process-semi-async"
PROCESS_ONLY_ROUND_FIELDS = (
    "aggregation_time_s",
    "client_wire_bytes",
    "selected_clients",
    "dispatched_clients",
    "busy_skipped_clients",
    "offline_clients",
    "packets_consumed",
    "fresh_packets",
    "mild_stale_packets",
    "moderate_stale_packets",
    "severe_stale_packets",
    "mean_version_lag",
    "max_version_lag",
    "mean_knowledge_age_s",
    "max_knowledge_age_s",
    "upload_attempt_drop_count",
    "rpc_timeout_count",
    "retry_count",
    "quorum_required",
    "quorum_reached",
    "soft_deadline_s",
    "hard_deadline_s",
)

ADMISSION_COLUMNS = (
    "run_uid", "dataset", "seed", "round", "topology", "num_clients",
    "partition_scheme", "admission_method", "client_id", "admitted",
    "score", "version_score", "age_seconds", "model_round",
    "minimum_accepted_round", "proxy_accuracy", "mean_entropy", "mean_kl",
    "num_classes", "accuracy_term", "entropy_term", "divergence_term",
    "content_score", "task_id", "packet_id", "source_round",
    "consumed_round", "version_lag", "knowledge_age_s",
    "vcaa_algorithm_version", "vcaa_nonfinite_policy",
    "result_schema_version", "vcaa_history_size", "received_at_s",
    "consumed_at_s", "proxy_version",
    "hard_valid", "hard_rejection_reason", "absolute_version_valid",
    "age_valid", "freshness_score", "content_reliability",
    "aggregation_weight", "transport_age_s",
    "queue_age_s", "consensus_divergence", "entropy_deviation",
    "strategy", "attack_type",
)

DEFENSE_COLUMNS = (
    "run_uid", "dataset", "seed", "round", "topology", "num_clients",
    "partition_scheme", "defense_method", "client_id", "anomaly_fraction",
    "mean_abs_deviation", "max_abs_deviation", "mean_suppression",
    "memory_eligible", "teacher_memory_score", "high_quantile_deviation",
    "mean_excess", "consensus_deviation", "niabd_algorithm_version",
    "result_schema_version", "niabd_prototype_update_reason", "task_id",
    "packet_id", "source_round", "consumed_round", "version_lag",
    "niabd_defense_available", "niabd_purification_applied",
    "niabd_memory_updated", "memory_candidate_teachers",
    "normal_eligible_teachers", "memory_update_teachers",
    "niabd_observations", "phase", "round_risk", "risk_ema",
    "consensus_shift", "eligible_ratio", "trusted_memory_frozen",
    "trusted_memory_updated", "threshold_update_mode",
    "reference_trusted_weight", "recovery_stable_rounds",
    "strategy", "attack_type",
)

RUNTIME_EVENT_COLUMNS = (
    "run_uid", "dataset", "seed", "runtime", "strategy", "topology",
    "num_clients", "partition_scheme", "server_device", "client_device",
    "result_schema_version", "vcaa_algorithm_version",
    "aggregation_algorithm_version", "run_class", "attack_condition",
    "client_id", "client_pid", "task_id", "packet_id", "payload_sha256",
    "inference_sha256", "source_round", "base_server_round",
    "receive_server_round", "consumed_round", "local_model_version",
    "dispatch_at_s", "dispatched_at_s", "compute_started_at_s",
    "compute_finished_at_s", "generated_at_s",
    "first_upload_attempt_at_s", "received_at_s", "consumed_at_s",
    "actual_compute_time_s", "proxy_inference_time_s",
    "injected_compute_delay_s", "total_compute_phase_s",
    "injected_upload_delay_s", "knowledge_age_s", "transport_age_s",
    "version_lag", "base_version_lag", "upload_attempts",
    "upload_attempt_drop_count", "rpc_timeout_count", "retry_count",
    "duplicate_receive_count", "rpc_elapsed_s", "payload_bytes",
    "wire_bytes", "logits_dtype", "logits_shape", "proxy_version",
    "local_train_count", "predict_logits_calls", "transport_status",
    "rpc_accept_status", "vcaa_version_score", "vcaa_content_score",
    "vcaa_final_score", "vcaa_threshold", "proxy_accuracy",
    "mean_entropy", "mean_kl", "admitted", "niabd_anomaly_fraction",
    "niabd_mean_suppression", "niabd_teacher_memory_score",
    "niabd_high_quantile_deviation", "niabd_mean_excess",
    "niabd_consensus_deviation", "niabd_memory_eligible",
    "niabd_algorithm_version",
    "niabd_prototype_update_reason", "is_malicious", "attack_active",
    "poisoned_samples", "eligible_poison_samples", "poisoned_batches",
    "dba_trigger_part", "attack_stats_missing", "phase", "round_risk",
    "risk_ema", "consensus_shift", "eligible_ratio",
    "trusted_memory_frozen", "trusted_memory_updated",
    "threshold_update_mode", "reference_trusted_weight",
    "recovery_stable_rounds", "vcaa_hard_valid",
    "vcaa_hard_rejection_reason", "vcaa_absolute_version_valid",
    "vcaa_age_valid", "vcaa_freshness_score",
    "vcaa_content_reliability", "vcaa_aggregation_weight",
    "vcaa_consensus_divergence", "vcaa_entropy_deviation",
)

BACKDOOR_COLUMNS = (
    "run_uid", "dataset", "seed", "round", "runtime", "strategy",
    "attack_type", "attack_plan_id", "target_label", "topology",
    "num_clients", "partition_scheme", "client_id", "task_id", "packet_id",
    "source_round", "consumed_round", "version_lag", "is_malicious",
    "attack_active", "poisoned_samples", "eligible_poison_samples",
    "poisoned_batches", "dba_trigger_part", "attack_stats_missing",
    "admitted", "admission_score", "niabd_anomaly_fraction",
    "niabd_mean_abs_deviation", "niabd_max_abs_deviation",
    "niabd_mean_suppression", "niabd_teacher_memory_score",
    "niabd_high_quantile_deviation", "niabd_mean_excess",
    "niabd_consensus_deviation", "niabd_memory_eligible",
    "diagnostic_scope",
    "diagnostic_usage", "diagnostic_reporter_trust", "diagnostic_seed",
    "diagnostic_proxy_samples", "clean_proxy_target_probability",
    "triggered_proxy_target_probability",
    "clean_trigger_logit_l1_deviation",
    "clean_trigger_logit_l2_deviation",
    "clean_trigger_prediction_flip_rate",
)


def _git_reproducibility_metadata() -> dict[str, str | bool]:
    """Collect immutable run metadata without making git a runtime dependency."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_commit_sha": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit_sha": "unavailable", "git_dirty": "unavailable"}


def _parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return [int(item) for item in items]


def _parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _metric(metrics, key: str, round_idx: int, default=0.0):
    values = metrics.get(key, [])
    if not isinstance(values, list) or round_idx >= len(values):
        return default
    return values[round_idx]


def _process_metric(metrics, key: str, round_idx: int):
    """Return measured process data, or NaN when the runtime cannot emit it."""

    if str(metrics.get("runtime", "sync")).lower() != PROCESS_RUNTIME:
        return np.nan
    return _metric(metrics, key, round_idx, np.nan)


def _lineage_versions(metrics: dict) -> dict[str, str]:
    """Return one immutable algorithm lineage for the exported run rows."""

    strategy = str(metrics.get("strategy", "baseline")).lower()
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "vcaa_algorithm_version": (
            VCAA_ALGORITHM_VERSION
            if "vcaa" in strategy
            else "none"
        ),
        "niabd_algorithm_version": (
            NIABD_ALGORITHM_VERSION
            if "niabd" in strategy
            else "none"
        ),
        "aggregation_algorithm_version": AGGREGATION_ALGORITHM_VERSION,
    }


def _round_rows(
    metrics,
    *,
    run_uid: str,
    dataset_name: str,
    seed: int,
    num_clients: int,
    partition_scheme: str,
) -> Iterable[dict]:
    rounds = len(metrics.get("acc_list", []))
    lineage = _lineage_versions(metrics)
    for round_idx in range(rounds):
        row_niabd_version = _metric(
            metrics, "niabd_algorithm_version", round_idx, None
        )
        if row_niabd_version in {None, "", "nan"}:
            row_niabd_version = lineage["niabd_algorithm_version"]
        row_schema_version = _metric(
            metrics, "result_schema_version", round_idx, None
        )
        if row_schema_version in {None, "", "nan"}:
            row_schema_version = lineage["result_schema_version"]
        yield {
            "run_uid": run_uid,
            "dataset": dataset_name,
            "seed": int(seed),
            "round": int(round_idx + 1),
            "runtime": str(metrics.get("runtime", "sync")),
            "strategy": str(metrics.get("strategy", "baseline")),
            "result_schema_version": lineage["result_schema_version"],
            "vcaa_algorithm_version": lineage["vcaa_algorithm_version"],
            "niabd_algorithm_version": lineage["niabd_algorithm_version"],
            "aggregation_algorithm_version": lineage[
                "aggregation_algorithm_version"
            ],
            "run_class": str(metrics.get("run_class", "smoke")),
            "attack_condition": str(
                metrics.get(
                    "attack_condition",
                    "clean"
                    if str(metrics.get("attack_type", "none")) == "none"
                    else "attacked",
                )
            ),
            "topology": "server-client",
            "server_role": "global-student",
            "client_role": "local-teacher",
            "server_model": str(metrics.get("server_model", "resnet18")),
            "client_model": str(metrics.get("client_model", "resnet18")),
            "architecture_assignment_hash": str(
                metrics.get("architecture_assignment_hash", "")
            ),
            "server_parameter_count": int(
                metrics.get("server_parameter_count", 0)
            ),
            "client_parameter_count": int(
                metrics.get("client_parameter_count", 0)
            ),
            "server_device": str(metrics.get("server_device", "")),
            "client_device": str(metrics.get("client_device", "")),
            "num_clients": int(num_clients),
            "partition_scheme": str(partition_scheme),
            "knowledge_interface": str(
                metrics.get(
                    "knowledge_interface",
                    "serialized-proxy-logits",
                )
            ),
            "aggregation_rule": str(
                metrics.get(
                    "aggregation_rule",
                    "mean-soft-probabilities",
                )
            ),
            "vcaa_enabled": int(metrics.get("vcaa_enabled", 0)),
            "admission_method": str(
                metrics.get("admission_method", "none")
            ),
            "niabd_enabled": int(metrics.get("niabd_enabled", 0)),
            "defense_method": str(
                metrics.get("defense_method", "none")
            ),
            "attack_type": str(metrics.get("attack_type", "none")),
            "attack_plan_id": str(metrics.get("attack_plan_id", "")),
            "target_label": int(metrics.get("target_label", -1)),
            "malicious_fraction": float(metrics.get("malicious_fraction", 0.0)),
            "poison_ratio": float(metrics.get("poison_ratio", 0.0)),
            "attack_start_round": int(metrics.get("attack_start_round", -1)),
            "attack_active": int(_metric(metrics, "attack_active", round_idx, 0)),
            "poisoned_samples": int(_metric(metrics, "poisoned_samples", round_idx, 0)),
            "eligible_poison_samples": int(
                _metric(metrics, "eligible_poison_samples", round_idx, 0)
            ),
            "attack_stats_missing_count": int(
                _metric(metrics, "attack_stats_missing_count", round_idx, 0)
            ),
            "basr_global": float(
                _metric(metrics, "basr_global", round_idx, np.nan)
            ),
            "basr_global_numerator": int(
                _metric(metrics, "basr_global_numerator", round_idx, 0)
            ),
            "basr_global_denominator": int(
                _metric(metrics, "basr_global_denominator", round_idx, 0)
            ),
            "basr_local_1": float(
                _metric(metrics, "basr_local_1", round_idx, np.nan)
            ),
            "basr_local_2": float(
                _metric(metrics, "basr_local_2", round_idx, np.nan)
            ),
            "basr_local_3": float(
                _metric(metrics, "basr_local_3", round_idx, np.nan)
            ),
            "basr_local_4": float(
                _metric(metrics, "basr_local_4", round_idx, np.nan)
            ),
            "malicious_mean_anomaly_fraction": float(
                _metric(
                    metrics,
                    "malicious_mean_anomaly_fraction",
                    round_idx,
                    np.nan,
                )
            ),
            "benign_mean_anomaly_fraction": float(
                _metric(
                    metrics,
                    "benign_mean_anomaly_fraction",
                    round_idx,
                    np.nan,
                )
            ),
            "malicious_mean_suppression": float(
                _metric(
                    metrics, "malicious_mean_suppression", round_idx, np.nan
                )
            ),
            "benign_mean_suppression": float(
                _metric(
                    metrics, "benign_mean_suppression", round_idx, np.nan
                )
            ),
            "malicious_memory_eligible_rate": float(
                _metric(
                    metrics,
                    "malicious_memory_eligible_rate",
                    round_idx,
                    np.nan,
                )
            ),
            "benign_memory_eligible_rate": float(
                _metric(
                    metrics,
                    "benign_memory_eligible_rate",
                    round_idx,
                    np.nan,
                )
            ),
            "accuracy": float(_metric(metrics, "acc_list", round_idx)),
            "loss": float(_metric(metrics, "loss_list", round_idx)),
            "local_train_time_s": float(
                _metric(metrics, "local_train_time_s", round_idx)
            ),
            "upload_time_s": float(
                _metric(metrics, "upload_time_s", round_idx)
            ),
            "admission_time_s": float(
                _metric(metrics, "admission_time_s", round_idx)
            ),
            "defense_time_s": float(
                _metric(metrics, "defense_time_s", round_idx)
            ),
            "distill_time_s": float(
                _metric(metrics, "distill_time_s", round_idx)
            ),
            "round_time_s": float(_metric(metrics, "round_time_s", round_idx)),
            "wall_clock_time_s": float(
                _metric(metrics, "wall_clock_time_s", round_idx)
            ),
            "clients_trained": int(
                _metric(metrics, "clients_trained", round_idx)
            ),
            "server_client_distillations": int(
                _metric(metrics, "server_client_distillations", round_idx)
            ),
            "server_updates_from_clients": int(
                _metric(metrics, "server_updates_from_clients", round_idx)
            ),
            "client_upload_bytes": int(
                _metric(metrics, "client_upload_bytes", round_idx)
            ),
            "server_broadcast_bytes": int(
                _metric(metrics, "server_broadcast_bytes", round_idx)
            ),
            "client_reverse_distillations": int(
                _metric(
                    metrics,
                    "client_reverse_distillations",
                    round_idx,
                )
            ),
            "server_update_applied": int(
                _metric(metrics, "server_update_applied", round_idx)
            ),
            "teachers_admitted": int(
                _metric(metrics, "teachers_admitted", round_idx)
            ),
            "admitted_teachers": int(
                _metric(metrics, "teachers_admitted", round_idx)
            ),
            "teachers_rejected": int(
                _metric(metrics, "teachers_rejected", round_idx)
            ),
            "teacher_utilization": float(
                _metric(metrics, "teacher_utilization", round_idx)
            ),
            "admission_threshold": float(
                _metric(metrics, "admission_threshold", round_idx)
            ),
            "admission_score_mean": float(
                _metric(metrics, "admission_score_mean", round_idx)
            ),
            "vcaa_version_score_mean": float(
                _metric(metrics, "vcaa_version_score_mean", round_idx)
            ),
            "vcaa_content_score_mean": float(
                _metric(metrics, "vcaa_content_score_mean", round_idx)
            ),
            "vcaa_proxy_accuracy_mean": float(
                _metric(metrics, "vcaa_proxy_accuracy_mean", round_idx)
            ),
            "vcaa_entropy_mean": float(
                _metric(metrics, "vcaa_entropy_mean", round_idx)
            ),
            "vcaa_kl_mean": float(
                _metric(metrics, "vcaa_kl_mean", round_idx)
            ),
            "vcaa_history_size": _metric(
                metrics, "vcaa_history_size", round_idx, None
            ),
            "teachers_purified": int(
                _metric(metrics, "teachers_purified", round_idx)
            ),
            "niabd_warmup": float(
                _metric(metrics, "niabd_warmup", round_idx)
            ),
            "niabd_anomaly_fraction": float(
                _metric(metrics, "niabd_anomaly_fraction", round_idx)
            ),
            "niabd_mean_suppression": float(
                _metric(metrics, "niabd_mean_suppression", round_idx)
            ),
            "niabd_threshold_mean": float(
                _metric(metrics, "niabd_threshold_mean", round_idx)
            ),
            "niabd_threshold_min": float(
                _metric(metrics, "niabd_threshold_min", round_idx)
            ),
            "niabd_threshold_max": float(
                _metric(metrics, "niabd_threshold_max", round_idx)
            ),
            "niabd_prototype_updated": float(
                _metric(metrics, "niabd_prototype_updated", round_idx)
            ),
            "niabd_prototype_observations": float(
                _metric(
                    metrics,
                    "niabd_prototype_observations",
                    round_idx,
                )
            ),
            "niabd_memory_eligible_teachers": float(
                _metric(
                    metrics,
                    "niabd_memory_eligible_teachers",
                    round_idx,
                    np.nan,
                )
            ),
            "niabd_algorithm_version": str(
                row_niabd_version
            ),
            "result_schema_version": str(
                row_schema_version
            ),
            "niabd_prototype_update_reason": str(
                _metric(
                    metrics,
                    "niabd_prototype_update_reason",
                    round_idx,
                    "",
                )
            ),
            "niabd_memory_candidate_teachers": float(
                _metric(
                    metrics,
                    "niabd_memory_candidate_teachers",
                    round_idx,
                    np.nan,
                )
            ),
            "memory_candidate_teachers": _metric(
                metrics, "niabd_memory_candidate_teachers", round_idx, None
            ),
            "normal_eligible_teachers": _metric(
                metrics, "niabd_memory_eligible_teachers", round_idx, None
            ),
            "drift_recovery_candidates": _metric(
                metrics, "drift_recovery_candidates", round_idx, None
            ),
            "memory_update_teachers": _metric(
                metrics, "memory_update_teachers", round_idx, None
            ),
            "niabd_memory_update_reason": _metric(
                metrics,
                "niabd_prototype_update_reason",
                round_idx,
                None,
            ),
            "niabd_observations": _metric(
                metrics, "niabd_eligible_teacher_observations", round_idx, None
            ),
            "niabd_defense_available": _metric(
                metrics, "niabd_defense_available", round_idx, None
            ),
            "niabd_purification_applied": _metric(
                metrics, "niabd_purification_applied", round_idx, None
            ),
            "niabd_memory_updated": _metric(
                metrics, "niabd_memory_updated", round_idx, None
            ),
            "niabd_teacher_score_mean": float(
                _metric(metrics, "niabd_teacher_score_mean", round_idx, np.nan)
            ),
            "niabd_teacher_score_median": float(
                _metric(
                    metrics,
                    "niabd_teacher_score_median",
                    round_idx,
                    np.nan,
                )
            ),
            "niabd_teacher_score_mad": float(
                _metric(metrics, "niabd_teacher_score_mad", round_idx, np.nan)
            ),
            "niabd_high_quantile_deviation": float(
                _metric(
                    metrics,
                    "niabd_high_quantile_deviation",
                    round_idx,
                    np.nan,
                )
            ),
            "niabd_mean_excess": float(
                _metric(metrics, "niabd_mean_excess", round_idx, np.nan)
            ),
            "niabd_consensus_deviation": float(
                _metric(
                    metrics,
                    "niabd_consensus_deviation",
                    round_idx,
                    np.nan,
                )
            ),
            "niabd_current_consensus_drift": float(
                _metric(
                    metrics,
                    "niabd_current_consensus_drift",
                    round_idx,
                    np.nan,
                )
            ),
            "niabd_all_ineligible_round": float(
                _metric(
                    metrics,
                    "niabd_all_ineligible_round",
                    round_idx,
                    np.nan,
                )
            ),
            "niabd_consecutive_frozen_rounds": float(
                _metric(
                    metrics,
                    "niabd_consecutive_frozen_rounds",
                    round_idx,
                    np.nan,
                )
            ),
            "niabd_effective_memory_weight": float(
                _metric(
                    metrics,
                    "niabd_effective_memory_weight",
                    round_idx,
                    np.nan,
                )
            ),
            "niabd_eligible_teacher_observations": float(
                _metric(
                    metrics,
                    "niabd_eligible_teacher_observations",
                    round_idx,
                    np.nan,
                )
            ),
            "niabd_memory_update_rounds": float(
                _metric(
                    metrics,
                    "niabd_memory_update_rounds",
                    round_idx,
                    np.nan,
                )
            ),
            "vcaa_hard_rejected": int(
                _metric(metrics, "vcaa_hard_rejected", round_idx, 0)
            ),
            "vcaa_version_rejected": int(
                _metric(metrics, "vcaa_version_rejected", round_idx, 0)
            ),
            "vcaa_age_rejected": int(
                _metric(metrics, "vcaa_age_rejected", round_idx, 0)
            ),
            "vcaa_freshness_score_mean": float(
                _metric(metrics, "vcaa_freshness_score_mean", round_idx, np.nan)
            ),
            "vcaa_content_reliability_mean": float(
                _metric(metrics, "vcaa_content_reliability_mean", round_idx, np.nan)
            ),
            "vcaa_aggregation_weight_mean": float(
                _metric(metrics, "vcaa_aggregation_weight_mean", round_idx, np.nan)
            ),
            "niabd_phase": _metric(metrics, "niabd_phase", round_idx, ""),
            "niabd_round_risk": float(_metric(metrics, "niabd_round_risk", round_idx, np.nan)),
            "niabd_risk_ema": float(_metric(metrics, "niabd_risk_ema", round_idx, np.nan)),
            "niabd_consensus_shift": float(_metric(metrics, "niabd_consensus_shift", round_idx, np.nan)),
            "niabd_eligible_ratio": float(_metric(metrics, "niabd_eligible_ratio", round_idx, np.nan)),
            "niabd_trusted_memory_frozen": _metric(metrics, "niabd_trusted_memory_frozen", round_idx, None),
            "niabd_trusted_memory_updated": _metric(metrics, "niabd_trusted_memory_updated", round_idx, None),
            "niabd_threshold_update_mode": _metric(metrics, "niabd_threshold_update_mode", round_idx, ""),
            "niabd_reference_trusted_weight": float(_metric(metrics, "niabd_reference_trusted_weight", round_idx, np.nan)),
            "niabd_recovery_stable_rounds": float(_metric(metrics, "niabd_recovery_stable_rounds", round_idx, np.nan)),
            "nonfinite_eval_batches": int(
                _metric(metrics, "nonfinite_eval_batches", round_idx)
            ),
            "nonfinite_distill_rollbacks": int(
                _metric(metrics, "nonfinite_distill_rollbacks", round_idx)
            ),
            "numeric_failure_count": float(
                _metric(metrics, "numeric_failure_count", round_idx)
            ),
            "transaction_id": _metric(
                metrics, "transaction_id", round_idx, None
            ),
            "transaction_status": _metric(
                metrics, "transaction_status", round_idx, "committed"
            ),
            "rollback_reason": _metric(
                metrics, "rollback_reason", round_idx, None
            ),
            "student_snapshot_sha256": _metric(
                metrics, "student_snapshot_sha256", round_idx, None
            ),
            "received_teachers": _metric(
                metrics,
                "received_teachers",
                round_idx,
                _metric(metrics, "clients_trained", round_idx, 0),
            ),
            "checkpoint_path": _metric(
                metrics, "checkpoint_path", round_idx, None
            ),
            "checkpoint_sha256": _metric(
                metrics, "checkpoint_sha256", round_idx, None
            ),
            **{
                key: _process_metric(metrics, key, round_idx)
                for key in PROCESS_ONLY_ROUND_FIELDS
            },
            # Reproducibility metadata is appended after the v3 round schema.
            "git_commit_sha": str(metrics.get("git_commit_sha", "unavailable")),
            "git_dirty": metrics.get("git_dirty", "unavailable"),
            "config_sha256": str(metrics.get("config_sha256", "")),
            "runtime_profile_sha256": str(metrics.get("runtime_profile_sha256", "")),
        }


def _summary_row(
    rows: List[dict],
    runtime_events: List[dict] | None = None,
) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty run.")
    last = rows[-1]
    is_process = str(last["runtime"]).lower() == PROCESS_RUNTIME

    def process_total(keys: str | tuple[str, ...]):
        if not is_process:
            return np.nan
        selected = (keys,) if isinstance(keys, str) else keys
        values = [
            row[key]
            for row in rows
            for key in selected
        ]
        if any(pd.isna(value) for value in values):
            return np.nan
        return sum(values)

    summary = {
        "run_uid": last["run_uid"],
        "dataset": last["dataset"],
        "seed": last["seed"],
        "runtime": last["runtime"],
        "strategy": last["strategy"],
        "result_schema_version": last.get(
            "result_schema_version", RESULT_SCHEMA_VERSION
        ),
        "vcaa_algorithm_version": last.get(
            "vcaa_algorithm_version", "none"
        ),
        "niabd_algorithm_version": last.get(
            "niabd_algorithm_version", "none"
        ),
        "aggregation_algorithm_version": last.get(
            "aggregation_algorithm_version", AGGREGATION_ALGORITHM_VERSION
        ),
        "run_class": last.get("run_class", "smoke"),
        "attack_condition": last.get("attack_condition", "clean"),
        "topology": last["topology"],
        "server_model": last["server_model"],
        "client_model": last["client_model"],
        "architecture_assignment_hash": last.get(
            "architecture_assignment_hash", ""
        ),
        "server_parameter_count": last.get("server_parameter_count", 0),
        "client_parameter_count": last.get("client_parameter_count", 0),
        "server_device": last["server_device"],
        "client_device": last["client_device"],
        "num_clients": last["num_clients"],
        "partition_scheme": last["partition_scheme"],
        "knowledge_interface": last["knowledge_interface"],
        "aggregation_rule": last["aggregation_rule"],
        "vcaa_enabled": last["vcaa_enabled"],
        "admission_method": last["admission_method"],
        "niabd_enabled": last["niabd_enabled"],
        "defense_method": last["defense_method"],
        "niabd_prototype_update_reason": last.get(
            "niabd_prototype_update_reason", ""
        ),
        "attack_type": last.get("attack_type", "none"),
        "attack_plan_id": last.get("attack_plan_id", ""),
        "target_label": last.get("target_label", -1),
        "malicious_fraction": last.get("malicious_fraction", 0.0),
        "poison_ratio": last.get("poison_ratio", 0.0),
        "attack_start_round": last.get("attack_start_round", -1),
        "rounds": len(rows),
        "final_accuracy": last["accuracy"],
        "best_accuracy": max(float(row["accuracy"]) for row in rows),
        "final_loss": last["loss"],
        "wall_clock_time_s": last["wall_clock_time_s"],
        "total_client_upload_bytes": sum(
            int(row["client_upload_bytes"]) for row in rows
        ),
        "total_server_broadcast_bytes": sum(
            int(row["server_broadcast_bytes"]) for row in rows
        ),
        "mean_teacher_utilization": float(
            np.mean([row["teacher_utilization"] for row in rows])
        ),
        "total_teachers_admitted": sum(
            int(row["teachers_admitted"]) for row in rows
        ),
        "admitted_teachers": int(last.get("admitted_teachers", 0)),
        "total_teachers_rejected": sum(
            int(row["teachers_rejected"]) for row in rows
        ),
        "total_teachers_purified": sum(
            int(row["teachers_purified"]) for row in rows
        ),
        "mean_niabd_anomaly_fraction": float(
            np.mean([row["niabd_anomaly_fraction"] for row in rows])
        ),
        "mean_niabd_suppression": float(
            np.mean([row["niabd_mean_suppression"] for row in rows])
        ),
        "final_niabd_threshold_mean": last["niabd_threshold_mean"],
        "total_niabd_prototype_updates": sum(
            float(row["niabd_prototype_updated"]) for row in rows
        ),
        "total_niabd_eligible_teacher_observations": float(
            rows[-1].get("niabd_eligible_teacher_observations", np.nan)
        ),
        "total_niabd_memory_update_rounds": float(
            rows[-1].get("niabd_memory_update_rounds", np.nan)
        ),
        "memory_candidate_teachers": last.get(
            "memory_candidate_teachers", None
        ),
        "normal_eligible_teachers": last.get(
            "normal_eligible_teachers", None
        ),
        "drift_recovery_candidates": last.get(
            "drift_recovery_candidates", None
        ),
        "memory_update_teachers": last.get("memory_update_teachers", None),
        "niabd_memory_update_reason": last.get(
            "niabd_memory_update_reason", None
        ),
        "niabd_observations": last.get("niabd_observations", None),
        "vcaa_history_size": last.get("vcaa_history_size", None),
        "niabd_defense_available": last.get(
            "niabd_defense_available", None
        ),
        "niabd_purification_applied": last.get(
            "niabd_purification_applied", None
        ),
        "niabd_memory_updated": last.get("niabd_memory_updated", None),
        "transaction_id": last.get("transaction_id", None),
        "transaction_status": last.get("transaction_status", "committed"),
        "rollback_reason": last.get("rollback_reason", None),
        "student_snapshot_sha256": last.get(
            "student_snapshot_sha256", None
        ),
        "received_teachers": last.get("received_teachers", 0),
        "checkpoint_path": last.get("checkpoint_path", None),
        "checkpoint_sha256": last.get("checkpoint_sha256", None),
        "total_rollbacks": sum(
            int(row["nonfinite_distill_rollbacks"]) for row in rows
        ),
        "total_numeric_failures": max(
            float(row["numeric_failure_count"]) for row in rows
        ),
        "numeric_failure_count": last.get("numeric_failure_count", 0),
        "teachers_purified": last.get("teachers_purified", 0),
        "total_client_wire_bytes": process_total("client_wire_bytes"),
        "total_packets_consumed": process_total("packets_consumed"),
        "total_stale_packets": process_total((
            "mild_stale_packets",
            "moderate_stale_packets",
            "severe_stale_packets",
        )),
        "max_version_lag": float(np.nanmax([
            row["max_version_lag"] for row in rows
        ])) if any(
            not pd.isna(row["max_version_lag"]) for row in rows
        ) else np.nan,
        "final_basr_global": last.get("basr_global", np.nan),
        "final_basr_local_1": last.get("basr_local_1", np.nan),
        "final_basr_local_2": last.get("basr_local_2", np.nan),
        "final_basr_local_3": last.get("basr_local_3", np.nan),
        "final_basr_local_4": last.get("basr_local_4", np.nan),
        "total_poisoned_samples": sum(
            int(row.get("poisoned_samples", 0)) for row in rows
        ),
        "total_attack_stats_missing": sum(
            int(row.get("attack_stats_missing_count", 0)) for row in rows
        ),
        "mean_attack_window_basr": (
            float(np.mean([
                float(row["basr_global"])
                for row in rows
                if int(row.get("attack_active", 0)) == 1
                and not pd.isna(row.get("basr_global", np.nan))
            ]))
            if any(
                int(row.get("attack_active", 0)) == 1
                and not pd.isna(row.get("basr_global", np.nan))
                for row in rows
            )
            else np.nan
        ),
    }
    attack_rows = [
        row
        for row in rows
        if int(row.get("attack_active", 0)) == 1
        and not pd.isna(row.get("basr_global", np.nan))
    ]
    if attack_rows:
        attack_rounds = np.asarray(
            [int(row["round"]) for row in attack_rows],
            dtype=float,
        )
        attack_values = np.asarray(
            [float(row["basr_global"]) for row in attack_rows],
            dtype=float,
        )
        peak_index = int(np.argmax(attack_values))
        contiguous = bool(
            len(attack_rounds) == 1
            or np.all(np.diff(attack_rounds.astype(int)) == 1)
        )
        summary.update({
            "peak_attack_window_basr": float(attack_values[peak_index]),
            "peak_attack_window_round": int(attack_rounds[peak_index]),
            "attack_window_basr_auc": float(
                np.trapz(attack_values, attack_rounds)
                if len(attack_values) > 1
                else attack_values[0]
            ) if contiguous else np.nan,
            "attack_window_auc_contiguous": int(contiguous),
            "attack_window_round_count": int(len(attack_rounds)),
        })
        last_attack_round = int(attack_rounds[-1])
        recovery_values = [
            float(row["basr_global"])
            for row in rows
            if int(row["round"]) > last_attack_round
            and not pd.isna(row.get("basr_global", np.nan))
        ]
        summary["post_attack_recovery_basr"] = (
            float(np.mean(recovery_values))
            if recovery_values
            else np.nan
        )
    else:
        summary.update({
            "peak_attack_window_basr": np.nan,
            "peak_attack_window_round": np.nan,
            "attack_window_basr_auc": np.nan,
            "attack_window_auc_contiguous": 0,
            "attack_window_round_count": 0,
            "post_attack_recovery_basr": np.nan,
        })
    events = [
        event for event in (runtime_events or [])
        if not pd.isna(event.get("version_lag", np.nan))
    ]
    if events:
        stale = [
            event for event in events
            if int(event["version_lag"]) > 0
        ]
        fresh = [
            event for event in events
            if int(event["version_lag"]) == 0
        ]
        rejected_stale = [
            event for event in stale
            if event.get("admitted") is False
        ]
        rejected_fresh = [
            event for event in fresh
            if event.get("admitted") is False
        ]
        admitted = [
            event for event in events
            if event.get("admitted") is True
        ]
        rejected = [
            event for event in events
            if event.get("admitted") is False
        ]
        summary.update({
            "stale_rejection_rate": (
                len(rejected_stale) / len(stale)
                if stale else np.nan
            ),
            "fresh_rejection_rate": (
                len(rejected_fresh) / len(fresh)
                if fresh else np.nan
            ),
            "mean_admitted_version_lag": (
                float(np.mean([
                    event["version_lag"] for event in admitted
                ]))
                if admitted else np.nan
            ),
            "mean_rejected_version_lag": (
                float(np.mean([
                    event["version_lag"] for event in rejected
                ]))
                if rejected else np.nan
            ),
            "mean_admitted_knowledge_age_s": (
                float(np.mean([
                    event["knowledge_age_s"] for event in admitted
                ]))
                if admitted else np.nan
            ),
            "mean_rejected_knowledge_age_s": (
                float(np.mean([
                    event["knowledge_age_s"] for event in rejected
                ]))
                if rejected else np.nan
            ),
        })
    else:
        summary.update({
            "stale_rejection_rate": np.nan,
            "fresh_rejection_rate": np.nan,
            "mean_admitted_version_lag": np.nan,
            "mean_rejected_version_lag": np.nan,
            "mean_admitted_knowledge_age_s": np.nan,
            "mean_rejected_knowledge_age_s": np.nan,
        })
    # Keep reproducibility metadata append-only after the historical summary
    # fields so old readers and column-order-sensitive exports remain stable.
    summary.update({
        "git_commit_sha": last.get("git_commit_sha", "unavailable"),
        "git_dirty": last.get("git_dirty", "unavailable"),
        "config_sha256": last.get("config_sha256", ""),
        "runtime_profile_sha256": last.get("runtime_profile_sha256", ""),
    })
    return summary


def _admission_rows(
    metrics,
    *,
    run_uid: str,
    dataset_name: str,
    seed: int,
    num_clients: int,
    partition_scheme: str,
) -> Iterable[dict]:
    method = str(metrics.get("admission_method", "none"))
    all_records = metrics.get("teacher_admission_records", [])
    for round_idx, records in enumerate(all_records, start=1):
        for record in records:
            yield {
                "run_uid": run_uid,
                "dataset": dataset_name,
                "seed": int(seed),
                "round": int(round_idx),
                "topology": "server-client",
                "num_clients": int(num_clients),
                "partition_scheme": str(partition_scheme),
                "admission_method": method,
                "strategy": str(metrics.get("strategy", "baseline")),
                "attack_type": str(metrics.get("attack_type", "none")),
                "vcaa_algorithm_version": _lineage_versions(metrics)[
                    "vcaa_algorithm_version"
                ],
                "vcaa_nonfinite_policy": (
                    "fail_closed" if method != "none" else "none"
                ),
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "vcaa_history_size": _metric(
                    metrics, "vcaa_history_size", round_idx - 1, None
                ),
                "received_at_s": None,
                "consumed_at_s": None,
                "proxy_version": None,
                **record,
            }


def _defense_rows(
    metrics,
    *,
    run_uid: str,
    dataset_name: str,
    seed: int,
    num_clients: int,
    partition_scheme: str,
) -> Iterable[dict]:
    method = str(metrics.get("defense_method", "none"))
    all_records = metrics.get("teacher_defense_records", [])
    for round_idx, records in enumerate(all_records, start=1):
        for record in records:
            yield {
                "run_uid": run_uid,
                "dataset": dataset_name,
                "seed": int(seed),
                "round": int(round_idx),
                "topology": "server-client",
                "num_clients": int(num_clients),
                "partition_scheme": str(partition_scheme),
                "defense_method": method,
                "strategy": str(metrics.get("strategy", "baseline")),
                "attack_type": str(metrics.get("attack_type", "none")),
                "niabd_algorithm_version": _metric(
                    metrics,
                    "niabd_algorithm_version",
                    round_idx - 1,
                    "",
                ),
                "result_schema_version": _metric(
                    metrics,
                    "result_schema_version",
                    round_idx - 1,
                    "",
                ),
                "niabd_prototype_update_reason": _metric(
                    metrics,
                    "niabd_prototype_update_reason",
                    round_idx - 1,
                    "",
                ),
                "niabd_defense_available": _metric(
                    metrics, "niabd_defense_available", round_idx - 1, None
                ),
                "niabd_purification_applied": _metric(
                    metrics, "niabd_purification_applied", round_idx - 1, None
                ),
                "niabd_memory_updated": _metric(
                    metrics, "niabd_memory_updated", round_idx - 1, None
                ),
                "memory_candidate_teachers": _metric(
                    metrics,
                    "niabd_memory_candidate_teachers",
                    round_idx - 1,
                    None,
                ),
                "normal_eligible_teachers": _metric(
                    metrics,
                    "niabd_memory_eligible_teachers",
                    round_idx - 1,
                    None,
                ),
                "memory_update_teachers": _metric(
                    metrics, "memory_update_teachers", round_idx - 1, None
                ),
                "niabd_observations": _metric(
                    metrics,
                    "niabd_eligible_teacher_observations",
                    round_idx - 1,
                    None,
                ),
                **record,
            }


def _backdoor_rows(
    metrics,
    *,
    run_uid: str,
    dataset_name: str,
    seed: int,
    num_clients: int,
    partition_scheme: str,
) -> Iterable[dict]:
    attack_type = str(metrics.get("attack_type", "none"))
    strategy = str(metrics.get("strategy", "baseline"))
    admissions = metrics.get("teacher_admission_records", [])
    defenses = metrics.get("teacher_defense_records", [])
    all_records = metrics.get("backdoor_client_records", [])
    for round_idx, records in enumerate(all_records, start=1):
        admission_by_client = {
            int(record["client_id"]): record
            for record in (
                admissions[round_idx - 1]
                if round_idx - 1 < len(admissions) else []
            )
        }
        defense_by_client = {
            int(record["client_id"]): record
            for record in (
                defenses[round_idx - 1]
                if round_idx - 1 < len(defenses) else []
            )
        }
        for record in records:
            client_id = int(record["client_id"])
            admission = admission_by_client.get(client_id)
            defense = defense_by_client.get(client_id)
            yield {
                "run_uid": run_uid,
                "dataset": dataset_name,
                "seed": int(seed),
                "round": int(round_idx),
                "runtime": str(metrics.get("runtime", "sync")),
                "strategy": strategy,
                "attack_type": attack_type,
                "attack_plan_id": str(metrics.get("attack_plan_id", "")),
                "target_label": int(metrics.get("target_label", -1)),
                "topology": "server-client",
                "num_clients": int(num_clients),
                "partition_scheme": str(partition_scheme),
                **record,
                "admitted": (
                    bool(admission["admitted"])
                    if admission is not None else True
                ),
                "admission_score": (
                    float(admission["score"])
                    if admission is not None else np.nan
                ),
                "niabd_anomaly_fraction": (
                    float(defense["anomaly_fraction"])
                    if defense is not None else np.nan
                ),
                "niabd_mean_abs_deviation": (
                    float(defense["mean_abs_deviation"])
                    if defense is not None else np.nan
                ),
                "niabd_max_abs_deviation": (
                    float(defense["max_abs_deviation"])
                    if defense is not None else np.nan
                ),
                "niabd_mean_suppression": (
                    float(defense["mean_suppression"])
                    if defense is not None else np.nan
                ),
                "niabd_teacher_memory_score": (
                    float(defense["teacher_memory_score"])
                    if defense is not None else np.nan
                ),
                "niabd_high_quantile_deviation": (
                    float(defense["high_quantile_deviation"])
                    if defense is not None else np.nan
                ),
                "niabd_mean_excess": (
                    float(defense["mean_excess"])
                    if defense is not None else np.nan
                ),
                "niabd_consensus_deviation": (
                    float(defense["consensus_deviation"])
                    if defense is not None else np.nan
                ),
                "niabd_memory_eligible": (
                    bool(defense["memory_eligible"])
                    if defense is not None else np.nan
                ),
                "phase": (
                    str(defense.get("phase", ""))
                    if defense is not None else ""
                ),
                "round_risk": (
                    float(defense.get("round_risk", np.nan))
                    if defense is not None else np.nan
                ),
                "risk_ema": (
                    float(defense.get("risk_ema", np.nan))
                    if defense is not None else np.nan
                ),
                "consensus_shift": (
                    float(defense.get("consensus_shift", np.nan))
                    if defense is not None else np.nan
                ),
                "eligible_ratio": (
                    float(defense.get("eligible_ratio", np.nan))
                    if defense is not None else np.nan
                ),
                "trusted_memory_frozen": (
                    bool(defense.get("trusted_memory_frozen", False))
                    if defense is not None else np.nan
                ),
                "trusted_memory_updated": (
                    bool(defense.get("trusted_memory_updated", False))
                    if defense is not None else np.nan
                ),
                "threshold_update_mode": (
                    str(defense.get("threshold_update_mode", ""))
                    if defense is not None else ""
                ),
                "reference_trusted_weight": (
                    float(defense.get("reference_trusted_weight", np.nan))
                    if defense is not None else np.nan
                ),
                "recovery_stable_rounds": (
                    int(defense.get("recovery_stable_rounds", 0))
                    if defense is not None else np.nan
                ),
            }


def _runtime_event_rows(
    metrics,
    *,
    run_uid: str,
    dataset_name: str,
    seed: int,
    num_clients: int,
    partition_scheme: str,
) -> Iterable[dict]:
    for event in metrics.get("runtime_events", []):
        yield {
            "run_uid": run_uid,
            "dataset": dataset_name,
            "seed": int(seed),
            "runtime": str(metrics.get("runtime", "sync")),
            "strategy": str(metrics.get("strategy", "baseline")),
            "topology": "server-client",
            "num_clients": int(num_clients),
            "partition_scheme": str(partition_scheme),
            "server_device": str(metrics.get("server_device", "")),
            "client_device": str(metrics.get("client_device", "")),
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "vcaa_algorithm_version": _lineage_versions(metrics)[
                "vcaa_algorithm_version"
            ],
            "niabd_algorithm_version": _lineage_versions(metrics)[
                "niabd_algorithm_version"
            ],
            "aggregation_algorithm_version": AGGREGATION_ALGORITHM_VERSION,
            "run_class": str(metrics.get("run_class", "smoke")),
            "attack_condition": str(
                metrics.get("attack_condition", "clean")
            ),
            **event,
        }


def _standard_csv_columns() -> dict[str, tuple[str, ...]]:
    """Return stable schemas, including tables that may have no data rows."""

    placeholder_metrics = {
        "acc_list": [np.nan],
        "runtime": "sync",
    }
    placeholder_round = next(_round_rows(
        placeholder_metrics,
        run_uid="",
        dataset_name="",
        seed=0,
        num_clients=1,
        partition_scheme="",
    ))
    placeholder_summary = _summary_row([placeholder_round])
    return {
        "round": tuple(placeholder_round),
        "summary": tuple(placeholder_summary),
        "admission": ADMISSION_COLUMNS,
        "defense": DEFENSE_COLUMNS,
        "runtime": RUNTIME_EVENT_COLUMNS,
        "backdoor": BACKDOOR_COLUMNS,
    }


def _ensure_csv_header(path: str, columns: tuple[str, ...]) -> None:
    if not os.path.exists(path):
        pd.DataFrame(columns=list(columns)).to_csv(path, index=False)


def _strategy_name(enable_vcaa: bool, enable_niabd: bool) -> str:
    if enable_vcaa and enable_niabd:
        return "vcaa-niabd"
    if enable_vcaa:
        return "vcaa"
    if enable_niabd:
        return "niabd"
    return "baseline"


def _trace_output_path(
    requested: str,
    *,
    seed: int,
    num_clients: int,
    partition_scheme: str,
) -> str:
    requested = os.path.abspath(requested)
    if requested.lower().endswith(".json"):
        return requested
    os.makedirs(requested, exist_ok=True)
    return os.path.join(
        requested,
        (
            f"runtime_trace_seed_{seed}_clients_{num_clients}_"
            f"{partition_scheme}.json"
        ),
    )


def _make_checkpoint_callback(
    *,
    checkpoint_every_rounds: int,
    checkpoint_root: str,
    config_hash: str,
    rounds: int,
    runtime: str,
    run_uid: str,
    server_model,
    client_models,
    admission_controller,
    defense_controller,
    attack_plan,
    data_plan,
    dataset_name: str,
    seed: int,
    server_architecture: str,
    client_architectures: List[str],
):
    if int(checkpoint_every_rounds) <= 0:
        return None

    def _callback(checkpoint_round: int, current_metrics: dict) -> None:
        if checkpoint_round % int(checkpoint_every_rounds):
            current_metrics.setdefault("checkpoint_path", []).append(None)
            current_metrics.setdefault("checkpoint_sha256", []).append(None)
            return
        checkpoint_path = os.path.join(
            checkpoint_root,
            f"{run_uid}_round_{checkpoint_round}.pt",
        )
        payload = build_checkpoint_payload(
            current_round=int(checkpoint_round),
            expected_rounds=int(rounds),
            run_uid=str(run_uid),
            config_sha256=str(config_hash),
            runtime=str(runtime),
            server_model=server_model,
            client_models=client_models,
            admission_controller=admission_controller,
            defense_controller=defense_controller,
            attack_plan_state=(
                {"identity": attack_plan.identity}
                if attack_plan is not None
                else None
            ),
            data_identity={
                "dataset": str(dataset_name),
                "seed": int(seed),
                "proxy_version": (
                    str(data_plan.proxy_version)
                    if data_plan is not None
                    else ""
                ),
                "proxy_size": (
                    len(data_plan.proxy_indices)
                    if data_plan is not None
                    else 0
                ),
            },
            proxy_identity=(
                {"version": str(data_plan.proxy_version)}
                if data_plan is not None
                else None
            ),
            architecture_assignment={
                "server": str(server_architecture),
                "clients": list(client_architectures),
            },
            manifest_identity={"manifest_sha256": str(config_hash)},
            metrics_state=current_metrics,
        )
        checkpoint_sha256 = save_checkpoint_atomic(payload, checkpoint_path)
        current_metrics.setdefault("checkpoint_path", []).append(
            checkpoint_path
        )
        current_metrics.setdefault("checkpoint_sha256", []).append(
            checkpoint_sha256
        )

    return _callback


def run_experiment(
    *,
    dataset_path: str,
    dataset_name: str,
    rounds: int,
    epochs: int,
    batch_size: int,
    device: str,
    seeds: List[int],
    num_clients_list: List[int],
    partition_schemes: List[str],
    label_skew_classes: int,
    quantity_skew_alpha: float,
    dirichlet_alpha: float,
    outdir: str,
    append: bool,
    num_workers: int,
    auxiliary_num_workers: int,
    pin_memory: bool,
    amp: bool,
    persistent_workers: bool,
    loader_mp_context: str,
    proxy_ratio: float,
    val_ratio: float,
    proxy_dataset_size: int,
    private_dataset_size: int = 0,
    distill_temperature: float,
    strict_numeric_checks: bool,
    enable_vcaa: bool,
    vcaa_config: VCAAConfig,
    enable_niabd: bool,
    niabd_config: NIABDConfig,
    enable_client_distillation: bool,
    runtime: str = "sync",
    process_config: ProcessRuntimeConfig | None = None,
    runtime_profile: str = "",
    runtime_trace_path: str = "",
    runtime_trace_out: str = "",
    attack_config: AttackConfig | None = None,
    attack_plan_path: str = "",
    enable_backdoor_diagnostics: bool = False,
    run_class: str = "smoke",
    aggregation_rule: str = "mean-soft-probabilities",
    aggregation_trim_fraction: float = 0.1,
    clean_ce_weight: float = 0.05,
    server_architecture: str = "resnet18",
    client_architectures: List[str] | None = None,
    attack_condition: str = "",
    checkpoint_every_rounds: int = 0,
    checkpoint_dir: str = "",
    resume_from_checkpoint: str = "",
) -> None:
    os.makedirs(outdir, exist_ok=True)
    schema_path = os.path.join(outdir, "result_schema_v3.json")
    write_schema(schema_path)
    round_csv = os.path.join(
        outdir,
        f"fedagg_experiment_results_{dataset_name}.csv",
    )
    summary_csv = os.path.join(
        outdir,
        f"fedagg_run_summary_{dataset_name}.csv",
    )
    admission_csv = os.path.join(
        outdir,
        f"fedagg_teacher_admission_{dataset_name}.csv",
    )
    defense_csv = os.path.join(
        outdir,
        f"fedagg_teacher_defense_{dataset_name}.csv",
    )
    runtime_event_csv = os.path.join(
        outdir,
        f"fedagg_runtime_events_{dataset_name}.csv",
    )
    backdoor_csv = os.path.join(
        outdir,
        f"fedagg_backdoor_defense_{dataset_name}.csv",
    )
    manifest = {
        "manifest_version": "fedagg-manifest-v3",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_class": str(run_class),
        "attack_condition": str(attack_condition or "auto"),
        "dataset": str(dataset_name),
        "runtime": str(runtime),
        "rounds": int(rounds),
        "seeds": [int(value) for value in seeds],
        "num_clients_list": [int(value) for value in num_clients_list],
        "partition_schemes": [str(value) for value in partition_schemes],
        "private_dataset_size": int(private_dataset_size),
        "aggregation_rule": str(aggregation_rule),
        "aggregation_trim_fraction": float(aggregation_trim_fraction),
        "server_architecture": str(server_architecture),
        "client_architectures": list(client_architectures or []),
        "formal_config_unchanged": True,
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    reproducibility = _git_reproducibility_metadata()
    runtime_profile_sha256 = ""
    if runtime_profile and os.path.isfile(runtime_profile):
        runtime_profile_sha256 = hashlib.sha256(
            Path(runtime_profile).read_bytes()
        ).hexdigest()
    Path(outdir, "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    runtime = str(runtime).lower()
    if runtime not in {"sync", "process-semi-async"}:
        raise ValueError(f"Unsupported runtime={runtime!r}.")
    if runtime == "process-semi-async" and process_config is None:
        raise ValueError("process_config is required for process runtime.")
    if resume_from_checkpoint and runtime != "sync":
        raise ValueError(
            "Process-runtime resume is refused until all live client task and "
            "coordinator state can be restored atomically."
        )
    resume_payload = None
    if resume_from_checkpoint:
        resume_payload = load_checkpoint(
            resume_from_checkpoint,
            expected_config_sha256=str(manifest["manifest_sha256"]),
            expected_runtime=runtime,
            expected_rounds=int(rounds),
        )
    if enable_backdoor_diagnostics and runtime != "sync":
        raise ValueError(
            "Experiment-only backdoor diagnostics currently support only "
            "sync runtime so process scheduling, deadlines, and traces remain "
            "unchanged."
        )
    if not append:
        for path in (
            round_csv,
            summary_csv,
            admission_csv,
            defense_csv,
            runtime_event_csv,
            backdoor_csv,
        ):
            if os.path.exists(path):
                os.remove(path)
    schemas = _standard_csv_columns()
    for path, schema_name in (
        (round_csv, "round"),
        (summary_csv, "summary"),
        (admission_csv, "admission"),
        (defense_csv, "defense"),
        (runtime_event_csv, "runtime"),
        (backdoor_csv, "backdoor"),
    ):
        _ensure_csv_header(path, schemas[schema_name])

    attack_config = attack_config or AttackConfig(attack_type="none")
    seeds = seeds or [0]
    num_clients_list = num_clients_list or [6]
    partition_schemes = partition_schemes or ["iid"]
    if (
        runtime == "process-semi-async"
        and runtime_trace_out.lower().endswith(".json")
        and (
            len(seeds)
            * len(num_clients_list)
            * len(partition_schemes)
        ) > 1
    ):
        raise ValueError(
            "A single --runtime-trace-out JSON file cannot represent "
            "multiple runs; provide a directory instead."
        )

    for seed in seeds:
        for num_clients in num_clients_list:
            if int(num_clients) <= 0:
                raise ValueError("Every client count must be positive.")
            for partition_scheme in partition_schemes:
                set_global_seed(seed)
                run_uid = (
                    str(resume_payload["run_uid"])
                    if resume_payload is not None
                    else uuid.uuid4().hex[:12]
                )
                print(
                    "\n===== FedAgg server-client run =====\n"
                    f"dataset={dataset_name} seed={seed} clients={num_clients} "
                    f"partition={partition_scheme}"
                )

                data_plan = None
                trace = None
                if runtime == "sync":
                    dataloaders = get_dataloaders(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        num_clients=int(num_clients),
                        batch_size=int(batch_size),
                        seed=int(seed),
                        partition_scheme=str(partition_scheme),
                        label_skew_classes=int(label_skew_classes),
                        quantity_skew_alpha=float(quantity_skew_alpha),
                        dirichlet_alpha=float(dirichlet_alpha),
                        num_workers=int(num_workers),
                        pin_memory=bool(pin_memory),
                        val_ratio=float(val_ratio),
                        proxy_ratio=float(proxy_ratio),
                        proxy_dataset_size=(
                            int(proxy_dataset_size)
                            if int(proxy_dataset_size) > 0
                            else None
                        ),
                        private_dataset_size=(
                            int(private_dataset_size)
                            if int(private_dataset_size) > 0
                            else None
                        ),
                        persistent_workers=bool(persistent_workers),
                        loader_mp_context=(
                            None
                            if str(loader_mp_context).lower() == "none"
                            else str(loader_mp_context)
                        ),
                        auxiliary_num_workers=int(auxiliary_num_workers),
                    )
                else:
                    assert process_config is not None
                    data_plan = build_federated_data_plan(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        num_clients=int(num_clients),
                        batch_size=int(batch_size),
                        seed=int(seed),
                        partition_scheme=str(partition_scheme),
                        label_skew_classes=int(label_skew_classes),
                        quantity_skew_alpha=float(quantity_skew_alpha),
                        dirichlet_alpha=float(dirichlet_alpha),
                        val_ratio=float(val_ratio),
                        proxy_ratio=float(proxy_ratio),
                        proxy_dataset_size=(
                            int(proxy_dataset_size)
                            if int(proxy_dataset_size) > 0
                            else None
                        ),
                        private_dataset_size=(
                            int(private_dataset_size)
                            if int(private_dataset_size) > 0
                            else None
                        ),
                    )
                    dataloaders = build_server_dataloaders_from_plan(
                        data_plan,
                        num_workers=int(auxiliary_num_workers),
                        pin_memory=bool(pin_memory),
                        loader_mp_context=(
                            None
                            if str(loader_mp_context).lower() == "none"
                            else str(loader_mp_context)
                        ),
                    )
                    if runtime_trace_path:
                        trace = RuntimeTrace.load(runtime_trace_path)
                        if (
                            int(trace.seed) != int(seed)
                            or int(trace.num_clients) != int(num_clients)
                            or int(trace.rounds) != int(rounds)
                            or int(trace.warmup_rounds)
                            != int(process_config.warmup_rounds)
                        ):
                            raise ValueError(
                                "Loaded runtime trace does not match "
                                "seed/client/round dimensions."
                            )
                    else:
                        profile = load_runtime_profile(runtime_profile)
                        trace = generate_runtime_trace(
                            profile=profile,
                            seed=int(seed),
                            num_clients=int(num_clients),
                            rounds=int(rounds),
                            warmup_rounds=int(
                                process_config.warmup_rounds
                            ),
                            participation_rate=float(
                                process_config.participation_rate
                            ),
                        )
                client_models = None
                server_model = None
                metrics = None
                admission_controller = None
                defense_controller = None
                attack_plan = AttackPlan.resolve(
                    seed=int(seed),
                    num_clients=int(num_clients),
                    config=attack_config,
                    plan_path=(attack_plan_path or None),
                )
                attack_plan_file = os.path.join(
                    outdir,
                    "attack_plans",
                    (
                        f"attack_plan_{dataset_name}_seed_{seed}_clients_{num_clients}_"
                        f"{partition_scheme}_{attack_config.attack_type}.json"
                    ),
                )
                attack_plan.save(attack_plan_file)
                try:
                    if runtime == "sync":
                        client_models, server_model = build_models(
                            dataset_name=dataset_name,
                            num_clients=int(num_clients),
                            device=device,
                            server_architecture=str(server_architecture),
                            client_architectures=(
                                client_architectures
                                if client_architectures is not None
                                else None
                            ),
                        )
                    else:
                        assert process_config is not None
                        server_model = build_model(
                            str(server_architecture),
                            dataset_name=dataset_name,
                            device=process_config.server_device,
                        )
                    if enable_vcaa:
                        admission_controller = VersionContentAwareAdmission(
                            vcaa_config
                        )
                    if enable_niabd:
                        defense_controller = (
                            NeuroInspiredAdaptiveBackdoorDefense(
                                niabd_config
                            )
                        )
                    effective_client_architectures = (
                        list(client_architectures)
                        if client_architectures is not None
                        else list(process_config.client_architectures)
                        if runtime == "process-semi-async"
                        and process_config is not None
                        and process_config.client_architectures is not None
                        else [
                            str(
                                process_config.client_architecture
                                if runtime == "process-semi-async"
                                and process_config is not None
                                else server_architecture
                            )
                        ] * int(num_clients)
                    )
                    checkpoint_callback = _make_checkpoint_callback(
                        checkpoint_every_rounds=int(checkpoint_every_rounds),
                        checkpoint_root=os.path.abspath(
                            checkpoint_dir
                            or os.path.join(outdir, "checkpoints")
                        ),
                        config_hash=str(manifest["manifest_sha256"]),
                        rounds=int(rounds),
                        runtime=str(runtime),
                        run_uid=str(run_uid),
                        server_model=server_model,
                        client_models=client_models,
                        admission_controller=admission_controller,
                        defense_controller=defense_controller,
                        attack_plan=attack_plan,
                        data_plan=data_plan,
                        dataset_name=dataset_name,
                        seed=int(seed),
                        server_architecture=str(server_architecture),
                        client_architectures=effective_client_architectures,
                    )
                    if runtime == "sync":
                        metrics = run_fedagg_server_client(
                            client_models,
                            server_model,
                            dataloaders,
                            device=device,
                            local_epochs=int(epochs),
                            rounds=int(rounds),
                            learning_rate=0.01,
                            distill_temperature=float(
                                distill_temperature
                            ),
                            amp=bool(amp),
                            strict_numeric_checks=bool(
                                strict_numeric_checks
                            ),
                            admission_controller=admission_controller,
                            defense_controller=defense_controller,
                            enable_client_distillation=bool(
                                enable_client_distillation
                            ),
                            aggregation_rule=str(aggregation_rule),
                            aggregation_trim_fraction=float(
                                aggregation_trim_fraction
                            ),
                            clean_ce_weight=float(clean_ce_weight),
                            checkpoint_callback=checkpoint_callback,
                            resume_payload=resume_payload,
                            attack_plan=attack_plan,
                            enable_backdoor_diagnostics=bool(
                                enable_backdoor_diagnostics
                            ),
                            backdoor_diagnostics_dataset=dataset_name,
                        )
                        metrics.update({
                            "runtime": "sync",
                            "server_device": str(device),
                            "client_device": str(device),
                            "runtime_events": [],
                        })
                    else:
                        assert process_config is not None
                        assert data_plan is not None
                        assert trace is not None
                        metrics = (
                            run_fedagg_server_client_process_async(
                                server_model=server_model,
                                server_dataloaders=dataloaders,
                                data_plan=data_plan,
                                trace=trace,
                                config=process_config,
                                local_epochs=int(epochs),
                                rounds=int(rounds),
                                learning_rate=0.01,
                                distill_temperature=float(
                                    distill_temperature
                                ),
                                admission_controller=(
                                    admission_controller
                                ),
                                defense_controller=defense_controller,
                                enable_client_distillation=bool(
                                    enable_client_distillation
                                ),
                                aggregation_rule=str(aggregation_rule),
                                aggregation_trim_fraction=float(
                                    aggregation_trim_fraction
                                ),
                                clean_ce_weight=float(clean_ce_weight),
                                checkpoint_callback=checkpoint_callback,
                                attack_plan=attack_plan,
                            )
                        )
                    strategy = _strategy_name(
                        enable_vcaa,
                        enable_niabd,
                    )
                    metrics["strategy"] = strategy
                    metrics.update(reproducibility)
                    metrics["config_sha256"] = str(manifest["manifest_sha256"])
                    metrics["runtime_profile_sha256"] = runtime_profile_sha256
                    metrics["run_class"] = str(run_class)
                    metrics["attack_condition"] = (
                        str(attack_condition)
                        if attack_condition
                        else (
                            "clean"
                            if attack_config.attack_type == "none"
                            else "attacked"
                        )
                    )
                    effective_client_architectures = (
                        list(client_architectures)
                        if client_architectures is not None
                        else list(process_config.client_architectures)
                        if runtime == "process-semi-async"
                        and process_config is not None
                        and process_config.client_architectures is not None
                        else [
                            str(
                                process_config.client_architecture
                                if runtime == "process-semi-async"
                                and process_config is not None
                                else server_architecture
                            )
                        ] * int(num_clients)
                    )
                    metrics["server_model"] = str(server_architecture)
                    metrics["client_model"] = ",".join(
                        effective_client_architectures
                    )
                    metrics["architecture_assignment_hash"] = (
                        architecture_assignment_hash(
                            server_architecture=str(server_architecture),
                            client_architectures=effective_client_architectures,
                        )
                    )
                    metrics["server_parameter_count"] = model_parameter_count(
                        server_model
                    )
                    metrics["client_parameter_count"] = (
                        model_parameter_count(client_models[0])
                        if client_models is not None
                        else model_parameter_count(
                            build_model(
                                str(
                                    process_config.client_architecture
                                    if process_config is not None
                                    else server_architecture
                                ),
                                dataset_name=dataset_name,
                                device="cpu",
                            )
                        )
                    )
                    metrics.setdefault("checkpoint_path", [])
                    metrics.setdefault("checkpoint_sha256", [])
                    metrics["attack_plan_path"] = attack_plan_file
                    metrics["dirichlet_alpha"] = float(dirichlet_alpha)
                    if runtime == "process-semi-async" and runtime_trace_out:
                        assert trace is not None
                        trace_path = _trace_output_path(
                            runtime_trace_out,
                            seed=int(seed),
                            num_clients=int(num_clients),
                            partition_scheme=str(partition_scheme),
                        )
                        trace.save(
                            trace_path,
                            metadata={
                                "run_uid": run_uid,
                                "seed": int(seed),
                                "strategy": strategy,
                                "dataset": dataset_name,
                                "partition_scheme": partition_scheme,
                            },
                        )
                        metrics["runtime_trace_path"] = trace_path
                    rows = list(
                        _round_rows(
                            metrics,
                            run_uid=run_uid,
                            dataset_name=dataset_name,
                            seed=int(seed),
                            num_clients=int(num_clients),
                            partition_scheme=str(partition_scheme),
                        )
                    )
                    pd.DataFrame(rows).reindex(
                        columns=schemas["round"]
                    ).to_csv(
                        round_csv,
                        mode="a",
                        header=False,
                        index=False,
                    )
                    pd.DataFrame([_summary_row(
                        rows,
                        list(metrics.get("runtime_events", [])),
                    )]).reindex(columns=schemas["summary"]).to_csv(
                        summary_csv,
                        mode="a",
                        header=False,
                        index=False,
                    )
                    admission_rows = list(
                        _admission_rows(
                            metrics,
                            run_uid=run_uid,
                            dataset_name=dataset_name,
                            seed=int(seed),
                            num_clients=int(num_clients),
                            partition_scheme=str(partition_scheme),
                        )
                    )
                    if admission_rows:
                        pd.DataFrame(admission_rows).reindex(
                            columns=schemas["admission"]
                        ).to_csv(
                            admission_csv,
                            mode="a",
                            header=False,
                            index=False,
                        )
                    defense_rows = list(
                        _defense_rows(
                            metrics,
                            run_uid=run_uid,
                            dataset_name=dataset_name,
                            seed=int(seed),
                            num_clients=int(num_clients),
                            partition_scheme=str(partition_scheme),
                        )
                    )
                    if defense_rows:
                        pd.DataFrame(defense_rows).reindex(
                            columns=schemas["defense"]
                        ).to_csv(
                            defense_csv,
                            mode="a",
                            header=False,
                            index=False,
                        )
                    runtime_rows = list(
                        _runtime_event_rows(
                            metrics,
                            run_uid=run_uid,
                            dataset_name=dataset_name,
                            seed=int(seed),
                            num_clients=int(num_clients),
                            partition_scheme=str(partition_scheme),
                        )
                    )
                    if runtime_rows:
                        pd.DataFrame(runtime_rows).reindex(
                            columns=schemas["runtime"]
                        ).to_csv(
                            runtime_event_csv,
                            mode="a",
                            header=False,
                            index=False,
                        )
                    backdoor_rows = list(
                        _backdoor_rows(
                            metrics,
                            run_uid=run_uid,
                            dataset_name=dataset_name,
                            seed=int(seed),
                            num_clients=int(num_clients),
                            partition_scheme=str(partition_scheme),
                        )
                    )
                    if backdoor_rows:
                        pd.DataFrame(backdoor_rows).reindex(
                            columns=schemas["backdoor"]
                        ).to_csv(
                            backdoor_csv,
                            mode="a",
                            header=False,
                            index=False,
                        )
                    print(f"[write] {round_csv}")
                    print(f"[write] {summary_csv}")
                    print(f"[write] {admission_csv}")
                    print(f"[write] {defense_csv}")
                    print(f"[write] {runtime_event_csv}")
                    print(f"[write] {backdoor_csv}")
                    print(f"[write] {attack_plan_file}")
                finally:
                    cleanup_dataloaders(dataloaders)
                    del metrics, client_models, server_model, dataloaders
                    del data_plan, trace
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedAgg server-client federated distillation runner"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--dataset-name",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "femnist",
            "cinic10",
            "tiny-imagenet-200",
        ],
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-clients-list", default="6")
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--partition-schemes",
        default="iid",
        help="Comma-separated: iid,label-skew,quantity-skew,dirichlet",
    )
    parser.add_argument("--label-skew-classes", type=int, default=2)
    parser.add_argument("--quantity-skew-alpha", type=float, default=0.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--proxy-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--proxy-dataset-size", type=int, default=0)
    parser.add_argument(
        "--private-dataset-size",
        type=int,
        default=0,
        help=(
            "Optional deterministic real-dataset subset size for smoke/control "
            "runs; 0 uses the complete private training split."
        ),
    )
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--aggregation-rule",
        choices=[
            "mean-soft-probabilities",
            "median-probabilities",
            "trimmed-mean-probabilities",
            "confidence-consistency-filtered-mean",
        ],
        default="mean-soft-probabilities",
    )
    parser.add_argument("--aggregation-trim-fraction", type=float, default=0.1)
    parser.add_argument(
        "--server-architecture",
        choices=["resnet18", "small_cnn", "mobilenet_v2"],
        default="resnet18",
    )
    parser.add_argument(
        "--client-architectures",
        default="",
        help="Optional comma-separated per-client architectures.",
    )
    parser.add_argument(
        "--run-class",
        choices=["formal", "smoke", "synthetic", "control"],
        default="smoke",
    )
    parser.add_argument(
        "--attack-condition",
        choices=["clean", "attacked", "triggered-no-poison"],
        default="",
    )
    parser.add_argument(
        "--runtime",
        choices=["sync", "process-semi-async"],
        default="sync",
    )
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument("--quorum-fraction", type=float, default=0.5)
    parser.add_argument(
        "--runtime-warmup-rounds",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--soft-deadline-factor",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--hard-deadline-factor",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--soft-deadline-s",
        type=float,
        default=0.0,
        help="Optional absolute soft deadline for debugging.",
    )
    parser.add_argument(
        "--hard-deadline-s",
        type=float,
        default=0.0,
        help="Optional absolute hard deadline for debugging.",
    )
    parser.add_argument("--rpc-timeout-s", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-s", type=float, default=0.05)
    parser.add_argument(
        "--runtime-registration-timeout-s",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--runtime-shutdown-timeout-s",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--runtime-profile",
        default=os.path.join("configs", "runtime_moderate.json"),
    )
    parser.add_argument("--runtime-trace", default="")
    parser.add_argument("--runtime-trace-out", default="")
    parser.add_argument(
        "--method",
        choices=["baseline", "vcaa", "niabd", "vcaa-niabd"],
        default="",
        help="Optional formal-method alias; existing --enable-vcaa/--enable-niabd flags remain supported.",
    )
    parser.add_argument(
        "--attack",
        choices=["none", "badnets", "dba", "blend", "dynamic"],
        default="none",
    )
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--malicious-fraction", type=float, default=0.2)
    parser.add_argument("--poison-ratio", type=float, default=0.2)
    parser.add_argument("--attack-start-round", type=int, default=15)
    parser.add_argument(
        "--attack-end-round",
        type=int,
        default=0,
        help="0 means the final configured FL round.",
    )
    parser.add_argument("--poison-interval", type=int, default=1)
    parser.add_argument(
        "--trigger-size",
        type=int,
        default=0,
        help="0 selects 4 for 32x32 datasets and 8 for Tiny-ImageNet-200.",
    )
    parser.add_argument("--trigger-value", type=float, default=1.0)
    parser.add_argument("--blend-alpha", type=float, default=0.2)
    parser.add_argument("--dynamic-period", type=int, default=10)
    parser.add_argument(
        "--attack-plan",
        default="",
        help="Optional exact AttackPlan JSON to replay; generated plans are always exported.",
    )
    parser.add_argument(
        "--enable-backdoor-diagnostics",
        action="store_true",
        help=(
            "Enable sync-only experiment oracle diagnostics after all "
            "admission, defense, aggregation, and student updates."
        ),
    )
    parser.add_argument(
        "--enable-vcaa",
        action="store_true",
        help="Enable version-content-aware teacher admission.",
    )
    parser.add_argument(
        "--enable-niabd",
        action="store_true",
        help="Enable neuro-inspired adaptive logits purification.",
    )
    parser.add_argument(
        "--disable-client-distillation",
        action="store_true",
        help=(
            "Disable the server-logits download/reverse-distillation step "
            "while retaining client uploads and server distillation."
        ),
    )
    parser.add_argument("--vcaa-version-weight", type=float, default=0.5)
    parser.add_argument("--vcaa-time-decay-gamma", type=float, default=0.99)
    parser.add_argument("--vcaa-time-unit-s", type=float, default=60.0)
    parser.add_argument("--vcaa-max-version-lag", type=int, default=1)
    parser.add_argument("--vcaa-max-knowledge-age-s", type=float, default=0.0)
    parser.add_argument("--vcaa-age-half-life-s", type=float, default=0.0)
    parser.add_argument("--vcaa-content-threshold-beta", type=float, default=-1.0)
    parser.add_argument("--vcaa-consensus-divergence-scale", type=float, default=0.0)
    parser.add_argument("--vcaa-accuracy-weight", type=float, default=0.5)
    parser.add_argument("--vcaa-entropy-weight", type=float, default=0.25)
    parser.add_argument(
        "--vcaa-divergence-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument("--vcaa-accuracy-scale", type=float, default=1.0)
    parser.add_argument(
        "--vcaa-entropy-scale",
        type=float,
        default=0.0,
        help="Entropy normalization H0; 0 uses log(number of classes).",
    )
    parser.add_argument(
        "--vcaa-divergence-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--vcaa-window-rounds", type=int, default=5)
    parser.add_argument("--vcaa-threshold-beta", type=float, default=1.0)
    parser.add_argument("--vcaa-warmup-rounds", type=int, default=1)
    parser.add_argument(
        "--niabd-initial-threshold",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--niabd-min-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--niabd-max-threshold",
        type=float,
        default=6.0,
    )
    parser.add_argument("--niabd-kappa", type=float, default=1.0)
    parser.add_argument(
        "--niabd-prototype-learning-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--niabd-threshold-learning-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--niabd-potentiation-balance",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--niabd-threshold-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--niabd-benign-deviation-limit",
        type=float,
        default=4.0,
        help=(
            "High-quantile history/consensus deviation limit; it is not "
            "an all-proxy maximum."
        ),
    )
    parser.add_argument(
        "--niabd-warmup-rounds",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--niabd-min-standard-deviation",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--niabd-reference-source",
        choices=["prototype", "student"],
        default="prototype",
    )
    parser.add_argument(
        "--niabd-memory-quantile",
        type=float,
        default=0.95,
        help="Teacher-level history/consensus deviation quantile.",
    )
    parser.add_argument(
        "--niabd-maximum-memory-anomaly-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--niabd-teacher-score-beta",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--niabd-teacher-score-scale-floor",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--niabd-minimum-consensus-teachers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--niabd-consensus-recovery-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--niabd-threshold-exposure-quantile",
        type=float,
        default=0.75,
    )
    parser.add_argument("--niabd-proxy-chunk-size", type=int, default=0)
    parser.add_argument("--niabd-risk-ema-beta", type=float, default=0.30)
    parser.add_argument("--niabd-risk-on", type=float, default=1.25)
    parser.add_argument("--niabd-risk-off", type=float, default=0.60)
    parser.add_argument("--niabd-onset-patience", type=int, default=2)
    parser.add_argument("--niabd-recovery-patience", type=int, default=2)
    parser.add_argument("--niabd-stable-patience", type=int, default=2)
    parser.add_argument("--niabd-memory-clip-z", type=float, default=3.0)
    parser.add_argument("--niabd-reference-clip-z", type=float, default=2.0)
    parser.add_argument("--niabd-normal-memory-lr", type=float, default=0.0)
    parser.add_argument("--niabd-suspicious-memory-lr", type=float, default=0.0)
    parser.add_argument("--niabd-recovery-memory-lr", type=float, default=0.20)
    parser.add_argument("--niabd-clean-ce-weight-normal", type=float, default=0.05)
    parser.add_argument("--niabd-clean-ce-weight-suspicious", type=float, default=0.10)
    parser.add_argument("--niabd-clean-ce-weight-recovery", type=float, default=0.20)
    parser.add_argument("--niabd-threshold-upward-step-limit", type=float, default=0.05)
    parser.add_argument("--clean-ce-weight", type=float, default=0.05)
    parser.add_argument("--device", default=default_main_device())
    parser.add_argument(
        "--server-device",
        default="",
        help="Process runtime server device; defaults to --device.",
    )
    parser.add_argument(
        "--client-device",
        default="cpu",
        help="Device used independently by each Client process.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--client-torch-threads",
        type=int,
        default=1,
    )
    parser.add_argument("--auxiliary-num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument(
        "--loader-mp-context",
        default="none",
        choices=["none", "fork", "spawn", "forkserver"],
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--strict-numeric-checks", action="store_true")
    parser.add_argument("--outdir", default="experiment_results")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--checkpoint-every-rounds", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume-from-checkpoint", default="")
    args = parser.parse_args()

    enable_vcaa = bool(args.enable_vcaa)
    enable_niabd = bool(args.enable_niabd)
    if args.method:
        method_flags = {
            "baseline": (False, False),
            "vcaa": (True, False),
            "niabd": (False, True),
            "vcaa-niabd": (True, True),
        }[args.method]
        explicitly_enabled = (bool(args.enable_vcaa), bool(args.enable_niabd))
        if any(explicitly_enabled) and explicitly_enabled != method_flags:
            raise ValueError(
                "--method conflicts with explicit --enable-vcaa/--enable-niabd flags."
            )
        enable_vcaa, enable_niabd = method_flags
    trigger_size = int(args.trigger_size)
    if trigger_size <= 0:
        trigger_size = 8 if args.dataset_name == "tiny-imagenet-200" else 4
    attack_end_round = (
        int(args.rounds)
        if int(args.attack_end_round) == 0
        else int(args.attack_end_round)
    )
    attack_config = AttackConfig(
        attack_type=args.attack,
        target_label=args.target_label,
        malicious_fraction=args.malicious_fraction,
        poison_ratio=args.poison_ratio,
        attack_start_round=args.attack_start_round,
        attack_end_round=attack_end_round,
        poison_interval=args.poison_interval,
        trigger_size=trigger_size,
        trigger_value=args.trigger_value,
        blend_alpha=args.blend_alpha,
        dynamic_period=args.dynamic_period,
    )
    if int(attack_config.target_label) >= int(dataset_spec(args.dataset_name).num_classes):
        raise ValueError(
            f"target_label={attack_config.target_label} is outside dataset class range."
        )

    device = normalize_device(args.device)
    server_device = normalize_device(
        args.server_device if args.server_device else args.device
    )
    client_device = normalize_device(args.client_device)
    for role, selected_device in (
        ("server", server_device),
        ("client", client_device),
    ):
        if (
            selected_device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                f"{role}_device requests CUDA but CUDA is unavailable."
            )
    pin_memory = bool(args.pin_memory and supports_pin_memory(device))
    amp = bool(args.amp and use_amp_for_device(device))
    process_config = ProcessRuntimeConfig(
        participation_rate=args.participation_rate,
        quorum_fraction=args.quorum_fraction,
        warmup_rounds=args.runtime_warmup_rounds,
        soft_deadline_factor=args.soft_deadline_factor,
        hard_deadline_factor=args.hard_deadline_factor,
        soft_deadline_override_s=args.soft_deadline_s,
        hard_deadline_override_s=args.hard_deadline_s,
        rpc_timeout_s=args.rpc_timeout_s,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff_s,
        registration_timeout_s=args.runtime_registration_timeout_s,
        shutdown_timeout_s=args.runtime_shutdown_timeout_s,
        server_device=str(server_device),
        client_device=str(client_device),
        server_architecture=str(args.server_architecture),
        client_architecture=(
            _parse_str_list(args.client_architectures)[0]
            if _parse_str_list(args.client_architectures)
            else str(args.server_architecture)
        ),
        client_architectures=(
            tuple(_parse_str_list(args.client_architectures))
            if args.client_architectures
            else None
        ),
        amp=bool(args.amp),
        strict_numeric_checks=args.strict_numeric_checks,
        client_num_workers=args.num_workers,
        client_torch_threads=args.client_torch_threads,
        client_pin_memory=bool(
            args.pin_memory and supports_pin_memory(client_device)
        ),
        loader_mp_context=(
            None
            if str(args.loader_mp_context).lower() == "none"
            else str(args.loader_mp_context)
        ),
    )
    vcaa_config = VCAAConfig(
        version_weight=args.vcaa_version_weight,
        time_decay_gamma=args.vcaa_time_decay_gamma,
        time_unit_s=args.vcaa_time_unit_s,
        max_version_lag=args.vcaa_max_version_lag,
        max_knowledge_age_s=(
            None if float(args.vcaa_max_knowledge_age_s) <= 0.0
            else float(args.vcaa_max_knowledge_age_s)
        ),
        age_half_life_s=(
            None if float(args.vcaa_age_half_life_s) <= 0.0
            else float(args.vcaa_age_half_life_s)
        ),
        content_threshold_beta=(
            None if float(args.vcaa_content_threshold_beta) < 0.0
            else float(args.vcaa_content_threshold_beta)
        ),
        consensus_divergence_scale=(
            None if float(args.vcaa_consensus_divergence_scale) <= 0.0
            else float(args.vcaa_consensus_divergence_scale)
        ),
        accuracy_weight=args.vcaa_accuracy_weight,
        entropy_weight=args.vcaa_entropy_weight,
        divergence_weight=args.vcaa_divergence_weight,
        accuracy_scale=args.vcaa_accuracy_scale,
        entropy_scale=(
            None
            if float(args.vcaa_entropy_scale) == 0.0
            else float(args.vcaa_entropy_scale)
        ),
        divergence_scale=args.vcaa_divergence_scale,
        history_window_rounds=args.vcaa_window_rounds,
        threshold_beta=args.vcaa_threshold_beta,
        warmup_rounds=args.vcaa_warmup_rounds,
    )
    niabd_config = NIABDConfig(
        initial_threshold=args.niabd_initial_threshold,
        minimum_threshold=args.niabd_min_threshold,
        maximum_threshold=args.niabd_max_threshold,
        transition_smoothness=args.niabd_kappa,
        prototype_learning_rate=args.niabd_prototype_learning_rate,
        threshold_learning_rate=args.niabd_threshold_learning_rate,
        potentiation_balance=args.niabd_potentiation_balance,
        threshold_decay=args.niabd_threshold_decay,
        benign_deviation_limit=args.niabd_benign_deviation_limit,
        warmup_rounds=args.niabd_warmup_rounds,
        minimum_standard_deviation=(
            args.niabd_min_standard_deviation
        ),
        reference_source=args.niabd_reference_source,
        memory_quantile=args.niabd_memory_quantile,
        maximum_memory_anomaly_fraction=(
            args.niabd_maximum_memory_anomaly_fraction
        ),
        teacher_score_beta=args.niabd_teacher_score_beta,
        teacher_score_scale_floor=args.niabd_teacher_score_scale_floor,
        minimum_consensus_teachers=args.niabd_minimum_consensus_teachers,
        consensus_recovery_fraction=args.niabd_consensus_recovery_fraction,
        threshold_exposure_quantile=args.niabd_threshold_exposure_quantile,
        proxy_chunk_size=args.niabd_proxy_chunk_size,
        risk_ema_beta=args.niabd_risk_ema_beta,
        risk_on=args.niabd_risk_on,
        risk_off=args.niabd_risk_off,
        onset_patience=args.niabd_onset_patience,
        recovery_patience=args.niabd_recovery_patience,
        stable_patience=args.niabd_stable_patience,
        memory_clip_z=args.niabd_memory_clip_z,
        reference_clip_z=args.niabd_reference_clip_z,
        normal_memory_lr=(
            None if float(args.niabd_normal_memory_lr) <= 0.0
            else float(args.niabd_normal_memory_lr)
        ),
        suspicious_memory_lr=args.niabd_suspicious_memory_lr,
        recovery_memory_lr=args.niabd_recovery_memory_lr,
        clean_ce_weight_normal=args.niabd_clean_ce_weight_normal,
        clean_ce_weight_suspicious=args.niabd_clean_ce_weight_suspicious,
        clean_ce_weight_recovery=args.niabd_clean_ce_weight_recovery,
        threshold_upward_step_limit=args.niabd_threshold_upward_step_limit,
    )

    run_experiment(
        dataset_path=args.dataset,
        dataset_name=args.dataset_name,
        rounds=args.rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=str(device),
        seeds=_parse_int_list(args.seeds),
        num_clients_list=_parse_int_list(args.num_clients_list),
        partition_schemes=_parse_str_list(args.partition_schemes),
        label_skew_classes=args.label_skew_classes,
        quantity_skew_alpha=args.quantity_skew_alpha,
        dirichlet_alpha=args.dirichlet_alpha,
        outdir=args.outdir,
        append=args.append,
        num_workers=args.num_workers,
        auxiliary_num_workers=args.auxiliary_num_workers,
        pin_memory=pin_memory,
        amp=amp,
        persistent_workers=args.persistent_workers,
        loader_mp_context=args.loader_mp_context,
        proxy_ratio=args.proxy_ratio,
        val_ratio=args.val_ratio,
        proxy_dataset_size=args.proxy_dataset_size,
        private_dataset_size=args.private_dataset_size,
        distill_temperature=args.distill_temperature,
        strict_numeric_checks=args.strict_numeric_checks,
        enable_vcaa=enable_vcaa,
        vcaa_config=vcaa_config,
        enable_niabd=enable_niabd,
        niabd_config=niabd_config,
        enable_client_distillation=not args.disable_client_distillation,
        runtime=args.runtime,
        process_config=process_config,
        runtime_profile=args.runtime_profile,
        runtime_trace_path=args.runtime_trace,
        runtime_trace_out=(
            args.runtime_trace_out
            if args.runtime_trace_out
            else os.path.join(args.outdir, "runtime_traces")
        ),
        attack_config=attack_config,
        attack_plan_path=args.attack_plan,
        enable_backdoor_diagnostics=args.enable_backdoor_diagnostics,
        run_class=args.run_class,
        aggregation_rule=args.aggregation_rule,
        aggregation_trim_fraction=args.aggregation_trim_fraction,
        clean_ce_weight=args.clean_ce_weight,
        server_architecture=args.server_architecture,
        client_architectures=(
            _parse_str_list(args.client_architectures)
            if args.client_architectures
            else None
        ),
        attack_condition=args.attack_condition,
        checkpoint_every_rounds=args.checkpoint_every_rounds,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
