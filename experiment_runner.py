from __future__ import annotations

import argparse
import gc
import os
import random
import uuid
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

from attacks import AttackConfig, AttackPlan
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
    for round_idx in range(rounds):
        yield {
            "run_uid": run_uid,
            "dataset": dataset_name,
            "seed": int(seed),
            "round": int(round_idx + 1),
            "runtime": str(metrics.get("runtime", "sync")),
            "strategy": str(metrics.get("strategy", "baseline")),
            "topology": "server-client",
            "server_role": "global-student",
            "client_role": "local-teacher",
            "server_model": "resnet18",
            "client_model": "resnet18",
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
            "niabd_memory_eligible_teachers": int(
                _metric(
                    metrics,
                    "niabd_memory_eligible_teachers",
                    round_idx,
                )
            ),
            "nonfinite_eval_batches": int(
                _metric(metrics, "nonfinite_eval_batches", round_idx)
            ),
            "nonfinite_distill_rollbacks": int(
                _metric(metrics, "nonfinite_distill_rollbacks", round_idx)
            ),
            "numeric_failure_count": float(
                _metric(metrics, "numeric_failure_count", round_idx)
            ),
            "aggregation_time_s": float(
                _metric(metrics, "aggregation_time_s", round_idx)
            ),
            "client_wire_bytes": int(
                _metric(metrics, "client_wire_bytes", round_idx)
            ),
            "selected_clients": int(
                _metric(metrics, "selected_clients", round_idx)
            ),
            "dispatched_clients": int(
                _metric(metrics, "dispatched_clients", round_idx)
            ),
            "busy_skipped_clients": int(
                _metric(metrics, "busy_skipped_clients", round_idx)
            ),
            "offline_clients": int(
                _metric(metrics, "offline_clients", round_idx)
            ),
            "packets_consumed": int(
                _metric(metrics, "packets_consumed", round_idx)
            ),
            "fresh_packets": int(
                _metric(metrics, "fresh_packets", round_idx)
            ),
            "mild_stale_packets": int(
                _metric(metrics, "mild_stale_packets", round_idx)
            ),
            "moderate_stale_packets": int(
                _metric(metrics, "moderate_stale_packets", round_idx)
            ),
            "severe_stale_packets": int(
                _metric(metrics, "severe_stale_packets", round_idx)
            ),
            "mean_version_lag": float(
                _metric(metrics, "mean_version_lag", round_idx, np.nan)
            ),
            "max_version_lag": float(
                _metric(metrics, "max_version_lag", round_idx, np.nan)
            ),
            "mean_knowledge_age_s": float(
                _metric(
                    metrics,
                    "mean_knowledge_age_s",
                    round_idx,
                    np.nan,
                )
            ),
            "max_knowledge_age_s": float(
                _metric(
                    metrics,
                    "max_knowledge_age_s",
                    round_idx,
                    np.nan,
                )
            ),
            "upload_attempt_drop_count": int(
                _metric(
                    metrics,
                    "upload_attempt_drop_count",
                    round_idx,
                )
            ),
            "rpc_timeout_count": int(
                _metric(metrics, "rpc_timeout_count", round_idx)
            ),
            "retry_count": int(
                _metric(metrics, "retry_count", round_idx)
            ),
            "quorum_required": int(
                _metric(metrics, "quorum_required", round_idx)
            ),
            "quorum_reached": int(
                _metric(metrics, "quorum_reached", round_idx)
            ),
            "soft_deadline_s": float(
                _metric(metrics, "soft_deadline_s", round_idx)
            ),
            "hard_deadline_s": float(
                _metric(metrics, "hard_deadline_s", round_idx)
            ),
        }


def _summary_row(
    rows: List[dict],
    runtime_events: List[dict] | None = None,
) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty run.")
    last = rows[-1]
    summary = {
        "run_uid": last["run_uid"],
        "dataset": last["dataset"],
        "seed": last["seed"],
        "runtime": last["runtime"],
        "strategy": last["strategy"],
        "topology": last["topology"],
        "server_model": last["server_model"],
        "client_model": last["client_model"],
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
        "total_rollbacks": sum(
            int(row["nonfinite_distill_rollbacks"]) for row in rows
        ),
        "total_numeric_failures": max(
            float(row["numeric_failure_count"]) for row in rows
        ),
        "total_client_wire_bytes": sum(
            int(row["client_wire_bytes"]) for row in rows
        ),
        "total_packets_consumed": sum(
            int(row["packets_consumed"]) for row in rows
        ),
        "total_stale_packets": sum(
            int(row["mild_stale_packets"])
            + int(row["moderate_stale_packets"])
            + int(row["severe_stale_packets"])
            for row in rows
        ),
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
                "niabd_memory_eligible": (
                    bool(defense["memory_eligible"])
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
            **event,
        }


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
) -> None:
    os.makedirs(outdir, exist_ok=True)
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
    runtime = str(runtime).lower()
    if runtime not in {"sync", "process-semi-async"}:
        raise ValueError(f"Unsupported runtime={runtime!r}.")
    if runtime == "process-semi-async" and process_config is None:
        raise ValueError("process_config is required for process runtime.")
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
                run_uid = uuid.uuid4().hex[:12]
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
                        )
                    else:
                        assert process_config is not None
                        server_model = build_model(
                            "resnet18",
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
                            attack_plan=attack_plan,
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
                                attack_plan=attack_plan,
                            )
                        )
                    strategy = _strategy_name(
                        enable_vcaa,
                        enable_niabd,
                    )
                    metrics["strategy"] = strategy
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
                    pd.DataFrame(rows).to_csv(
                        round_csv,
                        mode="a",
                        header=not os.path.exists(round_csv),
                        index=False,
                    )
                    pd.DataFrame([_summary_row(
                        rows,
                        list(metrics.get("runtime_events", [])),
                    )]).to_csv(
                        summary_csv,
                        mode="a",
                        header=not os.path.exists(summary_csv),
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
                        pd.DataFrame(admission_rows).to_csv(
                            admission_csv,
                            mode="a",
                            header=not os.path.exists(admission_csv),
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
                        pd.DataFrame(defense_rows).to_csv(
                            defense_csv,
                            mode="a",
                            header=not os.path.exists(defense_csv),
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
                        pd.DataFrame(runtime_rows).to_csv(
                            runtime_event_csv,
                            mode="a",
                            header=not os.path.exists(
                                runtime_event_csv
                            ),
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
                        pd.DataFrame(backdoor_rows).to_csv(
                            backdoor_csv,
                            mode="a",
                            header=not os.path.exists(backdoor_csv),
                            index=False,
                        )
                    print(f"[write] {round_csv}")
                    print(f"[write] {summary_csv}")
                    if admission_rows:
                        print(f"[write] {admission_csv}")
                    if defense_rows:
                        print(f"[write] {defense_csv}")
                    if runtime_rows:
                        print(f"[write] {runtime_event_csv}")
                    if backdoor_rows:
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
    parser.add_argument("--distill-temperature", type=float, default=2.0)
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
    )


if __name__ == "__main__":
    main()
