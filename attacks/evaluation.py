from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import torch

from attacks.attack_plan import AttackPlan
from attacks.trigger import (
    apply_badnets,
    apply_blend,
    apply_dba,
    apply_dynamic,
)


@dataclass(frozen=True)
class BASRResult:
    basr: float
    numerator: int
    denominator: int


@dataclass(frozen=True)
class AccuracyResult:
    """Invalid evaluations have a NaN value; their counts are diagnostic only."""

    value: float
    numerator: int
    denominator: int
    valid: bool
    nonfinite_batches: int = 0
    reason: str = ""


def _forward_logits(model, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return torch.nan_to_num(
        output,
        nan=0.0,
        posinf=30.0,
        neginf=-30.0,
    ).clamp(-30.0, 30.0)


def apply_evaluation_trigger(
    images: torch.Tensor,
    *,
    plan: AttackPlan,
    round_number: int,
    dba_part: int | None = None,
) -> torch.Tensor:
    config = plan.config
    if config.attack_type == "badnets":
        return apply_badnets(
            images,
            size=int(config.trigger_size),
            value=float(config.trigger_value),
            dataset_name=plan.dataset_name,
        )
    if config.attack_type == "dba":
        return apply_dba(
            images,
            size=int(config.trigger_size),
            part=dba_part,
            value=float(config.trigger_value),
            dataset_name=plan.dataset_name,
        )
    if config.attack_type == "blend":
        return apply_blend(
            images,
            alpha=float(config.blend_alpha),
            dataset_name=plan.dataset_name,
        )
    if config.attack_type == "dynamic":
        return apply_dynamic(
            images,
            size=int(config.trigger_size),
            round_number=int(round_number),
            attack_start_round=int(config.attack_start_round),
            period=int(config.dynamic_period),
            dataset_name=plan.dataset_name,
        )
    if config.attack_type == "none":
        return images
    raise ValueError(f"Unsupported attack_type={config.attack_type!r}.")


@torch.no_grad()
def evaluate_basr(
    model,
    dataloader,
    *,
    device,
    plan: AttackPlan,
    round_number: int,
    dba_part: int | None = None,
    amp: bool = False,
) -> BASRResult:
    """Evaluate backdoor ASR, excluding samples already in the target class."""

    result = evaluate_aa(
        model,
        dataloader,
        device=device,
        plan=plan,
        round_number=int(round_number),
        dba_part=dba_part,
        amp=bool(amp),
        strict_numeric_checks=False,
    )
    return BASRResult(result.value, result.numerator, result.denominator)


def _evaluation_logits(
    model,
    images: torch.Tensor,
    *,
    device_obj: torch.device,
    amp_enabled: bool,
    strict_numeric_checks: bool,
) -> tuple[torch.Tensor | None, bool]:
    if amp_enabled:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
    else:
        logits = model(images)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    finite = bool(torch.isfinite(logits).all().item())
    if not finite and strict_numeric_checks:
        return None, False
    if not finite:
        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        ).clamp(-30.0, 30.0)
    return logits, finite


@torch.no_grad()
def evaluate_ta(
    model,
    dataloader,
    *,
    device,
    amp: bool = False,
    strict_numeric_checks: bool = False,
) -> AccuracyResult:
    """Evaluate clean test accuracy without hiding non-finite logits."""

    model.eval()
    device_obj = torch.device(device)
    amp_enabled = bool(amp) and device_obj.type == "cuda"
    numerator = 0
    denominator = 0
    nonfinite_batches = 0
    for batch in dataloader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("TA evaluation requires labeled test batches.")
        images, labels = batch[0], batch[1]
        images = images.to(device_obj, non_blocking=True)
        labels = labels.to(device_obj, non_blocking=True).long()
        logits, finite = _evaluation_logits(
            model,
            images,
            device_obj=device_obj,
            amp_enabled=amp_enabled,
            strict_numeric_checks=bool(strict_numeric_checks),
        )
        if not finite:
            nonfinite_batches += 1
        if logits is None:
            continue
        numerator += int((logits.argmax(dim=1) == labels).sum().item())
        denominator += int(labels.numel())
    valid = denominator > 0 and nonfinite_batches == 0
    return AccuracyResult(
        value=(float(numerator) / float(denominator) if valid else float("nan")),
        numerator=int(numerator),
        denominator=int(denominator),
        valid=bool(valid),
        nonfinite_batches=int(nonfinite_batches),
        reason=("nonfinite_logits" if nonfinite_batches else "" if valid else "empty_test_set"),
    )


@torch.no_grad()
def evaluate_aa(
    model,
    dataloader,
    *,
    device,
    plan: AttackPlan,
    round_number: int,
    dba_part: int | None = None,
    amp: bool = False,
    strict_numeric_checks: bool = False,
) -> AccuracyResult:
    """Evaluate trigger AA on non-target test labels only."""

    if plan.config.attack_type == "none":
        return AccuracyResult(
            value=float("nan"),
            numerator=0,
            denominator=0,
            valid=False,
            reason="not_applicable_clean_run",
        )
    model.eval()
    device_obj = torch.device(device)
    amp_enabled = bool(amp) and device_obj.type == "cuda"
    numerator = 0
    denominator = 0
    nonfinite_batches = 0
    for batch in dataloader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("AA evaluation requires labeled test batches.")
        images, labels = batch[0], batch[1]
        labels = labels.long()
        keep = labels != int(plan.config.target_label)
        if not bool(keep.any().item()):
            continue
        images = images[keep].to(device_obj, non_blocking=True)
        images = apply_evaluation_trigger(
            images,
            plan=plan,
            round_number=int(round_number),
            dba_part=dba_part,
        )
        logits, finite = _evaluation_logits(
            model,
            images,
            device_obj=device_obj,
            amp_enabled=amp_enabled,
            strict_numeric_checks=bool(strict_numeric_checks),
        )
        if not finite:
            nonfinite_batches += 1
        if logits is None:
            continue
        numerator += int(
            (logits.argmax(dim=1) == int(plan.config.target_label)).sum().item()
        )
        denominator += int(keep.sum().item())
    valid = denominator > 0 and nonfinite_batches == 0
    return AccuracyResult(
        value=(float(numerator) / float(denominator) if valid else float("nan")),
        numerator=int(numerator),
        denominator=int(denominator),
        valid=bool(valid),
        nonfinite_batches=int(nonfinite_batches),
        reason=("nonfinite_logits" if nonfinite_batches else "" if valid else "no_eligible_samples"),
    )


@torch.no_grad()
def evaluate_clean_target_rate(
    model,
    dataloader,
    *,
    device,
    target_label: int,
    amp: bool = False,
    strict_numeric_checks: bool = False,
) -> AccuracyResult:
    """Target predictions on the same non-target test population, without a trigger."""

    model.eval()
    device_obj = torch.device(device)
    amp_enabled = bool(amp) and device_obj.type == "cuda"
    numerator = 0
    denominator = 0
    nonfinite_batches = 0
    for batch in dataloader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Clean target-rate evaluation requires labeled test batches.")
        images, labels = batch[0], batch[1].long()
        keep = labels != int(target_label)
        if not bool(keep.any().item()):
            continue
        images = images[keep].to(device_obj, non_blocking=True)
        logits, finite = _evaluation_logits(
            model,
            images,
            device_obj=device_obj,
            amp_enabled=amp_enabled,
            strict_numeric_checks=bool(strict_numeric_checks),
        )
        if not finite:
            nonfinite_batches += 1
        if logits is None:
            continue
        numerator += int((logits.argmax(dim=1) == int(target_label)).sum().item())
        denominator += int(keep.sum().item())
    valid = denominator > 0 and nonfinite_batches == 0
    return AccuracyResult(
        value=(float(numerator) / float(denominator) if valid else float("nan")),
        numerator=int(numerator),
        denominator=int(denominator),
        valid=bool(valid),
        nonfinite_batches=int(nonfinite_batches),
        reason=("nonfinite_logits" if nonfinite_batches else "" if valid else "no_eligible_samples"),
    )


def evaluate_backdoor_suite(
    model,
    dataloader,
    *,
    device,
    plan: AttackPlan,
    round_number: int,
    amp: bool = False,
    strict_numeric_checks: bool = False,
) -> dict[str, float | int]:
    """Evaluate the global trigger plus all DBA local triggers when relevant."""

    result: dict[str, float | int] = {
        "basr_global": float("nan"),
        "basr_global_numerator": 0,
        "basr_global_denominator": 0,
        "basr_local_1": float("nan"),
        "basr_local_2": float("nan"),
        "basr_local_3": float("nan"),
        "basr_local_4": float("nan"),
        "aa_valid": False,
        "aa_nonfinite_batches": 0,
        "aa_invalid_reason": "",
        "clean_target_rate": float("nan"),
        "clean_target_numerator": 0,
        "clean_target_denominator": 0,
        "clean_target_valid": False,
        "trigger_lift": float("nan"),
    }
    if plan.config.attack_type == "none":
        result["aa_invalid_reason"] = "not_applicable_clean_run"
        return result
    # Round metrics measure the attacked model after poisoning has begun.
    # Pre-onset triggered diagnostics are available through evaluate_aa, but
    # must not be presented as FL attack-phase AA or cause test-trigger work.
    if int(round_number) < int(plan.config.attack_start_round):
        result["aa_invalid_reason"] = "attack_not_started"
        return result
    global_result = evaluate_aa(
        model,
        dataloader,
        device=device,
        plan=plan,
        round_number=int(round_number),
        dba_part=None,
        amp=bool(amp),
        strict_numeric_checks=bool(strict_numeric_checks),
    )
    result.update({
        "basr_global": float(global_result.value),
        "basr_global_numerator": int(global_result.numerator),
        "basr_global_denominator": int(global_result.denominator),
        "aa_valid": bool(global_result.valid),
        "aa_nonfinite_batches": int(global_result.nonfinite_batches),
        "aa_invalid_reason": str(global_result.reason),
    })
    clean_result = evaluate_clean_target_rate(
        model,
        dataloader,
        device=device,
        target_label=int(plan.config.target_label),
        amp=bool(amp),
        strict_numeric_checks=bool(strict_numeric_checks),
    )
    result.update({
        "clean_target_rate": float(clean_result.value),
        "clean_target_numerator": int(clean_result.numerator),
        "clean_target_denominator": int(clean_result.denominator),
        "clean_target_valid": bool(clean_result.valid),
        "trigger_lift": (
            float(global_result.value - clean_result.value)
            if global_result.valid
            and clean_result.valid
            and global_result.denominator == clean_result.denominator
            else float("nan")
        ),
    })
    if plan.config.attack_type == "dba":
        for part in range(4):
            local = evaluate_aa(
                model,
                dataloader,
                device=device,
                plan=plan,
                round_number=int(round_number),
                dba_part=part,
                amp=bool(amp),
                strict_numeric_checks=bool(strict_numeric_checks),
            )
            result[f"basr_local_{part + 1}"] = float(local.value)
    return result


def split_defense_diagnostics(
    records: Sequence[Mapping[str, object]],
    *,
    malicious_client_ids: Iterable[int],
) -> dict[str, float]:
    """Compute experiment-only malicious/benign NIABD diagnostics.

    Ground-truth identities are joined *after* NIABD has returned its records;
    they are never passed into the defense controller.
    """

    malicious = {int(x) for x in malicious_client_ids}
    malicious_records = [
        record for record in records
        if int(record["client_id"]) in malicious
    ]
    benign_records = [
        record for record in records
        if int(record["client_id"]) not in malicious
    ]

    def mean(group, key: str) -> float:
        if not group:
            return float("nan")
        return float(sum(float(row[key]) for row in group) / len(group))

    def eligible_rate(group) -> float:
        if not group:
            return float("nan")
        return float(
            sum(bool(row["memory_eligible"]) for row in group) / len(group)
        )

    return {
        "malicious_mean_anomaly_fraction": mean(
            malicious_records, "anomaly_fraction"
        ),
        "benign_mean_anomaly_fraction": mean(
            benign_records, "anomaly_fraction"
        ),
        "malicious_mean_suppression": mean(
            malicious_records, "mean_suppression"
        ),
        "benign_mean_suppression": mean(
            benign_records, "mean_suppression"
        ),
        "malicious_memory_eligible_rate": eligible_rate(
            malicious_records
        ),
        "benign_memory_eligible_rate": eligible_rate(benign_records),
    }
