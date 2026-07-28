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
        )
    if config.attack_type == "dba":
        return apply_dba(
            images,
            size=int(config.trigger_size),
            part=dba_part,
            value=float(config.trigger_value),
        )
    if config.attack_type == "blend":
        return apply_blend(images, alpha=float(config.blend_alpha))
    if config.attack_type == "dynamic":
        return apply_dynamic(
            images,
            size=int(config.trigger_size),
            round_number=int(round_number),
            attack_start_round=int(config.attack_start_round),
            period=int(config.dynamic_period),
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

    if plan.config.attack_type == "none":
        return BASRResult(float("nan"), 0, 0)
    model.eval()
    device_obj = torch.device(device)
    amp_enabled = bool(amp) and device_obj.type == "cuda"
    numerator = 0
    denominator = 0
    for batch in dataloader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("BASR evaluation requires labeled test batches.")
        images, labels = batch[0], batch[1]
        keep = labels.long() != int(plan.config.target_label)
        if not bool(keep.any().item()):
            continue
        images = images[keep].to(device_obj, non_blocking=True)
        images = apply_evaluation_trigger(
            images,
            plan=plan,
            round_number=int(round_number),
            dba_part=dba_part,
        )
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = _forward_logits(model, images)
        else:
            logits = _forward_logits(model, images)
        prediction = logits.argmax(dim=1).detach().cpu()
        numerator += int(
            (prediction == int(plan.config.target_label)).sum().item()
        )
        denominator += int(keep.sum().item())
    return BASRResult(
        float(numerator) / float(max(denominator, 1)),
        int(numerator),
        int(denominator),
    )


def evaluate_backdoor_suite(
    model,
    dataloader,
    *,
    device,
    plan: AttackPlan,
    round_number: int,
    amp: bool = False,
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
    }
    if plan.config.attack_type == "none":
        return result
    global_result = evaluate_basr(
        model,
        dataloader,
        device=device,
        plan=plan,
        round_number=int(round_number),
        dba_part=None,
        amp=bool(amp),
    )
    result.update({
        "basr_global": float(global_result.basr),
        "basr_global_numerator": int(global_result.numerator),
        "basr_global_denominator": int(global_result.denominator),
    })
    if plan.config.attack_type == "dba":
        for part in range(4):
            local = evaluate_basr(
                model,
                dataloader,
                device=device,
                plan=plan,
                round_number=int(round_number),
                dba_part=part,
                amp=bool(amp),
            )
            result[f"basr_local_{part + 1}"] = float(local.basr)
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
